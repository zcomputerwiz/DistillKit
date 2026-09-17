"""The production recipient initializer, on the real module and the real model.

``test_gr_reference`` establishes that the *algebra* is right, against an independently
written float64 statement of it. That leaves the thing the conversion will actually be
run with untested: the module whose gains are a deviation on a deviation, whose norm may
already be on CUDA, and whose parameters may already have been cast. The checks reported
by hand in ``scratch/gr_retrofit`` live here instead, so they are rerun rather than
remembered.

The two failure modes this file exists for are both silent. A norm that is already on
CUDA against branch scales built on the CPU raises, so that one is merely broken; a
route that has already been cast to bf16 rounds ``1 + 2 * weight`` on the way in and
then sits bit-frozen against an AdamW step, which is not.
"""
import copy
import sys
from pathlib import Path

import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5RMSNorm

from distillkit.experimental.hyper_connection import DEFAULT_EPSILON, HyperConnection
from distillkit.models import Qwen35WidenedForCausalLM
from test_widened_residual import tiny_config, sample

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "gr_retrofit"))
import reference as independent  # noqa: E402

HIDDEN, BRANCHES, LOWRANK, EPS = 16, 4, 8, 1e-6


def converted(asymmetric=False, seed=3, dtype=torch.float32, device="cpu"):
    """A route initialized against a norm with nonunit, nonuniform weights."""
    torch.manual_seed(11)
    route = HyperConnection(HIDDEN, BRANCHES, LOWRANK, norm_eps=EPS).to(device=device,
                                                                        dtype=dtype)
    norm = Qwen3_5RMSNorm(HIDDEN, eps=EPS).to(device=device, dtype=dtype)
    with torch.no_grad():
        norm.weight.uniform_(-.4, .5)
    record = route.recipient_initialize(norm, asymmetric=asymmetric, seed=seed)
    return route, norm, record


def states(route, batch=2, length=5, dtype=torch.float32, device="cpu"):
    hidden = torch.randn(batch, length, HIDDEN, dtype=dtype, device=device) * 3
    return hidden, hidden.unsqueeze(-2).repeat(1, 1, route.num_branches, 1)


class TestTheModuleReproducesTheRecipient:
    """What `scratch/gr_retrofit` reported by hand, as assertions."""

    def test_symmetric_read_is_bitwise_the_original_norm(self):
        route, norm, _ = converted()
        hidden, branches = states(route)
        read, _weights = route.read(branches, norm)
        assert torch.equal(read, norm(hidden))

    def test_asymmetric_read_is_within_a_rounding_step(self):
        route, norm, _ = converted(asymmetric=True)
        hidden, branches = states(route)
        read, _weights = route.read(branches, norm)
        # Four perturbed gains only average to 2*gamma after summation, so this mode
        # converts to within a rounding step rather than exactly. Not a defect, but the
        # reason the symmetric mode is the primary candidate.
        assert not torch.equal(read, norm(hidden))
        torch.testing.assert_close(read, norm(hidden), atol=1e-6, rtol=0)

    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_write_weights_are_exactly_one_and_branches_stay_equal(self, asymmetric):
        route, norm, _ = converted(asymmetric=asymmetric)
        hidden, branches = states(route)
        read, weights = route.read(branches, norm)
        assert torch.equal(weights, torch.ones_like(weights))
        update = torch.randn_like(read)
        written = route.write(branches, update, weights)
        spread = (written - written[..., :1, :]).detach().abs().max()
        assert float(spread) == 0.0

    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_the_collapse_returns_the_recipients_own_sublayer(self, asymmetric):
        route, norm, _ = converted(asymmetric=asymmetric)
        hidden, branches = states(route)
        read, weights = route.read(branches, norm)
        update = torch.randn_like(read)
        written = route.write(branches, update, weights)
        assert torch.equal(written.mean(dim=-2), hidden + update)

    def test_the_read_bottleneck_keeps_a_gradient_path(self):
        route, _norm, _ = converted()
        assert float(route.W_up.weight.abs().max()) == 0.0
        assert float(route.W_write.weight.abs().max()) == 0.0
        assert float(route.W_down.weight.abs().max()) > 0.0

    def test_the_stored_delta_is_one_plus_twice_the_norm_weight(self):
        route, norm, _ = converted()
        # The trap: both the norm and the route store a deviation, so `2 * weight`
        # would be wrong by exactly one on every branch.
        expected = (1.0 + 2.0 * norm.weight.detach()).unsqueeze(0).expand(BRANCHES, -1)
        torch.testing.assert_close(route.branch_gain_delta, expected)


class TestItAgreesWithTheIndependentReference:
    """Same weights, two implementations, one of which imports nothing from distillkit."""

    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_the_read_matches_the_reference_given_identical_weights(self, asymmetric):
        route, norm, _ = converted(asymmetric=asymmetric)
        hidden, branches = states(route)
        read, _weights = route.read(branches, norm)
        gains = independent.branch_gains(
            (1.0 + norm.weight.detach()).double(), BRANCHES, asymmetric)
        # `gr_sublayer` returns written states, not the read. With the identity as the
        # sublayer and write multipliers of one, every branch is offset by exactly the
        # read, so the reference hands it back without being reimplemented here.
        written = independent.gr_sublayer(
            branches.double(), gains, route.W_down.weight.detach().double(),
            route.W_up.weight.detach().double(), route.W_write.weight.detach().double(),
            EPS, lambda x: x)
        expected = (written - branches.double())[..., 0, :]
        torch.testing.assert_close(read.detach().double(), expected, atol=1e-6, rtol=0)


class TestPlacementAndCasting:
    """Conversion before placement, after placement, and either side of a cast."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_conversion_after_placement_on_cuda(self):
        route, norm, _ = converted(device="cuda")
        assert route.branch_gain_delta.device.type == "cuda"
        hidden, branches = states(route, dtype=torch.float32, device="cuda")
        read, _weights = route.read(branches, norm)
        assert torch.equal(read, norm(hidden))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_conversion_before_placement_survives_the_move(self):
        route, norm, _ = converted()
        before = route.branch_gain_delta.detach().clone()
        route, norm = route.cuda(), norm.cuda()
        torch.testing.assert_close(route.branch_gain_delta.cpu(), before)
        hidden, branches = states(route, device="cuda")
        read, _weights = route.read(branches, norm)
        assert torch.equal(read, norm(hidden))

    def test_a_bf16_cast_does_not_round_the_gains(self):
        route, _norm, _ = converted()
        before = route.branch_gain_delta.detach().clone()
        route.bfloat16()
        assert route.branch_gain_delta.dtype == torch.float32
        assert torch.equal(route.branch_gain_delta, before)

    def test_converting_an_already_bf16_route_stores_fp32_gains(self):
        route, norm, _ = converted(dtype=torch.bfloat16)
        assert route.branch_gain_delta.dtype == torch.float32
        # bf16 spacing near 3 is 1.6e-2; the gains must not arrive pre-rounded to it.
        expected = (1.0 + 2.0 * norm.weight.detach().float()).unsqueeze(0)
        torch.testing.assert_close(route.branch_gain_delta, expected.expand(BRANCHES, -1))
        assert not torch.equal(route.branch_gain_delta,
                               route.branch_gain_delta.bfloat16().float())

    def test_a_wider_cast_is_still_honoured(self):
        route, _norm, _ = converted()
        route.double()
        assert route.branch_gain_delta.dtype == torch.float64


class TestItRefusesWhatItCannotConvert:
    def test_it_refuses_a_norm_whose_gain_convention_it_does_not_know(self):
        route = HyperConnection(HIDDEN, BRANCHES, LOWRANK, norm_eps=EPS)
        with pytest.raises(TypeError, match="Qwen3_5RMSNorm"):
            route.recipient_initialize(torch.nn.LayerNorm(HIDDEN))

    def test_it_refuses_a_norm_with_a_different_epsilon(self):
        route = HyperConnection(HIDDEN, BRANCHES, LOWRANK, norm_eps=EPS)
        with pytest.raises(ValueError, match="epsilon"):
            route.recipient_initialize(Qwen3_5RMSNorm(HIDDEN, eps=1e-5))

    def test_it_refuses_a_perturbation_that_moves_the_initial_read(self):
        route = HyperConnection(HIDDEN, BRANCHES, LOWRANK, norm_eps=EPS)
        with pytest.raises(ValueError, match="mean-zero"):
            route.recipient_initialize(Qwen3_5RMSNorm(HIDDEN, eps=EPS), asymmetric=True,
                                       epsilon=[.01, .01, .01, .01])

    def test_it_refuses_a_perturbation_of_the_wrong_length(self):
        route = HyperConnection(HIDDEN, BRANCHES, LOWRANK, norm_eps=EPS)
        with pytest.raises(ValueError, match="entries"):
            route.recipient_initialize(Qwen3_5RMSNorm(HIDDEN, eps=EPS), asymmetric=True,
                                       epsilon=[-.01, .01])


def widened(tmp_path, dtype=torch.float32):
    torch.manual_seed(83)
    stock = Qwen3_5ForCausalLM(tiny_config()).to(dtype).eval()
    with torch.no_grad():
        for name, parameter in stock.named_parameters():
            if "layernorm" in name:
                parameter.uniform_(-.3, .4)
    stock.save_pretrained(tmp_path / "stock")
    config = copy.deepcopy(stock.config)
    config.residual_stream_routing = "flash_next"
    config.residual_stream_num_branches = 4
    model = Qwen35WidenedForCausalLM.from_pretrained(tmp_path / "stock", config=config,
                                                     dtype=dtype).eval()
    return stock, model


class TestTheWholeModelConverts:
    def test_conversion_records_every_sublayer_and_activates_the_route(self, tmp_path):
        _stock, model = widened(tmp_path)
        records = model.recipient_initialize()
        assert len(records) == 2 * model.config.num_hidden_layers
        assert {record["mode"] for record in records} == {"symmetric"}
        # One seed per sublayer, not one for the whole model: identical bottlenecks
        # everywhere would remove the only thing the read can specialize.
        seeds = [record["seed"] for record in records]
        assert len(set(seeds)) == len(seeds)
        assert all(float(route.blend) == 1.0 for route in model.modules()
                   if isinstance(route, HyperConnection))

    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_the_converted_model_still_computes_the_recipients_logits(self, tmp_path,
                                                                      asymmetric):
        stock, model = widened(tmp_path)
        model.recipient_initialize(asymmetric=asymmetric)
        with torch.no_grad():
            expected, produced = stock(sample()).logits, model(sample()).logits
        if asymmetric:
            torch.testing.assert_close(produced, expected, atol=2e-4, rtol=0)
        else:
            assert torch.equal(produced, expected)

    def test_reload_restores_the_gains_and_the_mode_without_reconverting(self, tmp_path):
        _stock, model = widened(tmp_path)
        model.recipient_initialize()
        with torch.no_grad():
            # Stand in for training: the reload must bring these back, not the
            # initialization they started from.
            for route in model.modules():
                if isinstance(route, HyperConnection):
                    route.branch_gain_delta.add_(.01)
                    route.W_up.weight.normal_(0, .02)
        model.save_pretrained(tmp_path / "converted")
        restored = Qwen35WidenedForCausalLM.from_pretrained(tmp_path / "converted").eval()
        assert restored.config.residual_stream_recipient_mode == "symmetric"
        assert restored.config.residual_stream_recipient_epsilon is None
        routes = [m for m in restored.modules() if isinstance(m, HyperConnection)]
        assert routes and all(r.recipient_initialized and r.recipient_mode == "symmetric"
                              for r in routes)
        trained = [m for m in model.modules() if isinstance(m, HyperConnection)]
        for before, after in zip(trained, routes):
            assert torch.equal(before.branch_gain_delta, after.branch_gain_delta)
            assert torch.equal(before.W_up.weight, after.W_up.weight)
        with torch.no_grad():
            assert torch.equal(model(sample()).logits, restored(sample()).logits)

    def test_a_reloaded_conversion_refuses_to_be_converted_again(self, tmp_path):
        _stock, model = widened(tmp_path)
        model.recipient_initialize()
        model.save_pretrained(tmp_path / "converted")
        restored = Qwen35WidenedForCausalLM.from_pretrained(tmp_path / "converted")
        with pytest.raises(ValueError, match="already records"):
            restored.recipient_initialize()

    def test_the_asymmetric_perturbation_is_recorded_for_reload(self, tmp_path):
        _stock, model = widened(tmp_path)
        model.recipient_initialize(asymmetric=True)
        assert model.config.residual_stream_recipient_mode == "asymmetric"
        torch.testing.assert_close(
            torch.tensor(model.config.residual_stream_recipient_epsilon),
            DEFAULT_EPSILON, atol=1e-6, rtol=0)

    def test_it_refuses_a_model_that_is_not_on_the_flash_next_route(self, tmp_path):
        torch.manual_seed(5)
        model = Qwen35WidenedForCausalLM(tiny_config(residual_stream_num_branches=4))
        with pytest.raises(ValueError, match="flash_next"):
            model.recipient_initialize()
