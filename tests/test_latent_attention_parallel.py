"""Sharding MLA and CSA2 by head, and the two ways it fails silently.

Splitting attention by head is exact: the heads are independent and a concatenation of
halves is the whole. That is also what makes a mistake here invisible. A cut that lands
inside a head still produces a tensor of the right shape, still trains, still reports a
falling loss -- it has simply handed one rank a head's query and the other its gate.
Qwen3.5 packs that gate into `q_proj`, so the layout is the whole argument for why a
contiguous split is allowed at all, and it is worth a test rather than a comment.

The second silent failure is the router. It decides which positions every head reads, so
the ranks must agree on it exactly; if it were split, the halves could reach different
top-k sets wherever the indexer's scores tie at the cutoff -- and on the real checkpoint
they tie for up to 10.2% of queries. The heads would then attend to different histories
and the concatenation would be meaningless. So the router is replicated, and that is
checked by where its parameters live.
"""

import pytest
import torch
from torch import nn

from distillkit.parallel.latent_attention import (GatheredColumnLinear,
                                                  ReducedRowLinear,
                                                  is_latent_attention,
                                                  shard_latent_attention)

TWO_GPUS = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two CUDA devices"
)


def test_gathered_column_linear_matches_its_source_on_cpu():
    """Same arithmetic, parameters merely living elsewhere."""
    source = nn.Linear(16, 8)
    sharded = GatheredColumnLinear(source, ["cpu", "cpu"])
    x = torch.randn(3, 5, 16)
    torch.testing.assert_close(sharded(x), source(x))


def test_reduced_row_linear_matches_its_source_on_cpu():
    source = nn.Linear(16, 8)
    sharded = ReducedRowLinear(source, ["cpu", "cpu"])
    x = torch.randn(3, 5, 16)
    torch.testing.assert_close(sharded(x), source(x), rtol=1e-5, atol=1e-6)


def test_the_assembled_weight_reproduces_the_source():
    """`_project` fuses q_proj by reading `.weight`, and `_attend_absorbed` reads
    `kv_b_proj.weight` directly. Both need the assembled tensor to be the original."""
    source = nn.Linear(16, 8)
    column = GatheredColumnLinear(source, ["cpu", "cpu"])
    torch.testing.assert_close(column.weight, source.weight)
    torch.testing.assert_close(column.bias, source.bias)
    row = ReducedRowLinear(source, ["cpu", "cpu"])
    torch.testing.assert_close(row.weight, source.weight)


def test_an_unbiased_projection_reports_no_bias():
    """`_project` branches on `modules[0].bias is not None`, so answering with an empty
    container instead of None would send it down the wrong path."""
    sharded = GatheredColumnLinear(nn.Linear(16, 8, bias=False), ["cpu", "cpu"])
    assert sharded.bias is None


def test_a_cut_inside_a_head_is_refused():
    """The failure this guards against runs and trains; it does not raise on its own.

    Eight output channels over two devices is four each, which is fine for a head of
    four and wrong for a head of three.
    """
    source = nn.Linear(16, 8)
    GatheredColumnLinear(source, ["cpu", "cpu"], head_multiple=4)
    with pytest.raises(ValueError, match="inside a head"):
        GatheredColumnLinear(source, ["cpu", "cpu"], head_multiple=3)


def test_latent_attention_is_recognised_by_what_it_has():
    """MLA has no k_proj or v_proj; a borrowing CSA2 layer has no kv_a_proj either."""

    class Latent(nn.Module):
        def __init__(self):
            super().__init__()
            self.kv_b_proj = nn.Linear(4, 4)

    class Stock(nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = nn.Linear(4, 4)
            self.v_proj = nn.Linear(4, 4)

    assert is_latent_attention(Latent())
    assert not is_latent_attention(Stock())


@TWO_GPUS
def test_sharding_a_csa2_model_changes_nothing_it_computes():
    """End to end, in fp32, where the split is exact and rounding cannot hide an error."""
    from distillkit.models import Qwen35WidenedForCausalLM
    from distillkit.parallel.model import shard_model
    from tests.test_csa2_routing import csa2_config

    torch.manual_seed(0)
    config = csa2_config(linear_num_key_heads=2, linear_num_value_heads=2,
                         num_attention_heads=2, num_key_value_heads=2)
    model = Qwen35WidenedForCausalLM(config).to("cuda:0", dtype=torch.float32).eval()
    tokens = torch.randint(1, 64, (2, 128), device="cuda:0")
    with torch.no_grad():
        reference = model(input_ids=tokens, use_cache=False).logits.clone()

    shard_model(model, ["cuda:0", "cuda:1"], shard_embeddings=False)
    # One single-threaded step first: a cold Triton autotune under sharding races its own
    # `nargs` between the two backward threads.
    torch.autograd.set_multithreading_enabled(False)
    try:
        model(input_ids=tokens, use_cache=False).logits.square().mean().backward()
    finally:
        torch.autograd.set_multithreading_enabled(True)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    with torch.no_grad():
        sharded = model(input_ids=tokens, use_cache=False).logits
    scale = reference.abs().max().item()
    gap = (sharded - reference).abs().max().item()
    assert gap / scale < 1e-4, "sharded output diverges: relative %.2e" % (gap / scale)


@TWO_GPUS
def test_the_router_and_the_latent_stay_on_home():
    """Replicated by construction, and the test is where the parameters ended up.

    A split router is the failure that produces two ranks reading different positions,
    which no output comparison on a small toy would reliably catch.
    """
    from distillkit.models import Qwen35WidenedForCausalLM
    from distillkit.parallel.model import shard_model
    from tests.test_csa2_routing import csa2_config

    torch.manual_seed(0)
    config = csa2_config(linear_num_key_heads=2, linear_num_value_heads=2,
                         num_attention_heads=2, num_key_value_heads=2)
    model = Qwen35WidenedForCausalLM(config).to("cuda:0", dtype=torch.float32)
    shard_model(model, ["cuda:0", "cuda:1"], shard_embeddings=False)

    router = [(name, parameter) for name, parameter in model.named_parameters()
              if any(mark in name for mark in
                     ("index_k_proj", "index_q_proj", "indexer_proj", "kv_a_proj"))]
    assert router, "no router parameters found; the test is not testing anything"
    off_home = [name for name, parameter in router if parameter.device.index != 0]
    assert not off_home, "router or latent parameters left home: %s" % off_home[:4]


def test_exported_weights_carry_stock_names_not_shards():
    """A checkpoint saved from a sharded model has to load into an unsharded one.

    `smoke_train --tensor-parallel` called `save_pretrained` on the sharded model and
    wrote `self_attn.q_proj.shards.0` where the plain model wants
    `self_attn.q_proj.weight`. Nothing refused it: the file was written, and loading it
    back reported every plain key as missing and initialised them fresh. Ten hours of
    training produced weights that silently were not the trained ones.
    """
    from types import SimpleNamespace

    from distillkit.parallel.checkpoint import consolidated_state_dict

    source_q, source_o = nn.Linear(16, 8), nn.Linear(8, 16, bias=False)
    model = nn.Module()
    model.q_proj = GatheredColumnLinear(source_q, ["cpu", "cpu"])
    model.o_proj = ReducedRowLinear(source_o, ["cpu", "cpu"])
    model.config = SimpleNamespace(tie_word_embeddings=False)

    exported = consolidated_state_dict(model)
    assert sorted(exported) == ["o_proj.weight", "q_proj.bias", "q_proj.weight"]
    torch.testing.assert_close(exported["q_proj.weight"], source_q.weight)
    torch.testing.assert_close(exported["q_proj.bias"], source_q.bias)
    torch.testing.assert_close(exported["o_proj.weight"], source_o.weight)


def test_a_shard_with_no_merge_rule_is_refused():
    """The failure above was a wrapper `tensor_specs` did not know about, and the
    fallback copies an unknown tensor out under its own name -- right for a tensor that
    was never sharded, catastrophic for one that was. An unmergeable shard must stop the
    export rather than be written."""
    from distillkit.parallel.checkpoint import tensor_specs

    model = nn.Module()
    model.mystery = nn.Module()
    model.mystery.shards = nn.ParameterList(
        [nn.Parameter(torch.zeros(4, 16)), nn.Parameter(torch.zeros(4, 16))])

    with pytest.raises(ValueError, match="does not know how to merge"):
        tensor_specs(model)
