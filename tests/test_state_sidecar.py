"""Invariants for the state-conditioned hash sidecar.

The claims worth pinning are the ones that would make a null result unreadable: that the
module is an exact identity before training, that the blind and conditioned arms differ in
the gate and nothing else, that forcing ``g = 1`` on a trained module really does bypass
the admission policy, and that the wrong-address control genuinely misaddresses. Each of
those failing silently would turn "the mechanism does not help" into "the harness did not
test the mechanism". No network, no GPU.
"""

import pytest
import torch

from distillkit.experimental.state_sidecar import (
    StateConditionedSidecar, forced_identity, install_state_sidecar,
    remove_state_sidecar)

HIDDEN = 64
SEQ = 24


def rows(batch=2, seq=SEQ, heads=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 1 << 16, (batch, seq, heads), generator=generator)


def hidden(batch=2, seq=SEQ, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, HIDDEN, generator=generator)


@pytest.fixture
def sidecar():
    torch.manual_seed(20260915)
    return StateConditionedSidecar(HIDDEN, memory_dim=16)


class TestIdentity:
    def test_the_correction_is_exactly_zero_before_training(self, sidecar):
        """Zero-initialized output projection: the model must be bitwise stock at step 0,
        which is what makes "did it learn anything" answerable."""
        assert float(sidecar(hidden(), rows()).abs().max()) == 0.0

    def test_both_arms_are_identities_at_initialization(self, sidecar):
        sidecar.state_conditioned = False
        assert float(sidecar(hidden(), rows()).abs().max()) == 0.0

    def test_the_module_moves_once_the_output_projection_does(self, sidecar):
        torch.nn.init.normal_(sidecar.out.weight, std=0.05)
        assert float(sidecar(hidden(), rows()).abs().max()) > 0.0


class TestGate:
    def test_the_blind_arm_admits_everything(self, sidecar):
        sidecar.state_conditioned = False
        sidecar(hidden(), rows(), collect=True)
        assert torch.equal(sidecar.last["gate"], torch.ones_like(sidecar.last["gate"]))
        assert sidecar.last["compatibility"] is None

    def test_the_conditioned_arm_varies_with_the_hidden_state(self, sidecar):
        """The gate must actually read the state; a gate that ignores it would reproduce
        the blind arm while looking like a policy."""
        sidecar(hidden(seed=1), rows(), collect=True)
        first = sidecar.last["gate"].clone()
        sidecar(hidden(seed=2), rows(), collect=True)
        assert not torch.allclose(first, sidecar.last["gate"])

    def test_the_gate_is_insensitive_to_hidden_state_magnitude(self, sidecar):
        """Cosine compatibility, so scaling the state must not change admission.

        An unnormalized score could raise admission by growing ||q||, which is a way to
        learn "always admit" that looks like a learned policy.
        """
        states = hidden()
        sidecar(states, rows(), collect=True)
        plain = sidecar.last["gate"].clone()
        sidecar(states * 7.0, rows(), collect=True)
        assert torch.allclose(plain, sidecar.last["gate"], atol=1e-5)

    def test_forced_identity_bypasses_a_trained_policy(self, sidecar):
        """The parameter-matched ablation: same weights, admission held at 1."""
        torch.nn.init.normal_(sidecar.out.weight, std=0.05)
        torch.nn.init.normal_(sidecar.query.weight, std=0.5)
        learned = sidecar(hidden(), rows())
        with forced_identity(sidecar):
            opened = sidecar(hidden(), rows(), collect=True)
            assert torch.equal(sidecar.last["gate"],
                               torch.ones_like(sidecar.last["gate"]))
        assert not torch.allclose(learned, opened)
        assert sidecar.state_conditioned is True

    def test_gate_stays_in_the_unit_interval(self, sidecar):
        torch.nn.init.normal_(sidecar.query.weight, std=5.0)
        sidecar(hidden(), rows(), collect=True)
        gate = sidecar.last["gate"]
        assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0


class TestAddressing:
    def test_different_contexts_give_different_corrections(self, sidecar):
        torch.nn.init.normal_(sidecar.out.weight, std=0.05)
        first = sidecar(hidden(), rows(seed=0))
        second = sidecar(hidden(), rows(seed=5))
        assert not torch.allclose(first, second)

    def test_both_hash_heads_are_used(self, sidecar):
        """The canonical hasher emits two row ids per position. Consuming only one would
        silently address half the context and still produce plausible numbers."""
        base = rows()
        changed = base.clone()
        changed[..., 1] += 1                      # perturb the second head only
        torch.nn.init.normal_(sidecar.out.weight, std=0.05)
        assert not torch.allclose(sidecar(hidden(), base), sidecar(hidden(), changed))

    def test_the_head_count_is_checked(self, sidecar):
        with pytest.raises(ValueError):
            sidecar(hidden(), rows(heads=1))

    def test_wrong_context_reads_a_different_real_row(self):
        """The control must misaddress, and must still address something real: the
        question is whether the gain needs *correct* context, not any context."""
        class Hasher:
            def row_indices(self, ids):
                return (torch.arange(ids.shape[1]).unsqueeze(0).unsqueeze(-1)
                        .repeat(ids.shape[0], 1, 2))

        torch.manual_seed(0)
        module = StateConditionedSidecar(HIDDEN, memory_dim=16)
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
        handle = install_state_sidecar(model, module, Hasher(), 1)
        ids = torch.zeros((1, SEQ), dtype=torch.long)
        handle.set_context(ids)
        honest = handle.rows.clone()
        module.wrong_context = True
        handle.set_context(ids)
        assert not torch.equal(honest, handle.rows)
        assert sorted(handle.rows[0, :, 0].tolist()) == sorted(honest[0, :, 0].tolist())
        remove_state_sidecar(model)


class TestInjection:
    def test_exactly_one_layer_is_hooked(self):
        torch.manual_seed(0)
        module = StateConditionedSidecar(HIDDEN, memory_dim=16)

        class Hasher:
            def row_indices(self, ids):
                return torch.zeros(ids.shape + (2,), dtype=torch.long)

        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
        handle = install_state_sidecar(model, module, Hasher(), 2)
        hooked = [index for index, layer in enumerate(model.layers)
                  if layer._forward_hooks]
        assert hooked == [2]
        remove_state_sidecar(model)
        assert not any(layer._forward_hooks for layer in model.layers)

    def test_an_out_of_range_layer_is_refused(self):
        torch.manual_seed(0)
        module = StateConditionedSidecar(HIDDEN, memory_dim=16)
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(2)])
        with pytest.raises(ValueError):
            install_state_sidecar(model, module, None, 9)

    def test_a_forward_without_context_is_refused(self):
        """Stale hash rows are the failure that produced a wrong number once already:
        the scorer bound the handle under a different attribute and the module kept
        rows from the previous training batch."""
        torch.manual_seed(0)
        module = StateConditionedSidecar(HIDDEN, memory_dim=16)
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Identity()])
        handle = install_state_sidecar(model, module, None, 0)
        with pytest.raises(ValueError):
            handle.correction(hidden())
        remove_state_sidecar(model)


class TestParameters:
    def test_the_gate_only_count_excludes_the_value_path(self, sidecar):
        report = sidecar.parameter_report()
        assert report["gate_only"] == (report["query"] + report["key"]
                                       + report["norm_scale"] + 2)
        assert report["total"] > report["gate_only"]

    def test_the_code_width_covers_both_heads(self, sidecar):
        report = sidecar.parameter_report()
        assert report["code_width"] == report["heads"] * report["code_dim"]
        assert sidecar.memory.weight.shape[1] == report["code_width"]
