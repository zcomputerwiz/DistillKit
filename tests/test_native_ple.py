"""The student-native n-gram table: geometry, dormancy, gradient flow, ablations, reload.

The donor path reads a frozen 51.2-billion-element capture. This one owns its rows. What
these tests protect is the part that makes the experiment interpretable rather than the
part that makes it run: the hash addressing is still the reference's, the morphed model is
bit-identical to the stock one at load, and the two ablations the first training run needs
-- admission off, and right rows for the wrong context -- work without retraining anything.
"""

import json
from pathlib import Path

import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from distillkit.native_ple import NativePLESidecar, native_hash_config
from distillkit.ngram_hash import NGramHashConfig, NGramHasher
from distillkit.models import Qwen35SidecarForCausalLM
from distillkit.sidecar_collator import SidecarDataCollator
from tests.test_sidecar_model import tiny_config

#: Small enough for a test, built the reference way: 16 distinct primes just above the
#: base, running offsets, padded to a multiple of 128.
TINY_BASE = 97


def native_config(**kwargs):
    values = dict(sidecar_variant="ple", sidecar_table_mode="native",
                  sidecar_ngram_vocab_size_base=TINY_BASE, sidecar_layer_index=1)
    values.update(kwargs)
    return tiny_config(**values)


def build(**kwargs):
    torch.manual_seed(0)
    return Qwen35SidecarForCausalLM(native_config(**kwargs)).eval()


def sidecar_of(model):
    return model.model.layers[model.config.sidecar_layer_index].sidecar


def batch(model, batch_size=2, length=8, seed=7):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (batch_size, length), generator=generator)
    hasher = NGramHasher(native_hash_config(model.config))
    return ids, hasher.row_indices(ids)


# --- A. hash geometry --------------------------------------------------------


def test_scaling_the_base_moves_only_the_address_space():
    """The reference construction is unchanged; only the two scaled dimensions differ."""
    reference = NGramHasher(NGramHashConfig())
    scaled = NGramHasher(NGramHashConfig(ngram_vocab_size_base=131072, ple_embed_dim=1024))

    # Same heads, same order, same multipliers, same padding rule.
    assert scaled.config.ngram_heads == reference.config.ngram_heads == 16
    assert torch.equal(scaled.layer_multipliers, reference.layer_multipliers)
    assert scaled.padded_vocab_size % scaled.config.make_ngram_vocab_size_divisible_by == 0
    # Distinct primes just above the base, offsets running over them.
    assert len(set(scaled.head_vocab_sizes.tolist())) == 16
    assert all(size > 131072 for size in scaled.head_vocab_sizes.tolist())
    assert scaled.head_offsets.tolist() == torch.cat(
        [torch.zeros(1, dtype=torch.long),
         torch.cumsum(scaled.head_vocab_sizes, 0)[:-1]]).tolist()
    assert scaled.total_vocab_size == int(scaled.head_vocab_sizes.sum())
    # And the donor's geometry is untouched by any of it.
    assert reference.total_vocab_size == 320001446
    assert reference.padded_vocab_size == 320001536


# --- B. dimensional correctness ---------------------------------------------


def test_a_1024_wide_student_gets_16_heads_of_64():
    config = NGramHashConfig(ngram_vocab_size_base=131072, ple_embed_dim=1024)
    assert config.ngram_heads == 16
    assert config.head_dim == 64
    assert config.ngram_heads * config.head_dim == 1024


def test_the_table_is_rows_by_head_dim_and_features_are_stream_width():
    model = build(hidden_size=64, sidecar_ple_embed_dim=64)
    sidecar = sidecar_of(model)
    assert sidecar.table.weight.shape == (sidecar.padded_vocab_size, sidecar.head_dim)
    ids, rows = batch(model)
    features = sidecar.features(rows, torch.zeros(*ids.shape, model.config.hidden_size))
    assert features.shape == (*ids.shape, model.config.hidden_size)


def test_an_indivisible_embedding_width_is_refused():
    with pytest.raises(ValueError, match="16 n-gram heads"):
        build(hidden_size=64, sidecar_ple_embed_dim=100)


# --- C. native lookup --------------------------------------------------------


def test_lookup_returns_the_rows_the_hash_addressed():
    model = build(hidden_size=64, sidecar_ple_embed_dim=64)
    sidecar = sidecar_of(model)
    ids, rows = batch(model)
    features = sidecar.features(rows, torch.zeros(*ids.shape, 64))
    expected = sidecar.table.weight[rows.reshape(-1)].reshape(*rows.shape, sidecar.head_dim)
    assert torch.allclose(features, expected.flatten(-2))


def test_rows_outside_the_address_space_are_refused():
    model = build(hidden_size=64, sidecar_ple_embed_dim=64)
    sidecar = sidecar_of(model)
    _, rows = batch(model)
    rows = rows.clone()
    rows[0, 0, 0] = sidecar.padded_vocab_size
    with pytest.raises(ValueError, match="address space"):
        sidecar.features(rows, torch.zeros(*rows.shape[:2], 64))


# --- D. function preservation ------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_morphed_model_is_bit_identical_to_stock_at_load(seed):
    """rho is zero, so the block is dormant however its internals initialised.

    This is the property the outer admission scalar exists for, and it is checked on
    several sequences rather than one: the earlier retrofit bought identity by zeroing
    parts of the reference mechanism itself, which is what this design deliberately
    stopped doing.
    """
    config = native_config(hidden_size=64, sidecar_ple_embed_dim=64)
    torch.manual_seed(11)
    morphed = Qwen35SidecarForCausalLM(config).eval()
    stock = Qwen3_5ForCausalLM(config).eval()
    stock.load_state_dict(
        {name: tensor for name, tensor in morphed.state_dict().items()
         if ".sidecar." not in name}, strict=True)

    ids, rows = batch(morphed, seed=seed)
    with torch.no_grad():
        reference = stock(input_ids=ids).logits
        native = morphed(input_ids=ids, ngram_ids=rows).logits
    assert torch.equal(native, reference)
    # And the internals did *not* start at zero, or the test above would be vacuous.
    sidecar = sidecar_of(morphed)
    assert sidecar.ple.value_proj.weight.abs().max() > 0
    assert sidecar.table.weight.abs().max() > 0


# --- E. rho is what admits the block ----------------------------------------


def test_rho_controls_whether_the_block_contributes():
    model = build(hidden_size=64, sidecar_ple_embed_dim=64)
    ids, rows = batch(model)
    with torch.no_grad():
        dormant = model(input_ids=ids, ngram_ids=rows).logits
        sidecar_of(model).rho.fill_(0.5)
        admitted = model(input_ids=ids, ngram_ids=rows).logits
        sidecar_of(model).rho.zero_()
        closed_again = model(input_ids=ids, ngram_ids=rows).logits
    assert not torch.allclose(admitted, dormant, atol=1e-5)
    # Shutting it recovers the backbone exactly, so the ON/OFF ablation needs no run.
    assert torch.equal(closed_again, dormant)
    with torch.no_grad():
        bypassed = model(input_ids=ids, sidecar_enabled=False).logits
    assert torch.equal(bypassed, dormant)


# --- F. gradient flow --------------------------------------------------------


def test_gradients_reach_rho_the_table_and_the_projections():
    model = build(hidden_size=64, sidecar_ple_embed_dim=64)
    sidecar = sidecar_of(model)
    ids, rows = batch(model)

    # At exactly rho = 0 the table sees no gradient -- the write is multiplied by zero --
    # but rho itself must, or the block could never open. That asymmetry is the design.
    model(input_ids=ids, ngram_ids=rows).logits.float().pow(2).sum().backward()
    assert sidecar.rho.grad.abs().item() > 0
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        sidecar.rho.fill_(0.5)
    model(input_ids=ids, ngram_ids=rows).logits.float().pow(2).sum().backward()
    assert sidecar.rho.grad.abs().item() > 0
    assert sidecar.table.weight.grad.abs().max() > 0
    assert sidecar.ple.value_proj.weight.grad.abs().max() > 0
    assert sidecar.ple.key_proj.weight.grad.abs().max() > 0
    assert sidecar.ple.conv1d.weight.grad.abs().max() > 0
    # Only the rows this batch touched moved, which is what makes the dense table
    # wasteful and is the measurement the sparse question will start from.
    touched = torch.unique(rows)
    moved = sidecar.table.weight.grad.abs().sum(-1) > 0
    assert moved.sum() <= touched.numel()


# --- G. wrong-context ablation ----------------------------------------------


def test_wrong_context_changes_rows_but_not_their_validity():
    model = build(hidden_size=64, sidecar_ple_embed_dim=64)
    sidecar = sidecar_of(model)
    hasher = NGramHasher(native_hash_config(model.config))

    class Passthrough:
        def __call__(self, features):
            ids = torch.stack([torch.as_tensor(f["input_ids"]) for f in features])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    honest = SidecarDataCollator(Passthrough(), hasher=hasher, mode="native")
    rolled = SidecarDataCollator(Passthrough(), hasher=hasher, mode="native",
                                 shuffle_context=3)
    examples = [{"input_ids": list(range(4, 20))}, {"input_ids": list(range(30, 46))}]
    right, wrong = honest(examples)["ngram_ids"], rolled(examples)["ngram_ids"]

    assert right.shape == wrong.shape and right.dtype == wrong.dtype == torch.int64
    assert not torch.equal(right, wrong)
    # Same machinery, same address space, same learned rows -- only the correspondence
    # to this text is gone.
    assert int(wrong.min()) >= 0 and int(wrong.max()) < sidecar.padded_vocab_size
    for head in range(sidecar.ngram_heads):
        low = sidecar.head_offsets[head]
        high = low + sidecar.head_vocab_sizes[head]
        assert int(wrong[..., head].min()) >= low and int(wrong[..., head].max()) < high


def test_native_collation_emits_ids_and_never_donor_bytes():
    class Passthrough:
        def __call__(self, features):
            ids = torch.stack([torch.as_tensor(f["input_ids"]) for f in features])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    collator = SidecarDataCollator(Passthrough(), hasher=NGramHasher(NGramHashConfig(
        ngram_vocab_size_base=TINY_BASE, ple_embed_dim=64)), mode="native")
    produced = collator([{"input_ids": list(range(4, 20))}])
    assert "ngram_ids" in produced and "ngram_raw" not in produced


def test_the_two_input_representations_are_not_interchangeable():
    model = build(hidden_size=64, sidecar_ple_embed_dim=64)
    ids, rows = batch(model)
    raw = torch.zeros(*ids.shape, 16, 18, dtype=torch.uint8)
    with pytest.raises(ValueError, match="ngram_ids is required"):
        model(input_ids=ids)
    with pytest.raises(ValueError, match="does not accept ngram_raw"):
        model(input_ids=ids, ngram_ids=rows, ngram_raw=raw)


# --- H. checkpoint round trip ------------------------------------------------


def test_a_native_checkpoint_is_self_contained(tmp_path):
    model = build(hidden_size=64, sidecar_ple_embed_dim=64)
    sidecar = sidecar_of(model)
    with torch.no_grad():
        sidecar.rho.fill_(0.25)
        sidecar.table.weight[5:9] = torch.arange(4 * sidecar.head_dim,
                                                 dtype=sidecar.table.weight.dtype
                                                 ).reshape(4, sidecar.head_dim)
    ids, rows = batch(model)
    with torch.no_grad():
        before = model(input_ids=ids, ngram_ids=rows).logits

    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["sidecar_table_mode"] == "native"
    assert saved["sidecar_ngram_vocab_size_base"] == TINY_BASE
    # No donor path anywhere in the saved configuration: the rows travel with the model.
    assert not any("gguf" in str(value).lower() for value in saved.values())

    reloaded = Qwen35SidecarForCausalLM.from_pretrained(tmp_path).eval()
    restored = sidecar_of(reloaded)
    assert restored.geometry() == sidecar.geometry()
    assert float(restored.rho) == pytest.approx(0.25)
    assert torch.allclose(restored.table.weight[5:9], sidecar.table.weight[5:9])
    with torch.no_grad():
        after = reloaded(input_ids=ids, ngram_ids=rows).logits
    assert torch.allclose(after, before, atol=1e-6)


def test_the_table_is_identifiable_by_name_for_later_optimizer_grouping():
    model = build(hidden_size=64, sidecar_ple_embed_dim=64)
    names = [name for name, _ in model.named_parameters() if name.endswith("sidecar.table.weight")]
    assert len(names) == 1
    parameter = dict(model.named_parameters())[names[0]]
    assert parameter.requires_grad and parameter.dim() == 2


# --- I. donor regression -----------------------------------------------------


def test_donor_mode_is_unchanged_and_refuses_row_ids():
    from tests.test_sidecar_model import raw_batch

    torch.manual_seed(0)
    model = Qwen35SidecarForCausalLM(tiny_config(sidecar_variant="ple")).eval()
    assert model.config.sidecar_table_mode == "donor"
    ids = torch.randint(0, model.config.vocab_size, (2, 8))
    raw = raw_batch(batch=2, length=8)
    with torch.no_grad():
        assert model(input_ids=ids, ngram_raw=raw).logits.shape[-1] == model.config.vocab_size
    with pytest.raises(ValueError, match="does not accept ngram_ids"):
        model(input_ids=ids, ngram_raw=raw, ngram_ids=torch.zeros(2, 8, 2, dtype=torch.long))


# --- generalised geometry: nothing is pinned to the first implementation's 1024 --------


@pytest.mark.parametrize("hidden_size,expected_row", [(1024, 64), (2048, 128), (2560, 160)])
def test_row_width_is_derived_from_hidden_size(hidden_size, expected_row):
    """The first implementation was built against a constructed 1024-wide config.

    The real student is 2048 wide, so the row width has to come from the model rather
    than from the width that happened to be available when this was written.
    """
    config = NGramHashConfig(ngram_vocab_size_base=131072, ple_embed_dim=hidden_size)
    assert config.ngram_heads == 16
    assert config.head_dim == expected_row
    assert config.head_dim * 16 == hidden_size

    model = build(hidden_size=hidden_size, sidecar_ple_embed_dim=hidden_size,
                  num_attention_heads=8, num_key_value_heads=2,
                  head_dim=hidden_size // 8, linear_key_head_dim=hidden_size // 8,
                  linear_value_head_dim=hidden_size // 8)
    sidecar = sidecar_of(model)
    assert sidecar.head_dim == expected_row
    assert sidecar.table.weight.shape[1] == expected_row
    ids, rows = batch(model)
    features = sidecar.features(rows, torch.zeros(*ids.shape, hidden_size))
    assert features.shape[-1] == hidden_size


CONVERTED_2B = Path("D:/DeepThought/Projects/HybridModel/student-2b-hf/config.json")


@pytest.mark.skipif(not CONVERTED_2B.exists(), reason="converted 2B checkpoint not present")
def test_the_converted_2b_config_produces_the_expected_native_geometry():
    """Config-only, so it costs nothing: the real checkpoint's own numbers."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(CONVERTED_2B.parent, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.sidecar_ngram_vocab_size_base = 131072
    config.sidecar_ple_embed_dim = None

    hash_config = native_hash_config(config)
    assert config.hidden_size == 2048
    assert hash_config.ple_embed_dim == 2048
    assert hash_config.head_dim == 128
    hasher = NGramHasher(hash_config)
    assert hasher.padded_vocab_size == 2099200
    assert hasher.padded_vocab_size * hash_config.head_dim == 268697600


def test_the_chunked_table_norm_is_the_ordinary_norm():
    """Logging must not materialise a gibibyte of fp32 to report one scalar."""
    from distillkit.native_ple import _chunked_norm

    torch.manual_seed(0)
    weight = (torch.randn(300, 8) * 3).to(torch.bfloat16)
    assert _chunked_norm(weight, rows=64) == pytest.approx(
        float(weight.float().norm()), rel=1e-5)
    # One slice and many slices agree, so the accumulator is not losing the tail.
    assert _chunked_norm(weight, rows=4096) == pytest.approx(
        _chunked_norm(weight, rows=7), rel=1e-6)
