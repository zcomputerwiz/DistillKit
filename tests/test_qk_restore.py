"""QK-Restore rows: which parts of an MLA+CSA2 attention layer are its query and key maps.

`merge_weights --restore-qk` takes these from the base checkpoint (arXiv 2606.11052). A
head's `q_proj` rows are its query then its output gate, `kv_a_proj` is the latent then the
rotary key, and `kv_b_proj` is each head's content key then its value; the CSA2 indexer
and `q_norm` are restored whole, and nothing outside the attention layers is touched.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))

from merge_weights import qk_rows  # noqa: E402

# Two heads of 4 dimensions, a quarter of them rotary (1 rotary, 3 content), latent 6.
CONFIG = SimpleNamespace(head_dim=4, hidden_size=8, num_attention_heads=2, mla_latent_dim=6,
                         rope_parameters={"partial_rotary_factor": 0.25})
ATTN = "model.layers.3.self_attn."


def rows(name, count):
    mask = qk_rows(ATTN + name, (count, 8), CONFIG)
    return mask if mask is None or mask is True else mask.int().tolist()


def test_query_rows_skip_the_gate():
    assert rows("q_proj.weight", 16) == [1, 1, 1, 1, 0, 0, 0, 0] * 2


def test_key_rows_are_the_rotary_key_and_content_keys():
    assert rows("kv_a_proj.weight", 7) == [0] * 6 + [1]
    assert rows("kv_b_proj.weight", 14) == [1, 1, 1, 0, 0, 0, 0] * 2


def test_router_and_query_norm_whole_values_and_rest_untouched():
    for name in ("index_q_proj.weight", "index_k_proj.weight", "indexer_proj.weight",
                 "index_gate", "q_norm.weight"):
        assert rows(name, 4) is True, name
    for name in ("o_proj.weight", "kv_a_norm.weight"):
        assert rows(name, 8) is None, name
    assert qk_rows("model.layers.3.mlp.up_proj.weight", (8, 8), CONFIG) is None
    assert qk_rows("model.layers.0.linear_attn.in_proj_qkv.weight", (8, 8), CONFIG) is None


def test_layout_mismatch_refuses():
    try:
        qk_rows(ATTN + "q_proj.weight", (15, 8), CONFIG)
    except SystemExit:
        return
    raise AssertionError("a q_proj that is not whole heads should refuse")
