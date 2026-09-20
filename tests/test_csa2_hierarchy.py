"""Level one of the indexer hierarchy, and the rotary the reference puts on the router.

DeepSeek-V4.1 picks candidate blocks before it picks positions: one layer publishes a
wide selection and the layers after it choose inside it. It also ropes both the indexer
query and the indexer key, so a block's score carries distance and not only content.

Both are routing decisions, so they are tested on bare layers rather than through a model
forward -- `route` is the whole subject, and going through the block-sparse kernel would
pull Inductor into a test that has nothing to say about it.
"""
import pytest
import torch

from test_csa2_routing import BLOCK, bare_layer, csa2_config

BLOCKS, BATCH = 6, 1
SEQ = BLOCKS * BLOCK


def inputs(layer, seed=0):
    torch.manual_seed(seed)
    hidden = torch.randn(BATCH, SEQ, layer.config.hidden_size)
    width = int(layer.rope_dim)
    position = (torch.ones(BATCH, SEQ, width), torch.zeros(BATCH, SEQ, width))
    return hidden, layer.index_k_proj(hidden), layer.index_q_proj(hidden), position


def eligible_of(layer):
    rows = torch.arange(BLOCKS)
    offsets = rows.view(-1, 1) - rows.view(1, -1)
    return offsets > layer.local_blocks


def test_the_publisher_sets_a_candidate_set_wider_than_its_own_selection():
    layer = bare_layer(csa2_candidate_layer=0, csa2_candidate_k=3 * BLOCK).float()
    hidden, keys, queries, position = inputs(layer)
    with torch.no_grad():
        allowed, _ = layer.route(hidden, keys, queries, position)
    assert layer.last_candidates is not None
    eligible = eligible_of(layer).unsqueeze(0)
    assert ((allowed & eligible) & ~layer.last_candidates).sum() == 0
    assert (layer.last_candidates & eligible).sum() > (allowed & eligible).sum()


def test_a_layer_that_reads_candidates_picks_only_inside_them():
    layer = bare_layer().float()
    hidden, keys, queries, position = inputs(layer)
    eligible = eligible_of(layer)
    torch.manual_seed(1)
    candidates = ((torch.rand(BATCH, BLOCKS, BLOCKS) < 0.4) & eligible) | ~eligible
    with torch.no_grad():
        allowed, _ = layer.route(hidden, keys, queries, position, candidates)
    # The local window and the diagonal open regardless of score, so only the scored part
    # has to respect the candidate set.
    assert ((allowed & eligible.unsqueeze(0)) & ~candidates).sum() == 0


def test_a_layer_before_the_publisher_inherits_nothing():
    layer = bare_layer(csa2_candidate_layer=4, csa2_candidate_k=3 * BLOCK)
    layer.bus = type("Bus", (), {"candidates": torch.zeros(1, dtype=torch.bool)})()
    assert layer._inherited_candidates() is None
    assert not layer.publishes_candidates


def test_a_candidate_set_narrower_than_the_selection_is_refused():
    with pytest.raises(ValueError, match="narrower"):
        bare_layer(csa2_candidate_layer=0, csa2_candidate_k=BLOCK // 2, csa2_top_k=BLOCK)


def test_the_rotary_changes_which_blocks_the_router_picks():
    picks = {}
    for flag in (False, True):
        layer = bare_layer(csa2_rope_index=flag).float()
        hidden, keys, queries, position = inputs(layer)
        with torch.no_grad():
            picks[flag] = layer.route(hidden, keys, queries, position)[0]
    assert not torch.equal(picks[False], picks[True]), \
        "roping the index query and key left the selection identical"


def test_the_rotary_is_off_when_no_positions_are_supplied():
    layer = bare_layer(csa2_rope_index=True).float()
    hidden, keys, queries, _ = inputs(layer)
    plain = bare_layer(csa2_rope_index=False).float()
    plain.load_state_dict(layer.state_dict())
    with torch.no_grad():
        assert torch.equal(layer.route(hidden, keys, queries)[0],
                           plain.route(hidden, keys, queries)[0])


def test_the_config_defaults_leave_the_hierarchy_off():
    config = csa2_config()
    assert getattr(config, "csa2_candidate_layer", -1) == -1
    assert not bare_layer().publishes_candidates
