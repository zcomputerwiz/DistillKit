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

from test_csa2_routing import BLOCK, bare_layer, csa2_config, index_keys_for

BLOCKS, BATCH = 6, 1
SEQ = BLOCKS * BLOCK


def inputs(layer, seed=0):
    torch.manual_seed(seed)
    hidden = torch.randn(BATCH, SEQ, layer.config.hidden_size)
    width = int(layer.rope_dim)
    position = (torch.ones(BATCH, SEQ, width), torch.zeros(BATCH, SEQ, width))
    return (hidden, index_keys_for(layer, hidden), layer.index_q_proj(hidden), position)


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
    # Level one publishes indices, not a mask: three candidate blocks per query block.
    index = torch.stack([torch.randperm(BLOCKS)[:3] for _ in range(BLOCKS)]).unsqueeze(0)
    with torch.no_grad():
        allowed, _ = layer.route(hidden, keys, queries, position, index)
    inside = torch.zeros(BATCH, BLOCKS, BLOCKS, dtype=torch.bool)
    inside.scatter_(-1, index, True)
    # The local window and the diagonal open regardless of score, so only the scored part
    # has to respect the candidate set.
    assert ((allowed & eligible.unsqueeze(0)) & ~inside).sum() == 0


def test_reading_candidates_scores_only_those_blocks():
    # The point of level one is saving the product, not masking it afterwards, so the
    # scores outside the candidate set must never have been computed.
    layer = bare_layer().float()
    hidden, keys, queries, position = inputs(layer)
    torch.manual_seed(2)
    index = torch.stack([torch.randperm(BLOCKS)[:2] for _ in range(BLOCKS)]).unsqueeze(0)
    roped_q, roped_k, weights = layer._index_inputs(hidden, keys, queries, position)
    with torch.no_grad():
        narrow, _ = layer._score_blocks(roped_q, roped_k, weights, BLOCKS, index)
        full, _ = layer._score_blocks(roped_q, roped_k, weights, BLOCKS)
    inside = torch.zeros(BATCH, BLOCKS, BLOCKS, dtype=torch.bool)
    inside.scatter_(-1, index, True)
    assert torch.isinf(narrow[~inside]).all() and (narrow[~inside] < 0).all()
    torch.testing.assert_close(narrow[inside], full[inside], atol=1e-5, rtol=1e-5)


def test_a_layer_before_the_publisher_inherits_nothing():
    layer = bare_layer(csa2_candidate_layer=4, csa2_candidate_k=3 * BLOCK)
    layer.bus = type("Bus", (), {"candidates": torch.zeros(1, dtype=torch.bool)})()
    assert layer._inherited_candidates() is None
    assert not layer.publishes_candidates


def test_a_candidate_set_narrower_than_the_selection_is_refused():
    with pytest.raises(ValueError, match="narrower"):
        bare_layer(csa2_candidate_layer=0, csa2_candidate_k=BLOCK // 2, csa2_top_k=BLOCK)


def angles(layer, seed=0):
    """Real rotary tables, not the identity. cos=1/sin=0 rotates nothing and would let
    this whole file pass with `_rope_index` replaced by the identity function."""
    torch.manual_seed(seed + 7)
    width = int(layer.rope_dim)
    theta = torch.rand(BATCH, SEQ, width) * 6.0 - 3.0
    return torch.cos(theta), torch.sin(theta)


def paired(**overrides):
    """Two layers with identical weights, differing only in the override."""
    first = bare_layer(**overrides).float()
    second = bare_layer(csa2_rope_index=False).float()
    second.load_state_dict(first.state_dict())
    return first, second


def test_the_rotary_changes_what_the_router_scores():
    # Asserted on the scores rather than the selection: at six blocks with a local window
    # there may be no discretionary slot left to flip, so a selection comparison can pass
    # while the rotary does nothing.
    roped, plain = paired(csa2_rope_index=True)
    hidden, keys, queries, _ = inputs(roped)
    position = angles(roped)
    with torch.no_grad():
        with_rope = roped._score_blocks(
            *roped._index_inputs(hidden, keys, queries, position), BLOCKS)[0]
        without = plain._score_blocks(
            *plain._index_inputs(hidden, keys, queries, position), BLOCKS)[0]
    assert not torch.allclose(with_rope, without, atol=1e-6), \
        "roping the index query and key left the scores identical"


def test_the_rotary_matches_a_reference_rotation_of_the_trailing_slice():
    layer = bare_layer(csa2_rope_index=True).float()
    cos, sin = angles(layer)
    width = cos.shape[-1]
    torch.manual_seed(3)
    keys = torch.randn(BATCH, SEQ, layer.index_dim)

    got = layer._rope_index(keys, (cos, sin))
    head, tail = keys[..., :-width], keys[..., -width:]
    half = width // 2
    left, right = tail[..., :half], tail[..., half:]
    expected_tail = torch.cat([left * cos[..., :half] - right * sin[..., :half],
                               right * cos[..., half:] + left * sin[..., half:]], dim=-1)
    torch.testing.assert_close(got[..., :-width], head)
    torch.testing.assert_close(got[..., -width:], expected_tail, atol=1e-5, rtol=1e-5)


def test_an_index_narrower_than_the_rotary_width_still_rotates():
    # The tables are cut to the slice, not the other way round; cutting only the slice
    # leaves the two disagreeing and the stock rotary raises.
    layer = bare_layer(csa2_rope_index=True, csa2_index_dim=4).float()
    cos, sin = angles(layer)
    assert cos.shape[-1] > layer.index_dim
    keys = torch.randn(BATCH, SEQ, layer.index_dim)
    rotated = layer._rope_index(keys, (cos, sin))
    assert rotated.shape == keys.shape
    assert not torch.equal(rotated, keys)


def test_the_rotary_is_off_when_no_positions_are_supplied():
    roped, plain = paired(csa2_rope_index=True)
    hidden, keys, queries, _ = inputs(roped)
    with torch.no_grad():
        assert torch.equal(roped.route(hidden, keys, queries)[0],
                           plain.route(hidden, keys, queries)[0])


def test_the_config_defaults_leave_the_hierarchy_off():
    config = csa2_config()
    assert getattr(config, "csa2_candidate_layer", -1) == -1
    assert not bare_layer().publishes_candidates
