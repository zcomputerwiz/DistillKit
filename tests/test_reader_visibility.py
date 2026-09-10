"""The reader-visibility protocol's pure pieces.

These grade *where* the sidecar's information is decodable, so the properties that
matter are the ones that would otherwise let an arm cheat: a target must never be
readable from its own position, the splits must not move when a caller asks for a
different number of documents, a shuffled control must have no fixed points, and the
examiner must give every arm the same number of trainable parameters whatever the
width of the representation it grades.
"""

import math

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))

from reader_visibility.core import (  # noqa: E402
    MatchedExaminer, compare_scores, donor_map, prediction_positions,
    reader_representations, split_records,
)

from distillkit.ple_gated_sidecar import DirectionGatedPLESidecar  # noqa: E402


def _records(count, prefix="doc"):
    return [{"id": f"{prefix}{index:03d}", "text": f"body {index}"} for index in range(count)]


def test_split_membership_does_not_move_when_the_counts_change():
    """The whole point of hashing the ID: asking for more fit cannot pull from test."""
    records = _records(200)
    small = split_records(records, {"fit": 10, "validation": 5, "test": 5})
    large = split_records(records, {"fit": 40, "validation": 10, "test": 10})
    assert {row["id"] for row in small["test"]} <= {row["id"] for row in large["test"]}
    assert {row["id"] for row in small["fit"]} <= {row["id"] for row in large["fit"]}
    # And no document is ever in two places at once.
    seen = [row["id"] for split in large.values() for row in split]
    assert len(seen) == len(set(seen))


def test_split_refuses_duplicate_ids_and_deduplicates_identical_text():
    with pytest.raises(ValueError, match="duplicate document ID"):
        split_records(_records(4) + _records(1), {"fit": 2, "validation": 2, "test": 2})
    twinned = _records(60) + [{"id": "copy", "text": "body 0"}]
    everything = split_records(twinned, {"fit": 2, "validation": 2, "test": 2})
    kept = [row["text"] for split in everything.values() for row in split]
    assert len(kept) == len(set(kept)), "identical text must not be scored twice"


def test_split_refuses_to_silently_shrink_a_request():
    with pytest.raises(ValueError, match="need 100 fit documents"):
        split_records(_records(20), {"fit": 100, "validation": 2, "test": 2})


def test_donor_cycle_has_no_fixed_points():
    """A self-donor is a shuffled control that is not shuffled."""
    for size in (2, 3, 17, 64):
        donors = donor_map([f"d{index}" for index in range(size)])
        assert len(donors) == size
        assert all(key != value for key, value in donors.items())
        assert sorted(donors.values()) == sorted(donors)


def test_prediction_positions_never_read_the_token_they_predict():
    """Target at j is scored from position j-1. Off by one here is a leak."""
    ids = list(range(10))
    positions, targets, total = prediction_positions(ids, {"assistant": [(4, 8)]})
    assert positions.tolist() == [3, 4, 5, 6]
    assert targets.tolist() == [4, 5, 6, 7]
    assert total == 4
    assert (targets == torch.tensor(ids)[positions + 1]).all()


def test_prediction_positions_drop_the_first_token_of_a_leading_span():
    """Position 0 has no predictor, so a span at the start starts one target later."""
    positions, targets, _ = prediction_positions(list(range(6)), {"assistant": [(0, 3)]})
    assert positions.tolist() == [0, 1]
    assert targets.tolist() == [1, 2]


def test_prediction_positions_truncate_to_a_shorter_donor():
    """A shuffled arm scores the natural prefix; no fabricated padding row is graded."""
    ids = list(range(20))
    full, _, total = prediction_positions(ids, {"assistant": [(2, 18)]})
    short, _, short_total = prediction_positions(ids, {"assistant": [(2, 18)]}, donor_length=10)
    assert short.tolist() == [index for index in full.tolist() if index < 9]
    assert short_total == total, "the reported coverage is of the whole span"


def test_prediction_positions_refuse_a_span_outside_the_sequence():
    with pytest.raises(ValueError, match="role span outside"):
        prediction_positions(list(range(5)), {"assistant": [(2, 9)]})


def _reader(hidden=16, features=24, branches=2):
    torch.manual_seed(0)
    reader = DirectionGatedPLESidecar(hidden, features, hc_count=branches, gate_directions=2)
    with torch.no_grad():
        reader.value_proj.weight.normal_(std=0.1)
        reader.conv1d.weight.normal_(std=0.1)
    return reader.eval().requires_grad_(False)


def test_reader_representations_decompose_the_actual_forward():
    reader = _reader()
    torch.manual_seed(1)
    query = torch.randn(2, 7, reader.hc_count, reader.hidden_size)
    features = torch.randn(2, 7, reader.feature_dim)
    parts = reader_representations(reader, query, features)
    assert torch.equal(parts["realized_update"], reader(query, features).float() - query.float())
    assert torch.allclose(parts["reader_update"], parts["gated_value"] + parts["convolution"])
    assert parts["admission"].shape == (2, 7, reader.hc_count, 1)
    assert parts["ungated_value"].shape == (2, 7, reader.hidden_size)


def test_reader_representations_refuse_a_reader_that_could_still_learn():
    """A grade is only about visibility if the thing being graded cannot move."""
    reader = _reader()
    query = torch.randn(1, 3, reader.hc_count, reader.hidden_size)
    features = torch.randn(1, 3, reader.feature_dim)
    reader.train()
    with pytest.raises(ValueError, match="frozen"):
        reader_representations(reader, query, features)
    reader.eval().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        reader_representations(reader, query, features)


def test_examiner_gives_every_arm_the_same_trainable_parameter_count():
    """Matched capacity is the only reason arms of different width are comparable."""
    counts = set()
    for dim in (24, 160, 5120):
        examiner = MatchedExaminer([dim], hidden_size=16, width=8)
        counts.add(sum(p.numel() for p in examiner.parameters() if p.requires_grad))
    assert len(counts) == 1, counts
    # The projections carry the width difference and must not be trainable.
    examiner = MatchedExaminer([5120], hidden_size=16, width=8)
    assert not any(buffer.requires_grad for buffer in examiner.buffers())


def test_examiner_starts_as_the_identity_so_a_gain_is_all_it_learned():
    examiner = MatchedExaminer([12], hidden_size=6, width=4)
    baseline = torch.randn(2, 5, 6)
    output = examiner(baseline, torch.randn(2, 5, 12))
    assert torch.equal(output, baseline.float())


def test_examiner_refuses_unflattened_branches_rather_than_broadcasting():
    examiner = MatchedExaminer([12], hidden_size=6, width=4)
    with pytest.raises(ValueError, match="flatten branch axes"):
        examiner(torch.randn(2, 5, 6), torch.randn(2, 5, 2, 6))


def test_examiner_refuses_statistics_it_cannot_normalise_with():
    with pytest.raises(ValueError, match="normalization statistics"):
        MatchedExaminer([4], hidden_size=6, width=4,
                        statistics=[(torch.zeros(4), torch.zeros(4))])


def _scores(values, alignment="a"):
    return [{"id": key, "nll": nll, "alignment": alignment} for key, nll in values.items()]


def test_compare_scores_sign_convention_is_reference_minus_candidate():
    reference = _scores({"a": [2.0, 2.0], "b": [2.0, 2.0], "c": [2.0, 2.0]})
    candidate = _scores({"a": [1.0, 1.0], "b": [1.0, 1.0], "c": [1.0, 1.0]})
    result = compare_scores(reference, candidate, resamples=200)
    assert result["gain_nats"] == pytest.approx(1.0)
    assert result["status"] == "helpful"
    assert result["fraction_documents_helped"] == 1.0
    assert compare_scores(candidate, reference, resamples=200)["status"] == "harmful"


def test_compare_scores_reports_inconclusive_rather_than_a_bare_point_estimate():
    torch.manual_seed(0)
    reference = _scores({f"d{i}": [2.0] for i in range(24)})
    candidate = _scores({f"d{i}": [2.0 + (1.0 if i % 2 else -1.0)] for i in range(24)})
    assert compare_scores(reference, candidate, resamples=2000)["status"] == "inconclusive"


def test_compare_scores_refuses_arms_whose_tokens_do_not_line_up():
    reference = _scores({"a": [1.0, 1.0], "b": [1.0, 1.0]})
    candidate = _scores({"a": [1.0, 1.0], "b": [1.0, 1.0]}, alignment="different")
    with pytest.raises(ValueError, match="alignment mismatch"):
        compare_scores(reference, candidate, resamples=200)
    with pytest.raises(ValueError, match="same >=2 document IDs"):
        compare_scores(reference, _scores({"a": [1.0, 1.0]}), resamples=200)


def test_compare_scores_requires_an_alignment_fingerprint():
    rows = [{"id": "a", "nll": [1.0]}, {"id": "b", "nll": [1.0]}]
    with pytest.raises(ValueError, match="alignment fingerprint"):
        compare_scores(rows, rows, resamples=200)


def test_token_nll_matches_a_dense_reference_and_never_builds_dense_logits():
    """The chunking is the point: it must agree with the obvious version exactly."""
    from reader_visibility.core import token_nll

    torch.manual_seed(0)
    head = torch.nn.Linear(8, 64, bias=False).eval().requires_grad_(False)
    hidden = torch.randn(37, 8)
    targets = torch.randint(0, 64, (37,))
    dense = torch.nn.functional.cross_entropy(head(hidden).float(), targets, reduction="none")
    for chunk in (1, 7, 256):
        assert torch.allclose(torch.tensor(token_nll(hidden, head, targets, chunk_size=chunk)),
                              dense, atol=1e-6)


def test_token_nll_truncates_to_the_signal_vocabulary_when_the_head_is_padded():
    from reader_visibility.core import token_nll

    torch.manual_seed(0)
    head = torch.nn.Linear(8, 64, bias=False).eval().requires_grad_(False)
    hidden, targets = torch.randn(5, 8), torch.randint(0, 40, (5,))
    truncated = torch.nn.functional.cross_entropy(head(hidden)[:, :40].float(), targets,
                                                  reduction="none")
    assert torch.allclose(torch.tensor(token_nll(hidden, head, targets, vocab_size=40)),
                          truncated, atol=1e-6)


def test_token_nll_refuses_a_head_that_is_still_training():
    from reader_visibility.core import token_nll

    head = torch.nn.Linear(4, 8, bias=False)
    with pytest.raises(ValueError, match="frozen"):
        token_nll(torch.randn(3, 4), head, torch.zeros(3, dtype=torch.long))
