# Assisted-by: Codex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scratch/dense_gr'))
from execution_pairs import KINDS, fixture, reject_reason, witness


def test_every_chosen_branch_is_executable_and_preserves_permissions():
    for kind in KINDS:
        for n in range(8):
            w, good, _ = fixture(kind, n)
            p = witness(w, good)
            assert not p['errors'], (kind, n, p)
            if kind in ('renew', 'retry', 'report', 'uncertain_pending', 'uncertain_done', 'page'):
                assert p['state'] == p['desired'] and p['mutations'] == 1, (kind, n, p)
            else:
                assert p['mutations'] == 0


def test_renewal_is_not_a_write_and_timeout_requires_observation():
    w, good, bad = fixture('retry', 0)
    assert w.refreshed and w.state == 'pending' and not w.mutations
    assert reject_reason(w, bad) == 'false_completion'
    w, _, bad = fixture('uncertain_done', 0)
    assert w.state == w.desired and w.mutations == 1 and not w.reads
    # An attempted repeat write is rejected even when the original write committed.
    import json
    text = '<tool_call>' + json.dumps({'name': bad['function']['name'],
                                     'arguments': bad['function']['arguments']}) + '</tool_call>'
    assert reject_reason(w, text) == 'redundant_write'


def test_no_unproven_prose_is_auto_rejected():
    w, _, _ = fixture('retry', 0)
    for text in ('I will retry.', 'It was not updated.', 'What should I do?', 'The sky is blue.'):
        assert reject_reason(w, text) is None
    assert reject_reason(w, 'Done.') == 'false_completion'
    w, _, _ = fixture('readonly', 0)
    assert reject_reason(w, f'Done. The record has {w.units} units.') is None
    w, _, _ = fixture('no_tool', 0)
    assert reject_reason(w, f'{w.units + 7}. Done.') is None


def test_training_worlds_are_disjoint_from_transfer_worlds():
    for kind in KINDS:
        train, _, _ = fixture(kind, 0, 'train')
        transfer, _, _ = fixture(kind, 0, 'transfer')
        assert train.ident != transfer.ident and train.token != transfer.token
        assert set(train.names.values()).isdisjoint(transfer.names.values())


def test_bad_pair_settings_fail_before_model_allocation():
    import pytest
    from smoke_train import main
    for flag, value in (('--pair-weight', 'nan'), ('--pair-weight', '-1'),
                        ('--pair-sft-weight', 'inf'), ('--dpo-beta', '0'),
                        ('--pairs-per-step', '0')):
        with pytest.raises(SystemExit, match='preference counts/beta'):
            main(['--pairs', 'unused-pairs.jsonl', flag, value])
