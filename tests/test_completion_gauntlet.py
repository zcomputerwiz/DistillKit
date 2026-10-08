# Assisted-by: Codex
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scratch/dense_gr'))
from completion_gauntlet import check_inventory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scratch/downstream/code_bench'))
from generate import stop_mask


def test_comparison_refuses_missing_tasks_and_changed_prompts():
    a={'one':dict(prompt_sha256='same'),'two':dict(prompt_sha256='second')}
    check_inventory(a,dict(a),2)
    with pytest.raises(ValueError,match='inventory'):
        check_inventory(a,{'one':a['one']},2)
    b=dict(a,two=dict(prompt_sha256='changed'))
    with pytest.raises(ValueError,match='prompt'):
        check_inventory(a,b,2)


def test_dual_stop_policy_detects_both_boundaries_and_keeps_legacy_scalar():
    tokens=torch.tensor([[1,248044,2,248046]])
    assert stop_mask(tokens,[248044,248046]).tolist()==[[False,True,False,True]]
    assert stop_mask(tokens,248046).tolist()==[[False,False,False,True]]
