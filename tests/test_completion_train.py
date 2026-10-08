# Assisted-by: Codex
import json
import sys
from pathlib import Path

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scratch/dense_gr'))
from completion_train import OrderedPairSource, inventory_hash


def fixture(tmp_path):
    rows=[dict(pair_id=f'pair-{i}',chosen_ids=[i+1,7,8],rejected_ids=[i+1,9,8],
               chosen_start=1,rejected_start=1,ref_chosen=-1.,ref_rejected=-2.) for i in range(3)]
    path=tmp_path/'pairs.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    schedule=tmp_path/'schedule.json'
    schedule.write_text(json.dumps(dict(pair_inventory_sha256=inventory_hash(rows),indices=[2,0,1])))
    return path,schedule


def test_ordered_pairs_preserve_full_path_order_and_never_cycle(tmp_path):
    path,schedule=fixture(tmp_path)
    source=OrderedPairSource(path,0,schedule=schedule,device='cpu',seed=999)
    first=source.take(2)
    last=source.take(1)
    assert [r['chosen_ids'][0,0].item() for r in first+last]==[3,1,2]
    assert all(r['chosen_start']==1 and r['chosen_end']==3 for r in first+last)
    with pytest.raises(ValueError,match='exhausted'):
        source.take(1)


def test_changed_reference_inventory_is_rejected(tmp_path):
    path,schedule=fixture(tmp_path)
    rows=path.read_text().splitlines()
    path.write_text('\n'.join(reversed(rows))+'\n')
    with pytest.raises(ValueError,match='inventory changed'):
        OrderedPairSource(path,0,schedule=schedule,device='cpu')
