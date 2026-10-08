# Assisted-by: Codex
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scratch/dense_gr'))
import completion_run as runner


def test_training_gate_rejects_spill_and_nonfinite_metrics(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'OUT',tmp_path)
    folder=tmp_path/'completion'
    folder.mkdir()
    for step in (10,20,30,40):
        state=folder/'checkpoints'/f'state-step-{step:08d}'/'state.pt'
        state.parent.mkdir(parents=True)
        state.write_bytes(b'fixture')
    report=dict(steps=40,scored_tokens=966679,spill_telemetry_valid=True,spilled=False,
        history=[dict(step=0,loss=0.8,teacher_kl=0.4,dpo=0.693147,dpo_margin=0.0,chosen_logp=-3.0)])
    runner.write(folder/'train.json',report)
    runner.check_training('completion')
    report['spilled']=True
    runner.write(folder/'train.json',report)
    with pytest.raises(ValueError,match='spill'):
        runner.check_training('completion')
    report['spilled']=False
    report['history'][0]['dpo_margin']=float('nan')
    (folder/'train.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='nonfinite'):
        runner.check_training('completion')


def test_frozen_protocol_detects_data_and_environment_changes(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'OUT',tmp_path)
    path=tmp_path/'pairs-ref.jsonl'
    path.write_text('frozen reference scores')
    runner.write(tmp_path/'plan.json',dict(steps=40))
    runner.write(tmp_path/'evaluation.json',dict(assets=[dict(path=str(path),sha256=runner.digest(path))]))
    monkeypatch.setattr(runner,'versions',lambda:dict(torch='fixture-version'))
    runner.write(tmp_path/'launch.json',dict(protocol_sha256=runner.digest(tmp_path/'evaluation.json'),
        runtime_sha256={},environment=runner.versions()))
    assert runner.verify()[0]['steps']==40
    path.write_text('changed reference scores')
    with pytest.raises(ValueError,match='frozen asset changed'):
        runner.verify()
    monkeypatch.setattr(runner,'versions',lambda:dict(torch='changed-version'))
    with pytest.raises(ValueError,match='environment changed'):
        runner.verify()
