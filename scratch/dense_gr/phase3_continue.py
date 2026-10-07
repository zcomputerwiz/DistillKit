# Assisted-by: Codex
"""Continue the authorized bounded experiment after existing baseline workers finish.

Checks frozen case inventories and output contracts before training. Semantic review
flags are preserved for comparison; no model is promoted and no curriculum starts.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time
from phase3_run import OUT, main as run_stage, invoke


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_available(require_complete=False):
    protocol=read(OUT/'evaluation.json')
    bank_path=OUT.parent/'phase3-eval/math-dev-v3.json'
    bank=read(bank_path)
    scenarios=read(OUT.parent/'phase3-eval/agent-scenarios.json')['rows']
    legacy=[json.loads(l) for l in (OUT.parent/'agentic-v2-data/trajectories.jsonl').read_text().splitlines()]
    checked=[]
    for index,argv in enumerate(protocol['commands']['base']):
        destination=Path(argv[argv.index('--output')+1])
        if Path(argv[0]).name=='math_dev_eval.py':
            if not (destination/'summary.json').exists():
                if require_complete:
                    raise ValueError('missing completed output: '+str(destination))
                continue
            manifest=read(destination/'manifest.json')
            if manifest['status']!='complete' or manifest['bank_sha256']!=hashlib.sha256(bank_path.read_bytes()).hexdigest():
                raise ValueError('math output bank or status mismatch')
            rows=[json.loads(l) for l in (destination/'cases.jsonl').read_text().splitlines()]
            expected=bank['rows']
            if '--limit-per-subject' in argv:
                limit=int(argv[argv.index('--limit-per-subject')+1])
                expected=[r for s in ('gsm8k','math')
                          for r in [x for x in bank['rows'] if x['subject']==s][:limit]]
            expected_ids={r['id'] for r in expected}
            found=[r['id'] for r in rows]
            if len(found)!=len(expected_ids) or set(found)!=expected_ids:
                raise ValueError('math case inventory mismatch')
            by_id={r['id']:r for r in expected}
            for row in rows:
                if row['question_sha256']!=by_id[row['id']]['question_sha256'] or row['reference']!=by_id[row['id']]['answer']:
                    raise ValueError('math question or answer mismatch')
                if len(row['token_ids'])>manifest['budget'] or row['truncated']!=(row['stop_token'] is None):
                    raise ValueError('math truncation contract mismatch')
            if manifest['eos']!=[248044,248046]:
                raise ValueError('math stop policy mismatch')
        else:
            if not destination.exists():
                if require_complete:
                    raise ValueError('missing agent output: '+str(destination))
                continue
            output=read(destination)
            rows=output['records']
            expected=([r for r in legacy if r['split']=='eval'] if Path(argv[0]).name=='agentic_live_eval.py'
                      else [r for r in scenarios if '--short-only' not in argv or r['padding_lines']==0])
            expected_ids={r['doc_id'] for r in expected}
            found=[r['id'] for r in rows]
            if len(found)!=len(expected_ids) or set(found)!=expected_ids:
                raise ValueError('agent case inventory mismatch')
            if output['grading_version']!=2 or output['sample']!=('--sample' in argv):
                raise ValueError('agent decoding/grading contract mismatch')
            originals={r['doc_id']:r for r in expected}
            for row in rows:
                original=originals[row['id']]['messages']
                original=original[:next((i for i,m in enumerate(original) if m['role']=='assistant'),len(original))]
                if row['messages'][:len(original)]!=original:
                    raise ValueError('agent initial messages mismatch: '+row['id'])
        checkpoint=manifest['checkpoint'] if Path(argv[0]).name=='math_dev_eval.py' else output['checkpoint']
        if Path(checkpoint).resolve()!=Path(read(OUT/'plan.json')['start_checkpoint']).resolve():
            raise ValueError('baseline checkpoint mismatch')
        checked.append(dict(index=index,output=str(destination),cases=len(rows)))
    return checked


def main(check_only):
    if check_only:
        print(json.dumps(validate_available(),indent=2))
        return
    status=OUT/'execution-status.json'
    def update(stage,**details):
        status.write_text(json.dumps(dict(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**details),indent=2))
        print(stage,flush=True)
    try:
        update('waiting_for_baseline')
        protocol=read(OUT/'evaluation.json')
        while True:
            receipts=[OUT/'runs'/f'base-{i:02d}.json' for i in range(len(protocol['commands']['base']))]
            states=[read(p) if p.exists() else {} for p in receipts]
            if any(s.get('status')=='failed' for s in states):
                raise RuntimeError('baseline stage failed; training not started')
            if all(s.get('status')=='complete' for s in states):
                break
            time.sleep(10)
        update('validating_baseline')
        checked=validate_available(True)
        (OUT/'baseline-validation.json').write_text(json.dumps(dict(status='passed',checked=checked,
            note='Structural protocol validation. Free-prose review flags remain in the report; no promotion is authorized.'),indent=2))
        invoke('report-baseline',[str(OUT.parent/'phase3_report.py')],'')
        update('training_masked')
        run_stage('train')
        update('evaluating_candidate')
        run_stage('candidate')
        update('paired_retention')
        run_stage('paired')
        invoke('report-comparison',[str(OUT.parent/'phase3_report.py')],'')
        update('bounded_experiment_complete',promotion='not performed',
               next='Review paired retention, outcomes, free-prose flags and budget sensitivity before final confirmation or curriculum.')
    except Exception as error:
        update('failed',error=str(error))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-available',action='store_true')
    main(p.parse_args().check_available)
