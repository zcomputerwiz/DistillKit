# Assisted-by: Codex
"""One discarded validation step at the longest actual pairs; no measured arm starts."""
import json
import math

import phase3_run
from completion_plan import OUT, HERE, digest


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    prepared=json.loads((OUT/'prepared.json').read_text())
    for name,sha in prepared['sha256'].items():
        if digest(OUT/'data'/name)!=sha:
            raise ValueError('prepared input changed: '+name)
    rows=[json.loads(s) for s in (OUT/'data/pairs-ref.jsonl').read_text().splitlines()]
    schedule=json.loads((OUT/'data/pair-schedule.json').read_text())
    longest=sorted(schedule['indices'],key=lambda i:max(len(rows[i][s+'_ids']) for s in ('chosen','rejected')),reverse=True)[:2]
    small=dict(schedule,indices=longest,purpose='One discarded longest-pair validation step')
    folder=OUT/'preflight'
    folder.mkdir(exist_ok=True)
    for name,value in [('pairs.json',small)]:
        path=folder/name
        if path.exists() and json.loads(path.read_text())!=value:
            raise ValueError('validation schedule changed')
        path.write_text(json.dumps(value,indent=2))
    argv=list(plan['arms']['completion'])
    original=argv[argv.index('--ordered-batches')+1]
    replay=json.loads(open(original).read())
    prefix=dict(replay,groups=replay['groups'][:2],origin_sha256=digest(original),purpose='One-step validation')
    (folder/'batches.json').write_text(json.dumps(prefix,indent=2))
    for flag,value in (('--max-steps','1'),('--save-every','0'),('--evaluate-every','0'),
        ('--evaluate-windows','1'),('--report-every','1'),('--pair-schedule',str(folder/'pairs.json')),
        ('--ordered-batches',str(folder/'batches.json')),('--output',str(folder/'train.json')),
        ('--checkpoints',str(folder/'unused-checkpoints'))):
        argv[argv.index(flag)+1]=value
    argv+=['--no-checkpoint']
    phase3_run.OUT=OUT
    phase3_run.invoke('prepare-backward-validation',argv,'0,1')
    report=json.loads((folder/'train.json').read_text())
    if report.get('aborted') or report['steps']!=1 or not report['spill_telemetry_valid'] or report['spilled']:
        raise ValueError('validation failed or memory telemetry invalid')
    first=report['history'][0]
    if any(not math.isfinite(first[k]) for k in ('loss','teacher_kl','dpo','dpo_margin','chosen_logp')):
        raise ValueError('nonfinite validation objective')
    if abs(first['dpo_margin'])>.05 or abs(first['dpo']-math.log(2))>.05:
        raise ValueError('large initial reference mismatch')
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed_measured_arms_not_started',
        pairs=[rows[i]['pair_id'] for i in longest],max_tokens=max(len(rows[i][s+'_ids']) for i in longest for s in ('chosen','rejected')),
        initial_dpo=first['dpo'],initial_margin=first['dpo_margin'],
        training_peaks_gib=report['peak_allocated_gib_by_device'],shared_delta_gib=report['shared_delta_gib'],
        report_sha256=digest(folder/'train.json'),weights='validation copy discarded; no checkpoint saved'),indent=2))
    print('Longest-pair backward validation passed; measured arms not started.',flush=True)


if __name__=='__main__':
    main()
