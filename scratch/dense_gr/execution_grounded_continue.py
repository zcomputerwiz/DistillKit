# Assisted-by: Codex
"""One-shot pilot continuation; stops on failures and never promotes checkpoints."""
from datetime import datetime, timezone
import json
import math
import os

from execution_grounded_run import OUT, OLD, HERE, digest, verify, train, evaluate
import phase3_run


def status(stage, **extra):
    (OUT/'execution-status.json').write_text(json.dumps(dict(stage=stage,
        updated=datetime.now(timezone.utc).isoformat(), launcher_pid=os.getpid(),
        launcher_sha256=digest(__file__), promotion='not performed', **extra), indent=2))
    print(stage, flush=True)


def check_training(path, steps):
    report = json.loads(path.read_text())
    if report.get('aborted') or report['steps'] != steps:
        raise ValueError('incomplete training: ' + str(path))
    if not report['spill_telemetry_valid'] or report['spilled']:
        raise ValueError('invalid memory telemetry or spill: ' + str(path))
    for row in report['history']:
        for key in ('loss', 'teacher_kl', 'dpo', 'dpo_margin', 'chosen_logp'):
            if key in row and not math.isfinite(row[key]):
                raise ValueError('nonfinite training metric: ' + key)
    return report


def preflight():
    plan, _ = verify()
    argv = list(plan['arms']['targeted'])
    for flag, value in (('--max-steps', '2'), ('--save-every', '0'), ('--report-every', '1'),
                        ('--evaluate-every', '0'), ('--evaluate-windows', '1'),
                        ('--output', str(OUT/'preflight/train.json')),
                        ('--checkpoints', str(OUT/'preflight/checkpoints'))):
        argv[argv.index(flag)+1] = value
    argv += ['--no-checkpoint']
    phase3_run.invoke('preflight-training', argv, '0,1')
    report = check_training(OUT/'preflight/train.json', 2)
    # First measured objective precedes any update from u50. Padding, TP, CCE
    # and optimized norms must not create a large spurious reference margin.
    first = report['history'][0]
    if first['step'] != 0 or abs(first['dpo_margin']) > 0.05 or abs(first['dpo'] - math.log(2)) > 0.05:
        raise ValueError('initial preference reference mismatch; inspect preflight')
    print('Preflight:', report['peak_allocated_gib_by_device'], 'initial DPO', first['dpo'],
          'margin', first['dpo_margin'], flush=True)


def retention():
    plan, protocol = verify()
    for i, original in enumerate(protocol['paired_nll']):
        argv = list(original)
        argv[argv.index('--output-dir')+1] = str(OUT/'eval'/('retention' if i == 0 else 'atlas'))
        # Preserve the inherited frozen domain bank and evidence options.
        j = argv.index('--arm')
        argv = argv[:j] + ['--save-token-evidence'] + sum((['--arm', name+'='+checkpoint] for name, checkpoint in
            [('base', plan['reference']), *[(a, str(OUT/a/'checkpoints/smoke-r1-1-gr-s25-csa2'))
                                           for a in ('control', 'targeted')]]), [])
        phase3_run.invoke(f'paired-{i:02d}', argv, '0')
        evidence = OUT/'eval'/('retention' if i == 0 else 'atlas')/'token_evidence.npz'
        phase3_run.invoke(f'paired-review-{i:02d}', [str(HERE/'atlas_compare.py'),
            '--evidence', str(evidence), '--base', 'base', '--reference', 'control', '--arm', 'targeted',
            '--output', str(evidence.parent/'comparison.json')], '')


def summarize():
    from phase3_report import agent, math_summary, paired_rows, read
    arms = {'base': OLD/'eval/base', 'control': OUT/'eval/control', 'targeted': OUT/'eval/targeted'}
    result = dict(arms={}, paired={}, promotion='not performed',
                  note='Automatic proxy scores; final report semantics and state transitions require review.')
    for name, root in arms.items():
        result['arms'][name] = {p.name:agent(p) for p in sorted(root.glob('*.json'))}
        result['arms'][name].update({p.name:math_summary(p) for p in sorted(root.glob('math-*'))})
    for left, right in (('base','control'), ('base','targeted'), ('control','targeted')):
        group = result['paired'].setdefault(left+'->'+right, {})
        for path in sorted(arms[left].glob('*.json')):
            other = arms[right]/path.name
            rows = paired_rows(read(path)['records'], read(other)['records'], 'id', 'success')
            rows['candidate'] = rows.pop('masked')
            group[path.name] = rows
        for path in sorted(arms[left].glob('math-*')):
            other = arms[right]/path.name
            a = [json.loads(s) for s in (path/'cases.jsonl').read_text().splitlines()]
            b = [json.loads(s) for s in (other/'cases.jsonl').read_text().splitlines()]
            group[path.name] = {}
            for subject in ('gsm8k','math'):
                rows = paired_rows([r for r in a if r['subject']==subject],
                                   [r for r in b if r['subject']==subject], 'id', 'correct')
                rows['candidate'] = rows.pop('masked')
                group[path.name][subject] = rows
    (OUT/'comparison.json').write_text(json.dumps(result, indent=2))


def main():
    phase3_run.OUT = OUT
    status('preflight_running')
    preflight()
    for arm in ('control', 'targeted'):
        status('training_'+arm)
        train(arm)
        report = check_training(OUT/arm/'train.json', 40)
        if report['scored_tokens'] != 966679:
            raise ValueError('replay exposure changed')
    for arm in ('control', 'targeted'):
        status('evaluating_'+arm)
        evaluate(arm)
    status('paired_retention_running')
    retention()
    summarize()
    status('bounded_pilot_complete_review_required')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed_review_required', error=str(error))
        raise
