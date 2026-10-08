# Assisted-by: Codex
"""Bounded, execution-grounded pilot using existing rollout/reference/train/eval tools."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import phase3_run

HERE = Path(__file__).resolve().parent
OUT = HERE / 'execution-grounded'
OLD = HERE / 'phase3-masking'
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    import numpy as np
    if (OUT / 'plan.json').exists():
        raise ValueError('plan already exists')
    old = json.loads((OLD / 'plan.json').read_text())
    protocol = json.loads((OLD / 'evaluation.json').read_text())
    plan = dict(version=1, steps=40, reference=old['start_checkpoint'],
        treatment='Turn-local DPO plus chosen-response CE; all conversational context remains masked.',
        rationale='Half-rate masked control isolates the additional preference loss. No automatic promotion.',
        replay_sha256=old['batch_sha256'], weighted_replay_targets=966679,
        preparation=dict(student_rollouts='greedy, batch 4, width 1024, new 192',
                         reference='u50, no cache, BF16, max length 1536'),
        pair_config=dict(per_step=2, weight=0.1, beta=0.1, chosen_ce_weight=1.0, seed=25),
        input_sha256={str(OLD / 'plan.json'): digest(OLD / 'plan.json'),
                      str(OLD / 'batches.json'): digest(OLD / 'batches.json'),
                      str(OUT / 'data/manifest.json'): digest(OUT / 'data/manifest.json')},
        runtime_sha256={name:digest(HERE / name) for name in
            ('execution_pairs.py', 'smoke_train.py', 'training_step.py', 'ref_logprobs.py',
             'onpolicy_rollouts.py', 'execution_grounded_run.py')}, arms={})
    for arm in ('control', 'targeted'):
        argv = list(old['arms']['masked']['argv'])
        i = argv.index('--lr-depth-ramp')
        argv[i+1:i+3] = ['0.1375', '0.1375']
        for flag, value in (('--checkpoints', str(OUT / arm / 'checkpoints')),
                            ('--output', str(OUT / arm / 'train.json'))):
            argv[argv.index(flag)+1] = value
        if arm == 'targeted':
            argv += ['--pairs', str(OUT / 'data/pairs-ref.jsonl'), '--pairs-per-step', '2',
                     '--pair-weight', '0.1', '--dpo-beta', '0.1', '--pair-sft-weight', '1.0']
        plan['arms'][arm] = argv
    # Explicit finite schedule audit, mirroring PairSource's permutation/pop order.
    order = [int(i) for i in np.random.default_rng(25).permutation(96)]
    schedule = [order.pop() for _ in range(80)]
    plan['pair_indices'] = schedule
    plan['pair_kind_counts'] = dict(Counter(__import__('execution_pairs').KINDS[i//8] for i in schedule))
    plan['hyperparameter_status'] = 'Small exploratory settings chosen from local results, not literature optima.'
    # Reuse all nine frozen primary commands, changing checkpoint/output only.
    for arm in ('control', 'targeted'):
        ckpt = OUT / arm / 'checkpoints/smoke-r1-1-gr-s25-csa2'
        commands = []
        for original in protocol['commands']['masked']:
            argv = list(original)
            for flag, value in (('--checkpoint', str(ckpt)),):
                argv[argv.index(flag)+1] = value
            for flag in ('--output', '--output-dir'):
                if flag in argv:
                    old_output = Path(argv[argv.index(flag)+1])
                    argv[argv.index(flag)+1] = str(OUT / 'eval' / arm / old_output.relative_to(OLD / 'eval/masked'))
            commands.append(argv)
        protocol['commands'][arm] = commands
    protocol['inherited_protocol_sha256'] = digest(OLD / 'evaluation.json')
    protocol['status'] = 'frozen_for_execution_grounded_pilot'
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    (OUT / 'evaluation.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print('Frozen pilot plan. Pair coverage:', plan['pair_kind_counts'], flush=True)


def verify():
    plan = json.loads((OUT / 'plan.json').read_text())
    for path, sha in plan['input_sha256'].items():
        if digest(path) != sha:
            raise ValueError('frozen input changed: ' + path)
    for name, sha in plan['runtime_sha256'].items():
        if digest(HERE / name) != sha:
            raise ValueError('runtime source changed: ' + name)
    protocol = json.loads((OUT / 'evaluation.json').read_text())
    for asset in protocol['assets']:
        if digest(asset['path']) != asset['sha256']:
            raise ValueError('evaluation asset changed: ' + asset['path'])
    manifest = json.loads((OUT / 'data/manifest.json').read_text())
    for name, sha in manifest['sha256'].items():
        if digest(OUT / 'data' / name) != sha:
            raise ValueError('pair asset changed: ' + name)
    return plan, protocol


def preparation():
    plan, _ = verify()
    invoke = phase3_run.invoke
    invoke('prepare-rollouts', [str(HERE/'onpolicy_rollouts.py'), '--checkpoint', plan['reference'],
        '--inputs', str(OUT/'data/prompts.jsonl'), '--count', '96', '--width', '1024', '--new', '192',
        '--batch-size', '4', '--seed', '25', '--greedy', '--output', str(OUT/'data/rollouts.jsonl')], '0')
    invoke('prepare-integrate', [str(HERE/'execution_pairs.py'), '--checkpoint', plan['reference'],
        '--output', str(OUT/'data'), '--rollouts', str(OUT/'data/rollouts.jsonl')], '')
    invoke('prepare-reference', [str(HERE/'ref_logprobs.py'), '--reference', plan['reference'],
        '--pairs', str(OUT/'data/pairs-onpolicy.jsonl'), '--max-length', '1536',
        '--output', str(OUT/'data/pairs-ref.jsonl')], '0')
    rows = [json.loads(s) for s in (OUT/'data/pairs-ref.jsonl').read_text().splitlines()]
    if len(rows) != 96 or len({r['pair_id'] for r in rows}) != 96:
        raise ValueError('reference stage dropped or duplicated pairs')
    import math
    for r in rows:
        a, b = r['chosen_start'], r['rejected_start']
        if a != b or r['chosen_ids'][:a] != r['rejected_ids'][:b]:
            raise ValueError('preference prefixes differ')
        if not all(math.isfinite(r['ref_'+s]) and 0 < a < len(r[s+'_ids']) for s in ('chosen','rejected')):
            raise ValueError('invalid reference score/span')
    receipt = dict(status='complete', pairs=96, assistant_turns_only=True,
        sha256={n:digest(OUT/'data'/n) for n in ('pairs-onpolicy.jsonl', 'pairs-ref.jsonl',
                                               'onpolicy-audit.json', 'rollouts.jsonl')})
    path = OUT/'prepared.json'
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError('prepared artifacts changed')
    path.write_text(json.dumps(receipt, indent=2))
    print('Preparation complete; reference spans and counts verified.', flush=True)


def train(arm):
    plan, _ = verify()
    receipt = json.loads((OUT/'prepared.json').read_text())
    for name, sha in receipt['sha256'].items():
        if digest(OUT/'data'/name) != sha:
            raise ValueError('prepared artifact changed: ' + name)
    # Existing trainer warms every pair width before measured steps, applies
    # selection replay, monitors spill, clips/synchronizes gradients and saves.
    phase3_run.invoke('train-'+arm, plan['arms'][arm], '0,1')


def evaluate(arm):
    _, protocol = verify()
    commands = protocol['commands'][arm]
    def lane(primary):
        indices = [0] if primary else list(range(1, len(commands)))
        for i in indices:
            phase3_run.invoke(f'{arm}-{i:02d}', commands[i], '0' if primary else '1')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane, primary) for primary in (True, False)]
        for f in futures:
            f.result()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('freeze', 'prepare', 'train-control', 'train-targeted',
                                     'eval-control', 'eval-targeted'))
    a = p.parse_args()
    phase3_run.OUT = OUT
    if a.stage == 'freeze':
        freeze()
    elif a.stage == 'prepare':
        preparation()
    elif a.stage.startswith('train-'):
        train(a.stage.removeprefix('train-'))
    else:
        evaluate(a.stage.removeprefix('eval-'))
