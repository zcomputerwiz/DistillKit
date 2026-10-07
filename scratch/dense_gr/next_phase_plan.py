# Assisted-by: Codex
"""Freeze masked replay for comparison with u50. Planning only; never launches training."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
from agentic_arm import HERE, recipe, teacher_for
from training_state import OrderedBatches


def main(out, steps):
    if out.exists():
        raise ValueError('refuse to overwrite phase plan')
    out.mkdir(parents=True)
    data = HERE / 'agentic-v2-data'
    tuning = dict(control=True, steps=steps, code_multiplier=3, rate_scale=.275)
    _, options = recipe(data, out/'masked', mask_conversational=True, **tuning)
    teacher = teacher_for(options)
    groups = teacher._groups(1, 128, 32768)
    pools, epoch = defaultdict(list), defaultdict(float)
    for i, (docs, width) in enumerate(groups):
        owners = {str(teacher.cache._owner[d][0]) for d in docs}
        for d in docs:
            epoch[str(teacher.cache._owner[d][0])] += teacher.doc_weight(d, min(width, teacher.real_width[d]))
        # Select existing homogeneous groups, without changing packing or masks.
        if len(owners) == 1 and teacher.group_weight(docs, width) >= 1024:
            pools[next(iter(owners))].append(i)
    if set(pools) != set(epoch):
        raise ValueError('a replay source has no eligible canonical batch')
    rng = np.random.default_rng(25)
    for source in sorted(pools):
        rng.shuffle(pools[source])
    shares = {s:max(v/sum(epoch.values()), .02) for s,v in epoch.items()}
    norm = sum(shares.values())
    shares = {s:v/norm for s,v in shares.items()}
    counts = dict.fromkeys(shares, 0.)
    chosen = []
    for _ in range(steps*2):
        source = min(shares, key=lambda s:(counts[s]/shares[s],s))
        if not pools[source]:
            raise ValueError('source exhausted; do not silently repeat batches')
        i = pools[source].pop()
        chosen.append(i)
        counts[source] += teacher.group_weight(*groups[i])
    manifests = [c.manifest for c in teacher.cache.caches]
    frozen = dict(version=1, cache_sha256=hashlib.sha256(json.dumps(manifests,sort_keys=True).encode()).hexdigest(),
                  groups=[dict(documents=groups[i][0],width=groups[i][1]) for i in chosen])
    path = out/'batches.json'
    path.write_text(json.dumps(frozen,indent=2))
    arms = {}
    for name, mask in [('masked',True)]:
        stream = OrderedBatches(teacher,groups,frozen)
        argv, _ = recipe(data,out/name,mask_conversational=mask,**tuning)
        argv += ['--ordered-batches',str(path.resolve())]
        # Flat after warmup; independent of context targets removed by masking.
        argv[argv.index('--decay-fraction')+1] = '0'
        exposure = stream.exposure()
        targets = sum(row[2] for row in exposure.values())
        if targets >= 10_000_000:
            raise ValueError('phase exceeds budget; no partial schedule allowed')
        arms[name] = dict(argv=argv, weighted_targets=targets, exposure=exposure,
                          sample_fingerprint=stream.fingerprint)
    teacher.close()
    plan = dict(version=1, status='planned_not_started', start_checkpoint=str(HERE/'merges-long1/u50'),
                steps=steps, shared_batches=str(path.resolve()), batch_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                intended_source_shares=shares, arms=arms,
                treatment='Masked replay only, compared with unchanged u50. No legacy context-loss arm or new agent curriculum. This measures the combined replay update, not an isolated causal effect of masking.',
                gates='See scratch/csa2-eval/NEXT_TRAINING_PHASE.md. Do not auto-promote or launch a curriculum arm.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({a:v['weighted_targets'] for a,v in arms.items()}),flush=True)


def verify(out):
    from atlas import roles_of, ROLES
    plan=json.loads((out/'plan.json').read_text())
    path=Path(plan['shared_batches'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=plan['batch_sha256']:
        raise ValueError('shared batch file changed')
    _,options=recipe(HERE/'agentic-v2-data',out/'masked',control=True,steps=plan['steps'],
                     code_multiplier=3,rate_scale=.275,mask_conversational=True)
    teacher=teacher_for(options)
    groups=teacher._groups(1,128,32768)
    stream=OrderedBatches(teacher,groups,json.loads(path.read_text()))
    if stream.fingerprint!=plan['arms']['masked']['sample_fingerprint']:
        raise ValueError('runtime fingerprint differs from frozen plan')
    targets=context=batches=0
    raw={'teacher-cache-frontier-code-raw','teacher-cache-general-pilot-w8'}
    while (record:=stream.take(float('inf'))) is not None:
        ids=record['input_ids']
        weight=record.get('weight',torch.ones_like(ids,dtype=torch.float32))
        targets+=float(weight[:,:-1].sum())
        for n,doc in enumerate(record['doc_ids']):
            if Path(teacher.cache._owner[doc][0]).name in raw:
                continue
            labels=roles_of(ids[n].numpy())[1:]
            mask=np.isin(labels,[ROLES.index(r) for r in ('user','system','tool-result')])
            context+=float(weight[n,:-1].numpy()[mask].sum())
        batches+=1
    teacher.close()
    if context or targets!=plan['arms']['masked']['weighted_targets'] or batches!=plan['steps']*2:
        raise ValueError((context,targets,batches))
    result=dict(batches=batches,targets=targets,conversational_context_targets=context,
                plan_sha256=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),
                result='passed; CPU data/objective preflight, no model update')
    (out/'preflight.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)


def evaluation_assets(out, check=False):
    target=out/'evaluation.json'
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while block:=f.read(1024*1024):
                h.update(block)
        return h.hexdigest()
    if check:
        spec=json.loads(target.read_text())
        for item in spec['assets']:
            if digest(item['path'])!=item['sha256']:
                raise ValueError('evaluation asset changed: '+item['path'])
        bank=json.loads((HERE/'phase3-eval/math-dev-v3.json').read_text())
        sources={str(p.resolve()) for p in (HERE.parents[2]/'capture-data').glob('*.jsonl')}
        if sources!={str(Path(r['path']).resolve()) for r in bank['source_files']}:
            raise ValueError('capture inventory changed; repeat overlap screening')
        for item in bank['source_files']:
            if digest(item['path'])!=item['sha256']:
                raise ValueError('capture changed after overlap screening: '+item['path'])
        print('Evaluation assets and capture provenance verified',flush=True)
        return
    if target.exists():
        raise ValueError('refuse to overwrite evaluation protocol')
    bank_path=HERE/'phase3-eval/math-dev-v3.json'
    bank=json.loads(bank_path.read_text())
    math=[r for r in bank['rows'] if r['subject']=='math']
    if not bank['ready'] or len(math)!=64 or len({r['category'] for r in math})!=7:
        raise ValueError('math bank coverage incomplete')
    paths=[out/'plan.json',out/'batches.json',HERE/'phase3-eval/agent-scenarios.json',bank_path,
           HERE/'phase3-eval/retention/domains.pt',HERE/'phase3-eval/retention/manifest.json',
           HERE/'agentic-v2-data/trajectories.jsonl',HERE.parent/'csa2-eval/atlas/domains-qa32768-20261005.pt']
    commands={}
    checkpoints={'base':HERE/'merges-long1/u50','masked':out/'masked/checkpoints/smoke-r1-1-gr-s25-csa2'}
    for arm,ck in checkpoints.items():
        dest=out/'eval'/arm
        commands[arm]=[
            [str(HERE/'agentic_scenarios.py'),'run','--checkpoint',str(ck),'--output',str(dest/'scenarios-greedy.json')],
            [str(HERE/'agentic_live_eval.py'),'--checkpoint',str(ck),'--data',str(HERE/'agentic-v2-data/trajectories.jsonl'),
             '--output',str(dest/'live-v2.json'),'--grading-version','2']]
        for seed in (1,2,3):
            commands[arm].append([str(HERE/'agentic_scenarios.py'),'run','--checkpoint',str(ck),'--output',str(dest/f'scenarios-s{seed}.json'),
                                  '--short-only','--sample','--seed',str(seed),'--batch-size','1'])
        for seed in (0,1,2):
            commands[arm].append([str(HERE/'math_dev_eval.py'),'--checkpoint',str(ck),'--output',str(dest/f'math-s{seed}'),
                                  '--seed',str(seed),'--budget','2048'])
        commands[arm].append([str(HERE/'math_dev_eval.py'),'--checkpoint',str(ck),'--output',str(dest/'math-long-s0'),
                             '--budget','4096','--limit-per-subject','16','--batch','4'])
    paired=[]
    for name,domains in [('retention',paths[4]),('atlas',paths[-1])]:
        paired.append([str(HERE/'atlas.py'),'nll','--domains',str(domains),'--output-dir',str(out/'eval'/name),
                       '--arm','base='+str(checkpoints['base']),'--arm','masked='+str(checkpoints['masked']),'--save-token-evidence'])
    result=dict(version=1,status='frozen_not_executed',assets=[dict(path=str(p.resolve()),sha256=digest(p)) for p in paths],
                commands=commands,paired_nll=paired,
                instruction='Run with project Python from DistillKit. Create destination parents first. Baseline before training; candidate afterward. Final code/math benchmarks are separate confirmation, not recipe-selection metrics.')
    target.write_text(json.dumps(result,indent=2))
    print('Frozen evaluation protocol',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'phase3-masking')
    p.add_argument('--steps',type=int,default=40)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--verify',action='store_true')
    mode.add_argument('--freeze-evaluations',action='store_true')
    mode.add_argument('--verify-evaluations',action='store_true')
    a=p.parse_args()
    if a.steps < 16:
        p.error('at least 16 steps required for broad replay coverage')
    if a.verify:
        verify(a.output.resolve())
    elif a.freeze_evaluations or a.verify_evaluations:
        evaluation_assets(a.output.resolve(),a.verify_evaluations)
    else:
        main(a.output.resolve(),a.steps)
