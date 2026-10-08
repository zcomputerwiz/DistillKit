# Assisted-by: Codex
"""Full downstream confirmation using the project's existing benchmark tools."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import phase3_run
from completion_run import digest, read, write, runtime_sources, versions

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'completion-gauntlet'
CODE=ROOT/'scratch/downstream/code_bench'
MATH=ROOT/'scratch/downstream/math_bench'


def environment_versions():
    result=versions()
    # smoke_train's Windows compatibility shim aliases metadata.version('triton')
    # after import. Bind installed distributions rather than an import-order alias.
    for name in result['packages']:
        try:result['packages'][name]=metadata.distribution(name).version
        except metadata.PackageNotFoundError:result['packages'][name]=None
    return result


def freeze():
    if (OUT/'plan.json').exists():
        raise ValueError('gauntlet already frozen')
    OUT.mkdir(exist_ok=True)
    from datasets import load_dataset
    sys.path.insert(0,str(MATH))
    from run_math import problems
    checkpoints={'base':HERE/'merges-long1/u50',
        'candidate':HERE/'completion-v2/completion/checkpoints/smoke-r1-1-gr-s25-csa2'}
    datasets={}
    math_bank={}
    for bench in ('gsm8k','math500'):
        rows=problems(bench)
        datasets[bench]=[dict(id=i,question=q,answer=a) for i,q,a in rows]
        math_bank[bench]=[[q,a] for _,q,a in rows]
    for bench,repo in (('humaneval','evalplus/humanevalplus'),('mbpp','evalplus/mbppplus')):
        datasets[bench]=list(load_dataset(repo,split='test'))
    write(OUT/'datasets.json',datasets)
    write(OUT/'math-candidates.json',math_bank)
    protocol=read(HERE/'completion-v2/evaluation.json')
    commands={};scoring=[]
    for arm,checkpoint in checkpoints.items():
        jobs=[]
        for kind,benches,script,budget in (('code',('humaneval','mbpp'),CODE/'generate.py',2048),
                                         ('math',('gsm8k','math500'),MATH/'run_math.py',4096)):
            for bench in benches:
                for mode in ('nothink', 'think-s0','think-s1','think-s2'):
                    directory=OUT/arm/f'{bench}-{mode}'
                    argv=[str(script),'--checkpoint',str(checkpoint),'--bench',bench,
                        '--output',str(directory),'--compiled','--batch-size','8',
                        '--max-new-tokens',str(budget),'--eos-token-ids','248044','248046']
                    if mode=='nothink':
                        argv+=['--no-thinking']
                    else:
                        argv+=['--sample','--case-seeds','--seed',mode[-1]]
                    jobs.append(dict(label=f'{arm}-{bench}-{mode}',argv=argv,kind=kind))
                    if kind=='code':
                        scoring.append(dict(label=f'score-{arm}-{bench}-{mode}',argv=[str(Path(__file__).resolve()),
                            'score','--directory',str(directory),'--bench',bench]))
        for task in ('mmlu','arc'):
            argv=['-m','distillkit.independent_eval','evaluate','--checkpoint',str(checkpoint),
                '--bundle',str(ROOT/'scratch/independent-eval/full-bundle-384.json'),
                '--split','confirmation','--tasks',task,'--max-seconds','570',
                '--output',str(OUT/arm/(task+'.json'))]
            jobs.append(dict(label=f'{arm}-{task}',argv=argv,kind='mcq'))
        commands[arm]=jobs
    assets={str(OUT/'datasets.json'):digest(OUT/'datasets.json'),
            str(OUT/'math-candidates.json'):digest(OUT/'math-candidates.json'),
            str(ROOT/'scratch/independent-eval/full-bundle-384.json'):digest(ROOT/'scratch/independent-eval/full-bundle-384.json')}
    for path in (CODE/'run_docker.ps1',CODE/'robust_eval.py',HERE/'phase3-eval/math-dev-v3.json'):
        assets[str(path)]=digest(path)
    for checkpoint in checkpoints.values():
        for path in checkpoint.iterdir():
            if path.is_file():assets[str(path)]=digest(path)
    entrypoints=[Path(__file__).resolve(),CODE/'generate.py',MATH/'run_math.py',HERE/'freeze_math_dev.py']
    runtime={str(p):digest(p) for p in runtime_sources(entrypoints)}
    plan=dict(status='frozen_not_started',checkpoints={k:str(v) for k,v in checkpoints.items()},
        commands=commands,scoring=scoring,assets=assets,runtime_sha256=runtime,environment=environment_versions(),
        counts={k:len(v) for k,v in datasets.items()},
        semantics='Base is u50; candidate is completion step 40. No further training or promotion.',
        code_sandbox_image=subprocess.check_output(['docker','image','inspect','code-bench-sandbox','--format','{{.Id}}'],text=True).strip(),
        prior_agent_and_retention_protocol_sha256=digest(HERE/'completion-v2/evaluation.json'),
        limitation='Public benchmark exposure is possible. Math is also reported after the established local-capture overlap screen; this does not establish pretraining cleanliness.')
    write(OUT/'plan.json',plan)
    write(OUT/'evaluation.json',dict(status='frozen_gauntlet',assets=[dict(path=p,sha256=h) for p,h in assets.items()]))
    print('Frozen full benchmark counts:',plan['counts'],flush=True)


def verify():
    plan=read(OUT/'plan.json')
    if environment_versions()!=plan['environment']:raise ValueError('environment changed')
    for field in ('assets','runtime_sha256'):
        for path,sha in plan[field].items():
            if digest(path)!=sha:raise ValueError('frozen source/input changed: '+path)
    image=subprocess.check_output(['docker','image','inspect','code-bench-sandbox','--format','{{.Id}}'],text=True).strip()
    if image!=plan['code_sandbox_image']:raise ValueError('sandbox image changed')
    return plan


def score(directory,bench):
    subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(CODE/'run_docker.ps1'),
                    str(directory),bench],check=True)
    result=read(directory/'eval_results.json')
    if not result.get('results'):raise ValueError('sandbox scoring returned no cases')


def status(stage,**extra):
    write(OUT/'status.json',dict(stage=stage,updated=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),promotion='not performed',**extra))
    print(stage,flush=True)


def check_inventory(a,b,count):
    if a.keys()!=b.keys() or len(a)!=count:
        raise ValueError('benchmark case inventory mismatch')
    for key in a:
        if a[key]['prompt_sha256']!=b[key]['prompt_sha256']:
            raise ValueError('benchmark prompt mismatch')


def compare():
    from compare import mcnemar
    dataset=read(OUT/'datasets.json');overlap=read(OUT/'math-overlap.json')['excluded']
    rows={}; result={}
    for bench in dataset:
        result[bench]={}
        modes=('nothink','think-s0','think-s1','think-s2')
        for mode in modes:
            for arm in ('base','candidate'):
                directory=OUT/arm/f'{bench}-{mode}'
                if bench in ('humaneval','mbpp'):
                    completions=[json.loads(l) for l in (directory/'completions.jsonl').read_text().splitlines()]
                    scores=read(directory/'eval_results.json')['results']
                    def ident(value):
                        value=str(value)
                        return 'Mbpp/'+value if bench=='mbpp' and not value.startswith('Mbpp/') else value
                    inventory={ident(c['task_id']):c for c in completions}
                    outcomes={str(s['task_id']):dict(correct=s['base']=='pass' and s['plus']=='pass',
                        prompt_sha256=inventory[str(s['task_id'])]['prompt_sha256'],
                        truncated=inventory[str(s['task_id'])]['truncated']) for s in scores}
                else:
                    outcomes={str(x['id']):x for x in read(directory/'results.json')['records']}
                rows[arm,bench,mode]=outcomes
            a,b=rows['base',bench,mode],rows['candidate',bench,mode]
            check_inventory(a,b,len(dataset[bench]))
            scopes={'all':list(a)}
            if bench in ('gsm8k','math500'):
                scopes['local_capture_unmatched']=[str(row['id']) for n,row in enumerate(dataset[bench]) if f'{bench}:{n}' not in overlap]
            result[bench][mode]={}
            for scope,ids in scopes.items():
                gained=sum(b[i]['correct'] and not a[i]['correct'] for i in ids)
                lost=sum(a[i]['correct'] and not b[i]['correct'] for i in ids)
                result[bench][mode][scope]=dict(cases=len(ids),base=sum(a[i]['correct'] for i in ids),
                    candidate=sum(b[i]['correct'] for i in ids),gained=gained,lost=lost,
                    mcnemar_p=mcnemar(gained,lost),
                    truncations={arm:sum(rows[arm,bench,mode][i]['truncated'] for i in ids) for arm in ('base','candidate')})
        # Sampled mean pass@1, bootstrap paired by problem rather than treating
        # repeated draws of a task as independent observations.
        result[bench]['sampled_mean_delta']={}
        for scope,ids in scopes.items():
            diff=np.array([sum(rows['candidate',bench,f'think-s{s}'][i]['correct']-
                              rows['base',bench,f'think-s{s}'][i]['correct'] for s in range(3))/3 for i in ids])
            rng=np.random.default_rng(0); boots=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(10000)])
            result[bench]['sampled_mean_delta'][scope]=dict(estimate=float(diff.mean()),
                ci95=list(map(float,np.percentile(boots,[2.5,97.5]))))
    from distillkit.independent_eval import compare_results
    for task in ('mmlu','arc'):
        result[task]=[dict(row,comparison='candidate - base') for row in
            compare_results(read(OUT/'candidate'/(task+'.json')),read(OUT/'base'/(task+'.json')))
            if row['comparison']=='enabled - pre_retrofit']
    write(OUT/'comparison.json',dict(results=result,promotion='not performed',
        caution='Unadjusted exploratory intervals across multiple benchmarks; public exposure remains possible.'))


def run():
    phase3_run.OUT=OUT
    plan=verify();lock=OUT/'run.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    environment=dict(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    def lane(arm,device):
        cache=dict(environment,TORCHINDUCTOR_CACHE_DIR=str(OUT/('compile-cache-'+arm)))
        for job in plan['commands'][arm]:
            verify()
            phase3_run.invoke(job['label'],job['argv'],device,cache)
    try:
        status('generation_and_local_overlap_screen_running')
        # Reuse the established scanner and every-field/token-ID decoder; its
        # excluded map covers all supplied candidates, not only its 64-row sample.
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(lane,'base','0'),pool.submit(lane,'candidate','1'),
                pool.submit(phase3_run.invoke,'math-overlap',[str(HERE/'freeze_math_dev.py'),
                    '--candidates',str(OUT/'math-candidates.json'),'--output',str(OUT/'math-overlap.json')],'')]
            for future in futures:future.result()
        status('sandbox_code_scoring')
        verify()
        for job in plan['scoring']:
            phase3_run.invoke(job['label'],job['argv'],'')
        status('paired_comparison')
        sys.path.insert(0,str(CODE))
        compare()
        status('complete_review_required')
    except Exception as error:
        status('failed_review_required',error=str(error));raise
    finally:
        lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','verify','run','score'))
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--bench',choices=('humaneval','mbpp'))
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='verify':verify();print('Verified frozen gauntlet.',flush=True)
    elif args.stage=='score':score(args.directory,args.bench)
    else:run()
