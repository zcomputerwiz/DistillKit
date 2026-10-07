# Assisted-by: Codex
"""Execute frozen phase-3 commands, retaining logs and resumable stage receipts.

Evaluation uses GPU 0 for the long agent suite; GPU 1 runs math and short agent suites.
Training owns both GPUs; paired retention runs on GPU 0. No promotion is performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'phase3-masking'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invoke(label, argv, devices):
    runs=OUT/'runs'
    runs.mkdir(parents=True,exist_ok=True)
    receipt=runs/(label+'.json')
    signature=hashlib.sha256(json.dumps(argv).encode()).hexdigest()
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous.get('status')=='complete':
            if previous['argv_sha256']!=signature:
                raise ValueError('completed command changed: '+label)
            print('Already complete:',label,flush=True)
            return
        raise ValueError('unfinished stage needs explicit inspection before retry: '+label)
    for flag in ('--output','--output-dir','--checkpoints'):
        if flag in argv:
            Path(argv[argv.index(flag)+1]).parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONIOENCODING='utf-8',
             PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=devices)
    info=dict(label=label,status='running',argv=argv,argv_sha256=signature,
              cuda_visible_devices=devices,started=datetime.now(timezone.utc).isoformat(),
              protocol_sha256=digest(OUT/'evaluation.json'),plan_sha256=digest(OUT/'plan.json'))
    if '--prefill-query-chunk' in argv:
        info['runtime_source_sha256']={name:digest(HERE/name) for name in
            ('bounded_cached_prefill.py','agentic_live_eval.py','agentic_scenarios.py')}
    with (runs/(label+'.log')).open('w',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,*argv],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        info['pid']=child.pid
        receipt.write_text(json.dumps(info,indent=2))
        print('Started:',label,'PID',child.pid,'GPU',devices,flush=True)
        code=child.wait()
    info.update(status='complete' if code==0 else 'failed',exit_code=code,
                finished=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(info,indent=2))
    if code:
        raise RuntimeError(f'{label} failed with exit code {code}; inspect its log')
    print('Completed:',label,flush=True)


def main(stage):
    spec=json.loads((OUT/'evaluation.json').read_text())
    plan=json.loads((OUT/'plan.json').read_text())
    for item in spec['assets']:
        if digest(item['path'])!=item['sha256']:
            raise ValueError('frozen asset changed: '+item['path'])
    if stage in ('baseline','candidate'):
        arm='base' if stage=='baseline' else 'masked'
        commands=spec['commands'][arm]
        def lane(secondary):
            # Long distractor prefills dominate greedy agent evaluation. Fill the
            # other GPU with math, then independent sampled trials while it runs.
            indices=([i for i,a in enumerate(commands) if Path(a[0]).name=='math_dev_eval.py']+
                     [i for i,a in enumerate(commands) if Path(a[0]).name=='agentic_live_eval.py']+
                     [i for i,a in reversed(list(enumerate(commands))) if '--sample' in a]) if secondary else [
                         i for i,a in enumerate(commands) if Path(a[0]).name=='agentic_scenarios.py' and '--sample' not in a]
            for index in indices:
                invoke(f'{arm}-{index:02d}',commands[index],'1' if secondary else '0')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(lane,math) for math in (False,True)]
            for future in futures:
                future.result()
    elif stage=='train':
        for index in range(len(spec['commands']['base'])):
            receipt=json.loads((OUT/'runs'/f'base-{index:02d}.json').read_text())
            if receipt['status']!='complete':
                raise ValueError('baseline incomplete')
        invoke('train-masked',plan['arms']['masked']['argv'],'0,1')
    else:
        for index,argv in enumerate(spec['paired_nll']):
            invoke(f'paired-nll-{index:02d}',argv,'0')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['baseline','candidate','train','paired'])
    main(p.parse_args().stage)
