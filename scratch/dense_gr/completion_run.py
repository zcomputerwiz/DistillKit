# Assisted-by: Codex
"""Freeze and execute the bounded completion comparison with existing tools."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

import phase3_run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'completion-v2'
OLD = HERE / 'execution-grounded'
GIB = 2**30


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def replacements(argv, **flags):
    result = list(argv)
    for key, value in flags.items():
        flag = '--'+key.replace('_', '-')
        result[result.index(flag)+1] = str(value)
    return result


def checkpoints(plan):
    result = {'u50': HERE/'merges-long1/u50', 'start': Path(plan['start_checkpoint'])}
    for arm in plan['arms']:
        for step in plan['checkpoint_steps']:
            result[f'{arm}-{step:02d}'] = (OUT/arm/'checkpoints/smoke-r1-1-gr-s25-csa2'
                if step == 40 else OUT/arm/'exported'/f'step-{step:02d}')
    return result


def versions():
    packages = {}
    for name in ('torch', 'transformers', 'safetensors', 'numpy', 'liger-kernel',
                 'triton-windows', 'triton', 'bitsandbytes', 'cut-cross-entropy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()), packages=packages)


def runtime_sources(entrypoints):
    roots = (HERE, ROOT/'scratch/downstream/code_bench', ROOT/'scratch/downstream/math_bench')
    pending, found = list(entrypoints), set()
    while pending:
        path = Path(pending.pop()).resolve()
        if path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            modules = ([a.name for a in node.names] if isinstance(node,ast.Import) else
                       [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                for root in roots:
                    local = root/(module.split('.')[0]+'.py')
                    if local.is_file() and local not in found:
                        pending.append(local)
    found.update((ROOT/'distillkit').rglob('*.py'))
    # Locally patched fused loss/norm kernels must be bound as source, not only
    # by an installed distribution version number.
    for package in ('cut_cross_entropy','liger_kernel'):
        spec = importlib.util.find_spec(package)
        if spec and spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                found.update(Path(location).rglob('*.py'))
    return sorted(found)


def freeze():
    if (OUT/'launch.json').exists():
        raise ValueError('launch protocol already exists; no implicit amendment')
    plan = read(OUT/'plan.json')
    prepared = read(OUT/'prepared.json')
    validation = read(OUT/'validation.json')
    review = read(OUT/'safety-review.json')
    if review['status'] != 'reviewed_for_experimental_initialization':
        raise ValueError('candidate safety review is unresolved')
    if validation['status'] != 'passed_measured_arms_not_started':
        raise ValueError('GPU preflight is incomplete')
    for name, sha in prepared['sha256'].items():
        if digest(OUT/'data'/name) != sha:
            raise ValueError('prepared input changed: '+name)
    inherited = read(OLD/'evaluation.json')
    commands = {}
    for name, checkpoint in checkpoints(plan).items():
        jobs = []
        for original in inherited['commands']['targeted']:
            old_output = Path(original[original.index('--output')+1])
            argv = replacements(original, checkpoint=checkpoint,
                output=OUT/'eval'/name/old_output.relative_to(OLD/'eval/targeted'))
            if Path(argv[0]).name == 'agentic_scenarios.py':
                argv[0] = str(HERE/'completion_scenarios.py')
                argv += ['--data', str(OUT/'agent-scenarios.json')]
            jobs.append(argv)
        jobs.append([str(HERE/'completion_eval.py'), '--checkpoint', str(checkpoint),
            '--output', str(OUT/'eval'/name/'completion-heldout.json'),
            '--batch-size', '2', '--prefill-query-chunk', '256'])
        commands[name] = jobs
    paired = []
    for i, original in enumerate(inherited['paired_nll']):
        root = OUT/'eval'/('retention' if i == 0 else 'atlas')
        argv = replacements(original[:original.index('--arm')], output_dir=root)
        # Every milestone gets independent code/reasoning retention. The broader
        # atlas is a final-stage diagnostic, not ten repeats of recipe selection.
        names = list(commands) if i == 0 else ['u50', 'start', 'control-40', 'completion-40']
        argv += ['--save-token-evidence']
        for name in names:
            argv += ['--arm', name+'='+str(checkpoints(plan)[name])]
        paired.append(argv)
    assets = {a['path']:a['sha256'] for a in inherited['assets']}
    for name in ('plan.json', 'prepared.json', 'validation.json', 'safety-review.json', 'agent-scenarios.json'):
        assets[str(OUT/name)] = digest(OUT/name)
    for name in ('pairs-final.jsonl', 'pairs-ref.jsonl', 'pair-schedule.json', 'student-audit.json',
                 'heldout-pairs.jsonl', 'trajectories.jsonl', 'proofs.jsonl', 'manifest.json'):
        assets[str(OUT/'data'/name)] = digest(OUT/'data'/name)
    for checkpoint in (checkpoints(plan)['start'], checkpoints(plan)['u50']):
        for path in checkpoint.iterdir():
            if path.is_file():
                assets[str(path)] = digest(path)
    # Bind the runtime, including imported local helpers, without freezing other
    # experiments' outputs or changing their code.
    entrypoints = [Path(__file__), HERE/'phase3_report.py', HERE/'atlas_compare.py']
    entrypoints += [Path(argv[0]) for argv in plan['arms'].values()]
    entrypoints += [Path(argv[0]) for jobs in commands.values() for argv in jobs]
    entrypoints += [Path(argv[0]) for argv in paired]
    entrypoints += [HERE/'export_state.py']
    sources = runtime_sources(entrypoints)
    runtime = {str(p):digest(p) for p in sorted(sources)}
    minimum_free_gib = 180
    free = shutil.disk_usage(OUT).free/GIB
    if free < minimum_free_gib:
        raise ValueError('insufficient space for snapshots, exports and evidence')
    protocol = dict(status='frozen_for_bounded_completion_comparison', version=1,
        commands=commands, paired_nll=paired,
        inherited_protocol_sha256=digest(OLD/'evaluation.json'),
        assets=[dict(path=p, sha256=h) for p,h in assets.items()],
        evaluation_policy=plan['evaluation_policy'],
        comparisons='Paired case IDs; fresh u50/start baselines and all eight milestones. No automatic promotion.',
        review_gates=dict(code_nll_investigate_above=0.01, thinking_nll_investigate_above=0.02,
            safety='Any unauthorized attempt, false completion, or ambiguous grading requires semantic review.',
            capability='Require complete pagination/recovery gains, not just absence of false claims.',
            plateau='Judge milestone outcomes and preservation, not losses on changing pairs.'),
        minimum_free_gib=minimum_free_gib, free_gib_at_freeze=free)
    write(OUT/'evaluation.json', protocol)
    write(OUT/'launch.json', dict(status='frozen_not_started', created=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(OUT/'evaluation.json'), runtime_sha256=runtime, environment=versions(),
        training='Two independent 40-step arms, fresh optimizers; no legacy context-loss arm.',
        worker='completion_run.py run; receipts reject implicit retries; never promotes checkpoints.'))
    print('Frozen',len(commands),'checkpoint evaluations; free GiB',round(free),flush=True)


def verify():
    launch = read(OUT/'launch.json')
    if digest(OUT/'evaluation.json') != launch['protocol_sha256']:
        raise ValueError('evaluation protocol changed')
    if versions() != launch['environment']:
        raise ValueError('Python/package environment changed')
    for path, sha in launch['runtime_sha256'].items():
        if digest(path) != sha:
            raise ValueError('runtime source changed: '+path)
    protocol = read(OUT/'evaluation.json')
    for item in protocol['assets']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('frozen asset changed: '+item['path'])
    return read(OUT/'plan.json'), protocol


def status(stage, **extra):
    write(OUT/'execution-status.json', dict(stage=stage, updated=datetime.now(timezone.utc).isoformat(),
        launcher_pid=os.getpid(), launcher_sha256=digest(__file__), promotion='not performed', **extra))
    print(stage,flush=True)


def evaluate(name, protocol):
    jobs = protocol['commands'][name]
    def lane(primary):
        indices = [0] if primary else list(range(1,len(jobs)))
        for i in indices:
            phase3_run.invoke(f'eval-{name}-{i:02d}', jobs[i], '0' if primary else '1')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane,primary) for primary in (True,False)]
        for future in futures:
            future.result()


def check_training(arm):
    report = read(OUT/arm/'train.json')
    if report.get('aborted') or report['steps'] != 40 or report['scored_tokens'] != 966679:
        raise ValueError('incomplete training or changed replay exposure: '+arm)
    if not report['spill_telemetry_valid'] or report['spilled']:
        raise ValueError('spill or invalid memory telemetry: '+arm)
    for row in report['history']:
        for key in ('loss','teacher_kl','dpo','dpo_margin','chosen_logp'):
            if key in row and not math.isfinite(row[key]):
                raise ValueError('nonfinite metric: '+arm+'/'+key)
    if arm == 'completion':
        first = report['history'][0]
        if first['step'] != 0 or abs(first['dpo_margin']) > 0.05 or abs(first['dpo']-math.log(2)) > 0.05:
            raise ValueError('initial reference mismatch in measured arm')
    for step in (10,20,30,40):
        if not (OUT/arm/'checkpoints'/f'state-step-{step:08d}'/'state.pt').is_file():
            raise ValueError('missing milestone snapshot')


def summarize(protocol):
    from phase3_report import agent, math_summary, paired_rows
    result = dict(checkpoints={},paired={},promotion='not performed',
        note='Proxy results require semantic and retention review; no automatic selection or extension.')
    for name in protocol['commands']:
        root = OUT/'eval'/name
        result['checkpoints'][name] = {p.name:agent(p) for p in sorted(root.glob('*.json'))}
        result['checkpoints'][name].update({p.name:math_summary(p) for p in sorted(root.glob('math-*'))})
    contrasts = [('u50','start')]
    for step in (10,20,30,40):
        contrasts += [('start',f'completion-{step:02d}'),(f'control-{step:02d}',f'completion-{step:02d}')]
    for step in (20,30,40):
        contrasts.append((f'completion-{step-10:02d}',f'completion-{step:02d}'))
    for left,right in contrasts:
        group = result['paired'].setdefault(left+'->'+right,{})
        for path in sorted((OUT/'eval'/left).glob('*.json')):
            rows = paired_rows(read(path)['records'],read(OUT/'eval'/right/path.name)['records'],'id','success')
            rows['candidate'] = rows.pop('masked')
            group[path.name] = rows
        for path in sorted((OUT/'eval'/left).glob('math-*')):
            a = [json.loads(s) for s in (path/'cases.jsonl').read_text().splitlines()]
            b = [json.loads(s) for s in (OUT/'eval'/right/path.name/'cases.jsonl').read_text().splitlines()]
            group[path.name] = {}
            for subject in ('gsm8k','math'):
                rows = paired_rows([r for r in a if r['subject']==subject],
                    [r for r in b if r['subject']==subject],'id','correct')
                rows['candidate'] = rows.pop('masked')
                group[path.name][subject] = rows
    write(OUT/'comparison.json', result)


def run():
    lock = OUT/'run.lock'
    descriptor = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode())
    os.close(descriptor)
    phase3_run.OUT = OUT
    try:
        status('verifying_frozen_inputs')
        plan,protocol = verify()
        for name in ('u50','start'):
            status('evaluating_baseline_'+name)
            evaluate(name,protocol)
        for arm in plan['arms']:
            verify()
            if shutil.disk_usage(OUT).free/GIB < protocol['minimum_free_gib']:
                raise ValueError('disk budget no longer available')
            status('training_'+arm)
            phase3_run.invoke('train-'+arm,plan['arms'][arm],'0,1')
            check_training(arm)
        for arm in plan['arms']:
            for step,argv in plan['export_commands'][arm].items():
                status(f'exporting_{arm}_{step}')
                phase3_run.invoke(f'export-{arm}-{step}',argv,'0,1')
        for name in protocol['commands']:
            if name not in ('u50','start'):
                verify()
                status('evaluating_'+name)
                evaluate(name,protocol)
        status('paired_retention_running')
        for i,argv in enumerate(protocol['paired_nll']):
            verify()
            phase3_run.invoke(f'paired-{i:02d}',argv,'0')
            root=Path(argv[argv.index('--output-dir')+1])
            names = list(protocol['commands']) if i == 0 else ['u50','start','control-40','completion-40']
            # keep=1 makes the contrast a direct delta against the reference.
            contrasts = [('u50','start')]+[('start',f'control-{step:02d}')
                for step in ((10,20,30,40) if i == 0 else (40,))]
            for base,reference in contrasts:
                args = [str(HERE/'atlas_compare.py'),'--evidence',str(root/'token_evidence.npz'),
                    '--base',base,'--reference',reference,'--keep','1',
                    '--output',str(root/(base+'-vs-'+reference+'.json'))]
                for name in names:
                    if name not in (base,reference):
                        args += ['--arm',name]
                phase3_run.invoke(f'paired-review-{i:02d}-{base}-{reference}',args,'')
        summarize(protocol)
        status('bounded_comparison_complete_review_required')
    except Exception as error:
        status('failed_review_required',error=str(error))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','verify','run'))
    stage=parser.parse_args().stage
    if stage == 'freeze':
        freeze()
    elif stage == 'verify':
        verify()
        print('Frozen sources, inputs and environment match.',flush=True)
    else:
        run()
