# Assisted-by: Codex
"""Describe frozen code outcomes without executing model-generated code."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.DOTALL)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def ident(value, bench):
    value = str(value)
    return 'Mbpp/' + value if bench == 'mbpp' and not value.startswith('Mbpp/') else value


def entry_point(problem):
    if 'entry_point' in problem:
        return problem['entry_point']
    tree = ast.parse(problem['code'])
    return next(n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))


def final_answer_code(raw):
    """Use only the final channel when it exists; otherwise preserve old extraction.

    This does not complete, repair or execute the returned source.
    """
    text = raw.split('</think>', 1)[1] if '</think>' in raw else raw
    match = FENCE.search(text)
    return (match.group(1) if match else text).strip()


def features(row, score, entry, thinking=True):
    raw, code = row['raw'], row['code']
    fences = list(FENCE.finditer(raw))
    closing = raw.find('</think>')
    flags = []
    if row['truncated']:
        flags.append('truncated')
    if thinking and closing < 0:
        flags.append('no_think_close')
    if fences and closing >= 0 and fences[0].start() < closing:
        flags.append('first_fence_in_thinking')
    after = [m for m in fences if closing >= 0 and m.start() > closing]
    final_code = final_answer_code(raw) if closing >= 0 else None
    if final_code is not None and final_code != code:
        flags.append('different_final_code')
    syntax = None
    try:
        tree = ast.parse(code)
        definitions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        assignments = {n.id for stmt in tree.body if isinstance(stmt, (ast.Assign, ast.AnnAssign))
                       for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if entry not in definitions | assignments:
            flags.append('missing_entry_point')
    except SyntaxError as error:
        syntax = dict(message=error.msg, line=error.lineno)
        flags.append('syntax_error')
    correct = score['base'] == 'pass' and score['plus'] == 'pass'
    if correct:
        category = 'pass'
    elif thinking and row['truncated'] and closing < 0:
        category = 'cap_before_final'
    elif 'first_fence_in_thinking' in flags and 'different_final_code' in flags:
        category = 'extracted_thinking_instead_of_final'
    elif syntax:
        category = 'syntax_error'
    elif 'missing_entry_point' in flags:
        category = 'missing_entry_point'
    elif score['base'] in ('crash', 'timeout') or score['plus'] in ('crash', 'timeout'):
        category = 'scorer_' + score['base']
    elif score['base'] == 'pass':
        category = 'extended_test_failure'
    else:
        category = 'base_test_failure'
    ids = row.get('token_ids', [])
    think_tokens = ids.index(248069) if 248069 in ids else None
    return dict(correct=correct, category=category, flags=flags, syntax=syntax,
                tokens=row['generated_tokens'], think_tokens=think_tokens,
                final_tokens=len(ids)-think_tokens-1 if think_tokens is not None else None,
                final_code=final_code)


def bootstrap(values):
    values = np.array(values, dtype=float)
    rng = np.random.default_rng(0)
    draws = [rng.choice(values, len(values), replace=True).mean() for _ in range(10000)]
    return dict(estimate=float(values.mean()), ci95=np.percentile(draws, [2.5, 97.5]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gauntlet', type=Path, default=HERE/'completion-gauntlet')
    parser.add_argument('--output', type=Path, default=HERE/'completion-gauntlet/failure-audit')
    parser.add_argument('--score-final', action='store_true', help='Rescore changed extraction in the existing Docker sandbox')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    datasets = read(args.gauntlet/'datasets.json')
    evidence = {}; examples = []; source_hashes = {}
    for bench in ('humaneval', 'mbpp'):
        problems = {ident(p['task_id'], bench): p for p in datasets[bench]}
        features_by_mode = {}; evidence[bench] = {}
        for mode in ('nothink', 'think-s0', 'think-s1', 'think-s2'):
            pairs = {}; evidence[bench][mode] = {}
            for arm in ('base', 'candidate'):
                directory = args.gauntlet/arm/(bench+'-'+mode)
                for filename in ('completions.jsonl', 'eval_results.json'):
                    path = directory/filename
                    source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                records = {ident(r['task_id'], bench): r for r in
                           map(json.loads, (directory/'completions.jsonl').read_text(encoding='utf-8-sig').splitlines())}
                scores = {ident(s['task_id'], bench): s for s in read(directory/'eval_results.json')['results']}
                if records.keys() != scores.keys() or records.keys() != problems.keys():
                    raise ValueError('incomplete case inventory')
                stats = {tid: features(r, scores[tid], entry_point(problems[tid]), mode!='nothink') for tid, r in records.items()}
                pairs[arm] = records, scores, stats
                lengths = [s['tokens'] for s in stats.values()]
                thinking = [s['think_tokens'] for s in stats.values() if s['think_tokens'] is not None]
                evidence[bench][mode][arm] = dict(categories=dict(Counter(s['category'] for s in stats.values())),
                    flags=dict(Counter(f for s in stats.values() for f in s['flags'])),
                    tokens=dict(mean=float(np.mean(lengths)), median=float(np.median(lengths)), p90=float(np.percentile(lengths,90))),
                    closed_think_tokens=dict(cases=len(thinking), mean=float(np.mean(thinking)) if thinking else None,
                                             median=float(np.median(thinking)) if thinking else None))
            ar, ass, a = pairs['base']; br, bss, b = pairs['candidate']
            changed = Counter(); excluded = Counter()
            both_nontruncated = [tid for tid in a if not ar[tid]['truncated'] and not br[tid]['truncated']]
            for tid in a:
                if ar[tid]['prompt_sha256'] != br[tid]['prompt_sha256']:
                    raise ValueError('prompt mismatch')
                direction = 'lost' if a[tid]['correct'] and not b[tid]['correct'] else 'gained' if b[tid]['correct'] and not a[tid]['correct'] else None
                if direction:
                    failed = b[tid] if direction == 'lost' else a[tid]
                    changed[direction + ':' + failed['category']] += 1
                    if tid not in both_nontruncated:
                        excluded[direction] += 1
                    examples.append(dict(bench=bench, mode=mode, task_id=tid, direction=direction,
                        problem=problems[tid]['prompt'], entry_point=entry_point(problems[tid]),
                        base=dict(row=ar[tid], score=ass[tid], features=a[tid]),
                        candidate=dict(row=br[tid], score=bss[tid], features=b[tid])))
            evidence[bench][mode]['discordant_categories'] = dict(changed)
            evidence[bench][mode]['both_nontruncated'] = dict(cases=len(both_nontruncated),
                base=sum(a[i]['correct'] for i in both_nontruncated), candidate=sum(b[i]['correct'] for i in both_nontruncated),
                gained=sum(not a[i]['correct'] and b[i]['correct'] for i in both_nontruncated),
                lost=sum(a[i]['correct'] and not b[i]['correct'] for i in both_nontruncated),
                excluded_discordant=dict(excluded))
            features_by_mode[mode] = pairs
        # Conditional diagnostics select on output length and are descriptive,
        # not an unbiased estimate of an intervention on the output budget.
        common = [tid for tid in problems if all(not features_by_mode[f'think-s{s}'][arm][0][tid]['truncated']
                                                 for s in range(3) for arm in ('base','candidate'))]
        diff = [sum(int(features_by_mode[f'think-s{s}']['candidate'][2][tid]['correct'])-
                    int(features_by_mode[f'think-s{s}']['base'][2][tid]['correct']) for s in range(3))/3 for tid in common]
        evidence[bench]['all_six_nontruncated_sampled'] = dict(cases=len(common), **bootstrap(diff))
    (args.output/'summary.json').write_text(json.dumps(dict(results=evidence, source_sha256=source_hashes,
        caveat='Descriptive classifications; no generated code executed. Length-conditioned subsets are selected outcomes.'), indent=2), encoding='utf-8')
    with (args.output/'discordant.jsonl').open('w', encoding='utf-8') as handle:
        for row in examples:
            handle.write(json.dumps(row)+'\n')
    if args.score_final:
        score_final(args)
    print(json.dumps(evidence, indent=2))


def score_final(args):
    """Keep original scores and outputs; copy changed extraction to separate subsets."""
    plan = read(args.gauntlet/'plan.json')
    image = subprocess.check_output(['docker','image','inspect','code-bench-sandbox','--format','{{.Id}}'],text=True).strip()
    if image != plan['code_sandbox_image']:
        raise ValueError('sandbox image differs from frozen gauntlet')
    scorer_hashes = {}
    for name in ('run_docker.ps1','robust_eval.py'):
        path = HERE.parent/'downstream/code_bench'/name
        scorer_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if scorer_hashes[str(path)] != plan['assets'][str(path)]:
            raise ValueError('scorer differs from frozen gauntlet')
    source_hashes = {}; outcomes = {}; changes = []; final_features = {}; original_records = {}
    datasets = read(args.gauntlet/'datasets.json')
    for bench in ('humaneval', 'mbpp'):
        for mode in ('think-s0', 'think-s1', 'think-s2'):
            for arm in ('base', 'candidate'):
                source = args.gauntlet/arm/(bench+'-'+mode)
                records = list(map(json.loads, (source/'completions.jsonl').read_text(encoding='utf-8-sig').splitlines()))
                scores = {ident(s['task_id'],bench): s for s in read(source/'eval_results.json')['results']}
                changed = [dict(r, code=final_answer_code(r['raw'])) for r in records
                           if final_answer_code(r['raw']) != r['code']]
                target = args.output/'final-answer-scoring'/(arm+'-'+bench+'-'+mode)
                target.mkdir(parents=True, exist_ok=True)
                payload = ''.join(json.dumps(r)+'\n' for r in changed)
                path = target/'completions.jsonl'
                if path.exists() and path.read_text(encoding='utf-8') != payload:
                    raise ValueError('refusing to change diagnostic inputs')
                if not path.exists():
                    path.write_text(payload,encoding='utf-8')
                source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                if changed and not (target/'eval_results.json').exists():
                    print('Rescoring final answers:',target.name,len(changed),flush=True)
                    with (target/'sandbox.log').open('w',encoding='utf-8') as log:
                        subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(HERE.parent/'downstream/code_bench/run_docker.ps1'),
                                        str(target),bench],check=True,stdout=log,stderr=subprocess.STDOUT)
                rescored = {ident(s['task_id'],bench): s for s in read(target/'eval_results.json')['results']} if changed else {}
                if set(rescored) != {ident(r['task_id'],bench) for r in changed}:
                    raise ValueError('diagnostic scoring inventory mismatch')
                for tid,s in rescored.items():
                    old = scores[tid]
                    changes.append(dict(bench=bench,mode=mode,arm=arm,task_id=tid,original=old,final_answer=s))
                    scores[tid] = s
                outcomes[arm,bench,mode] = scores
                problems = {ident(p['task_id'],bench):p for p in datasets[bench]}
                original_records[arm,bench,mode] = {ident(r['task_id'],bench):r for r in records}
                final_features[arm,bench,mode] = {ident(r['task_id'],bench):features(
                    dict(r,code=final_answer_code(r['raw'])),scores[ident(r['task_id'],bench)],
                    entry_point(problems[ident(r['task_id'],bench)])) for r in records}
    results = {}
    for bench in ('humaneval','mbpp'):
        rows = {};diffs = []
        for mode in ('think-s0','think-s1','think-s2'):
            a,b = outcomes['base',bench,mode], outcomes['candidate',bench,mode]
            def passed(s):return int(s['base']=='pass' and s['plus']=='pass')
            rows[mode] = dict(cases=len(a),base=sum(map(passed,a.values())),candidate=sum(map(passed,b.values())),
                             gained=sum(passed(b[i]) and not passed(a[i]) for i in a),
                             lost=sum(passed(a[i]) and not passed(b[i]) for i in a))
            rows[mode]['discordant_categories'] = dict(Counter(
                ('lost:' + final_features['candidate',bench,mode][tid]['category']) if passed(a[tid]) else
                ('gained:' + final_features['base',bench,mode][tid]['category'])
                for tid in a if passed(a[tid]) != passed(b[tid])))
        ids = list(outcomes['base',bench,'think-s0'])
        diffs = [sum(passed(outcomes['candidate',bench,f'think-s{s}'][tid])-
                     passed(outcomes['base',bench,f'think-s{s}'][tid]) for s in range(3))/3 for tid in ids]
        nontruncated = [tid for tid in ids if all(not original_records[arm,bench,f'think-s{s}'][tid]['truncated']
                                                for s in range(3) for arm in ('base','candidate'))]
        nontruncated_diffs = [sum(passed(outcomes['candidate',bench,f'think-s{s}'][tid])-
                                 passed(outcomes['base',bench,f'think-s{s}'][tid]) for s in range(3))/3 for tid in nontruncated]
        results[bench] = dict(seeds=rows,sampled_mean_delta=bootstrap(diffs),
            all_six_nontruncated_sampled=dict(cases=len(nontruncated),**bootstrap(nontruncated_diffs)))
    (args.output/'final-answer-comparison.json').write_text(json.dumps(dict(results=results,changes=changes,
        diagnostic_input_sha256=source_hashes,code_sandbox_image=image,scorer_sha256=scorer_hashes,
        caveat='Separate diagnostic using final channel, no code repairs; original frozen gauntlet unchanged.'),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
