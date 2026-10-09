# Assisted-by: Codex
"""Review code-task exposure and compare frozen scores on unmatched task subsets.

Only a copied HumanEval contract or an MBPP stem with matching concrete tests is
confirmed. Shared phrases and corpus labels alone never trigger an exclusion.
The outputs are a future-use supplement; frozen training inputs stay untouched.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from code_dataset_audit import OUT, ROOT, OfflineTeacherCache, grams, read, signature
from code_failure_audit import bootstrap, entry_point, ident

USER = re.compile(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', re.S)


def match_evidence(text, problem, bench):
    normalized = signature(text)
    entry = entry_point(problem)
    full_stem = signature(problem['prompt']) in normalized
    user = '\n'.join(USER.findall(text))
    result = dict(full_stem=full_stem, entry_point=entry,
                  stem_in_user_turn=signature(problem['prompt']) in signature(user))
    if bench == 'humaneval':
        tree = ast.parse(problem['prompt'])
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        contract = ast.get_docstring(node)
        copied = bool(contract and signature(contract) in normalized)
        definition = bool(re.search(r'\bdef\s+' + re.escape(entry) + r'\s*\(', text))
        result.update(copied_contract=copied, matching_definition=definition,
                      confirmed=copied and definition,
                      reason='Copied full function contract and matching function definition'
                      if copied and definition else 'Shared phrase without the complete matching contract')
    else:
        tests = sum(signature(test) in normalized for test in problem['test_list'])
        interface = bool(re.search(r'\b' + re.escape(entry) + r'\s*\(', text))
        confirmed = full_stem and interface and tests >= 2
        result.update(matching_concrete_tests=tests, matching_interface=interface,
                      confirmed=confirmed,
                      reason='Matching task stem, interface and at least two concrete benchmark tests'
                      if confirmed else 'Generic or modified task; matching concrete tests not established')
    return result


def screened_scores(gauntlet, review, source_hashes):
    changes_path = gauntlet / 'failure-audit/final-answer-comparison.json'
    source_hashes[str(changes_path)] = hashlib.sha256(changes_path.read_bytes()).hexdigest()
    changes = read(changes_path)['changes']
    outcomes = {}
    for bench in ('humaneval', 'mbpp'):
        for arm in ('base', 'candidate'):
            for mode in ('nothink', 'think-s0', 'think-s1', 'think-s2'):
                path = gauntlet / arm / (bench + '-' + mode) / 'eval_results.json'
                source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                outcomes[arm, bench, mode] = {
                    ident(r['task_id'], bench): r for r in read(path)['results']}
    for change in changes:
        outcomes[change['arm'], change['bench'], change['mode']][change['task_id']] = change['final_answer']

    def passed(score):
        return int(score['base'] == 'pass' and score['plus'] == 'pass')

    results = {}
    for label, selected_only in (('recent_prefix_unmatched', True), ('retained_pool_unmatched', False)):
        results[label] = {}
        for bench in ('humaneval', 'mbpp'):
            excluded = {ident(r['benchmark'].split(':', 1)[1], bench) for r in review
                        if r['confirmed'] and r['benchmark'].startswith(bench + ':')
                        and (not selected_only or r['selected'])}
            inventory = set(outcomes['base', bench, 'think-s0'])
            if not excluded <= inventory:
                raise ValueError('review contains unknown benchmark task')
            ids = sorted(inventory - excluded)
            modes = {}
            for mode in ('nothink', 'think-s0', 'think-s1', 'think-s2'):
                a, b = outcomes['base', bench, mode], outcomes['candidate', bench, mode]
                if set(a) != inventory or set(b) != inventory:
                    raise ValueError('score inventory mismatch')
                modes[mode] = dict(cases=len(ids), base=sum(passed(a[i]) for i in ids),
                                   candidate=sum(passed(b[i]) for i in ids),
                                   gained=sum(passed(b[i]) and not passed(a[i]) for i in ids),
                                   lost=sum(passed(a[i]) and not passed(b[i]) for i in ids))
            diffs = [sum(passed(outcomes['candidate', bench, f'think-s{s}'][i]) -
                         passed(outcomes['base', bench, f'think-s{s}'][i]) for s in range(3))/3 for i in ids]
            greedy = [passed(outcomes['candidate', bench, 'nothink'][i]) -
                      passed(outcomes['base', bench, 'nothink'][i]) for i in ids]
            results[label][bench] = dict(excluded_task_ids=sorted(excluded), cases=len(ids), modes=modes,
                sampled_base=sum(modes[f'think-s{s}']['base'] for s in range(3))/(3*len(ids)),
                sampled_candidate=sum(modes[f'think-s{s}']['candidate'] for s in range(3))/(3*len(ids)),
                sampled_mean_delta=bootstrap(diffs), greedy_delta=bootstrap(greedy))
    return dict(results=results, source_sha256=source_hashes,
                caveats=['Final-answer diagnostic scores, with no additional generation or code execution.',
                         'Unmatched refers only to the audited retained training pool or recent prefix.',
                         'Older student and foundation-model exposure is not ruled out.',
                         'Paired task bootstrap across three seeds; exploratory, unadjusted intervals.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-remainder', action='store_true', help='Screen the other ten replay caches on CPU')
    args = parser.parse_args()
    audit_path = OUT / 'audit.json'
    audit = read(audit_path)
    if not audit['full_retained_token_screen']:
        raise ValueError('first run code_dataset_audit.py --screen-code')
    hashes = dict(audit['source_sha256'])
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('audit input changed: ' + path)
    hashes[str(audit_path)] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run = read(ROOT / 'scratch/dense_gr/completion-v2/control/train.json')['run_args']
    excluded = set(read(Path(run['exclude_documents'])))
    visits = Counter(d for group in read(Path(run['ordered_batches']))['groups'] for d in group['documents'])
    gauntlet = OUT.parent
    datasets = read(gauntlet / 'datasets.json')
    problems = {bench + ':' + str(row['task_id']): row for bench in ('humaneval', 'mbpp') for row in datasets[bench]}
    bank = defaultdict(set)
    for task, row in problems.items():
        for gram in grams(row['prompt']):
            bank[gram].add(task)
    short = {task: signature(row['prompt']) for task, row in problems.items()
             if len(signature(row['prompt']).split()) < 13}
    flags_by_source = defaultdict(lambda: defaultdict(set))
    for flag in audit['retained_cache_code_overlap_flags']:
        flags_by_source[flag['source']][flag['doc_id']].add(flag['benchmark'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'scratch/dense_gr/merges-long1/u50', local_files_only=True)
    review, inventories = [], []
    for path in map(Path, run['teacher_cache']):
        source = path.name.removeprefix('teacher-cache-')
        initial = source in audit['sources']
        if not initial and not args.screen_remainder:
            continue
        manifest_path = path / 'manifest.json'
        hashes[str(manifest_path)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        documents = [d for d in read(manifest_path)['documents']
                     if d['split'] == 'train' and d['doc_id'] not in excluded]
        cache = OfflineTeacherCache(path)
        scanned, tokens, matches = 0, 0, 0
        for document in documents:
            doc = document['doc_id']
            if initial and doc not in flags_by_source[source]:
                continue
            ids = cache.read_document(doc, tokens_only=True)['input_ids']
            if hashlib.sha256(ids.tobytes()).hexdigest() != document['sha256']['input_ids']:
                raise ValueError('cached input token hash mismatch: ' + doc)
            text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
            scanned += 1
            tokens += len(ids)
            if initial:
                candidates = flags_by_source[source][doc]
            else:
                candidates = set().union(*(bank.get(g, set()) for g in grams(text)))
                normalized = signature(text)
                candidates |= {task for task, stem in short.items() if stem in normalized}
            for task in sorted(candidates):
                evidence = match_evidence(text, problems[task], task.split(':')[0])
                review.append(dict(source=source, doc_id=doc, benchmark=task, selected=doc in visits,
                                   input_ids_sha256=document['sha256']['input_ids'], **evidence))
                matches += int(evidence['confirmed'])
        cache.close()
        inventories.append(dict(source=source, retained_documents=len(documents), retained_tokens=sum(d['length'] for d in documents),
                                decoded_this_review=scanned, tokens_decoded=tokens, initial_audit=initial, confirmed_matches=matches))
        print(source, 'decoded', scanned, 'confirmed', matches, flush=True)
    confirmed_ids = sorted({r['doc_id'] for r in review if r['confirmed']})
    supplement_path = OUT / 'exclude-code-confirmed-20261008.json'
    supplement_path.write_text(json.dumps(confirmed_ids, indent=2), encoding='utf-8')
    report = dict(policy='Exclude confirmed matching tasks only; preserve historical frozen inputs.',
                  screened_all_replay_caches=args.screen_remainder, inventories=inventories, review=review,
                  confirmed_document_count=len(confirmed_ids), confirmed_active_documents=sorted({r['doc_id'] for r in review if r['confirmed'] and r['selected']}),
                  source_sha256=hashes, supplement=str(supplement_path),
                  caveat='Lexical/contract review is not proof that all semantic paraphrases or older exposures are absent.')
    (OUT / 'overlap-review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    comparison = screened_scores(gauntlet, review, dict(hashes))
    (OUT / 'screened-comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    print('Confirmed documents:', len(confirmed_ids), 'active:', len(report['confirmed_active_documents']), flush=True)


if __name__ == '__main__':
    main()
