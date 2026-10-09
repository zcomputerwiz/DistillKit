# Assisted-by: Codex
"""Audit retained code/reasoning caches and the recent frozen replay, on CPU."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))
from distillkit.offline_cache import OfflineTeacherCache

DATA = ROOT.parent/'capture-data'
OUT = HERE/'completion-gauntlet/dataset-audit'
WORDS = re.compile(r'\w+')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig') as handle:
        yield from map(json.loads,handle)


def signature(text):
    return ' '.join(WORDS.findall(text.lower()))


def grams(text):
    words = signature(text).split()
    return {' '.join(words[i:i+13]) for i in range(len(words)-12)}


def recheck_extraction():
    """Recheck only changed final-fence selection with the existing CPU sandbox."""
    import subprocess
    sys.path.insert(0, str(ROOT/'scratch/downstream/code_bench'))
    from verify_code import BLOCK, sandbox, solution_of
    from code_failure_audit import FENCE, final_answer_code
    exclusion = set(read(OUT/'exclude-replay-next-20261008.json'))
    args = read(HERE/'completion-v2/control/train.json')['run_args']
    visits = {doc for group in read(Path(args['ordered_batches']))['groups'] for doc in group['documents']}
    samples, result = [], {}
    hashes = {}
    for source, filename, prompt_file in (
            ('teacher-code', 'teacher-code-verified.jsonl', 'code-prompts-teacher.jsonl'),
            ('r8-code-short-w8', 'onpolicy-r8-code-short.jsonl', 'code-prompts-r8.jsonl')):
        paths = [DATA/filename, DATA/prompt_file, ROOT.parent/('teacher-cache-'+source)/'manifest.json']
        for path in paths:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        prompts = {r['doc_id']: r for r in rows(DATA/prompt_file)}
        retained = {d['doc_id'] for d in read(paths[-1])['documents']
                    if d['split']=='train' and d['doc_id'] not in exclusion}
        counts, changed = Counter(), []
        for row in rows(DATA/filename):
            doc = 'tcode:'+row['doc_id'] if source=='teacher-code' else row['doc_id']
            if doc not in retained:
                continue
            reply = row['text'][row['prompt_chars']:]
            final = reply.split('</think>', 1)[1] if '</think>' in reply else reply
            blocks = list(FENCE.finditer(final))
            counts['documents'] += 1
            counts['single_final_fence' if len(blocks)==1 else 'multiple_or_no_final_fences'] += 1
            old, first = (solution_of(reply) or '').strip(), final_answer_code(reply)
            if old == first:
                continue
            key = re.sub(r'^onpolicy:|:s\d+$|:greedy$', '', row['doc_id'])
            concrete = prompts[key]['tests']
            for kind, code in (('historical_last', old), ('first_final', first)):
                samples.append(dict(id=doc+'#'+kind, solution=code, test=concrete))
            named = re.findall(r'```(?:python|py)\s*\n(.*?)```', final, re.S)
            changed.append(dict(doc_id=doc, selected=doc in visits, final_fences=len(blocks),
                first_final_sha256=hashlib.sha256(first.encode()).hexdigest(),
                historical_last_sha256=hashlib.sha256(old.encode()).hexdigest(),
                first_explicit_python_equals_verified_last=bool(named and named[0].strip()==old)))
        result[source] = dict(counts=dict(counts), changed_extraction=changed)
    image = subprocess.check_output(['docker','image','inspect','code-verify-sandbox','--format','{{.Id}}'], text=True).strip()
    status = sandbox(samples) if samples else {}
    if set(status) != {s['id'] for s in samples}:
        raise ValueError('verification inventory mismatch')
    for pool in result.values():
        for row in pool['changed_extraction']:
            row['historical_last_status'] = status[row['doc_id']+'#historical_last']
            row['first_final_status'] = status[row['doc_id']+'#first_final']
    for path in (Path(__file__), ROOT/'scratch/downstream/code_bench/verify_code.py',
                 ROOT/'scratch/downstream/code_bench/verify_runner.py'):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'verification-extraction-audit.json').write_text(json.dumps(dict(pools=result,
        source_sha256=hashes, sandbox_image=image,
        caveat='Rechecked changed extractions only. Existing labels/captures unchanged; no model generation.'), indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-code',action='store_true',help='Decode retained tokens and flag code benchmark stem overlap')
    parser.add_argument('--recheck-extraction',action='store_true',help='Verify changed first/last final fences in the existing Docker sandbox')
    options = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if options.recheck_extraction:
        recheck_extraction()
        return
    hashes = {}
    def bound(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    args = read(bound(HERE/'completion-v2/control/train.json'))['run_args']
    excluded = set(read(bound(Path(args['exclude_documents']))))
    replay = read(bound(Path(args['ordered_batches'])))
    visits = Counter(doc for row in replay['groups'] for doc in row['documents'])
    source_files = dict(frontier_code_raw='frontier-code-raw.jsonl', teacher_code='teacher-code-verified.jsonl',
                        expand_code_w8='expand-code.jsonl',r8_code_short_w8='onpolicy-r8-code-short.jsonl',
                        thinking_w8='thinking-code-math.jsonl',think_first_w8='think-first-5m.jsonl')
    metadata = {}
    for source,filename in source_files.items():
        mapping = {}
        for row in rows(bound(DATA/filename)):
            doc = ('tcode:'+row['doc_id']) if source=='teacher_code' else row['doc_id']
            mapping[doc] = {k:v for k,v in row.items() if k not in ('text','input_ids','prompt','tests','reference')}
        metadata[source] = mapping
    teacher_prompts = {r['doc_id']:r for r in rows(bound(DATA/'code-prompts-teacher.jsonl'))}
    teacher_judge = {r['id']:r for r in rows(bound(DATA/'teacher-code-index-r5.jsonl'))}
    r8_prompts = {r['doc_id']:r for r in rows(bound(DATA/'code-prompts-r8.jsonl'))}
    bench = read(bound(HERE/'completion-gauntlet/datasets.json'))
    exact = {};bank = {}; benchmark_text = {}; benchmark_grams = {}
    for name in ('humaneval','mbpp'):
        for row in bench[name]:
            tid = name+':'+str(row['task_id'])
            benchmark_text[tid] = signature(row['prompt'])
            benchmark_grams[tid] = grams(row['prompt'])
            exact.setdefault(signature(row['prompt']),set()).add(tid)
            for gram in grams(row['prompt']):
                bank.setdefault(gram,set()).add(tid)
    tokenizer = None
    if options.screen_code:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(HERE/'merges-long1/u50',local_files_only=True)
    overlaps = []; cache_overlaps = []; code_prompt_test_names = {}; report = {}
    for source in source_files:
        name = source.replace('_','-')
        path = ROOT.parent/('teacher-cache-'+name)
        manifest = read(bound(path/'manifest.json'))
        cache = OfflineTeacherCache(path)
        retained = [d for d in manifest['documents'] if d['split']=='train' and d['doc_id'] not in excluded]
        stats = Counter(); selected_stats = Counter(); origins = Counter(); selected_origins = Counter()
        flags = []; reasoning_lengths = []; selected_reasoning_lengths = []
        for document in retained:
            doc = document['doc_id']; meta = metadata[source].get(doc,{})
            ids = np.asarray(cache.read_document(doc,tokens_only=True)['input_ids'])
            opens = np.flatnonzero(ids==248068); closes = np.flatnonzero(ids==248069)
            lengths = [int(closes[closes>i][0])-int(i)-1 for i in opens if np.any(closes>i)]
            meaningful = [x for x in lengths if x>2]
            features = dict(documents=1,tokens=len(ids),nonempty_thinking=int(bool(meaningful)),
                empty_thinking=int(bool(lengths) and not meaningful),
                capture_truncated=int(document['original_length']>document['length']),
                ends_im_end=int(len(ids)>0 and ids[-1]==248046))
            for field in ('verified','finished','thought_closed'):
                if field in meta:features[field+':'+str(meta[field])] = 1
            if source=='teacher_code':
                judge = teacher_judge.get(doc.removeprefix('tcode:'),{})
                features['judge_kept:'+str(judge.get('kept'))] = 1
                features['judge_valid:'+str(judge.get('valid'))] = 1
                problem = teacher_prompts.get(doc.removeprefix('tcode:'),{})
            elif source=='r8_code_short_w8':
                key = re.sub(r'^onpolicy:|:s\d+$|:greedy$','',doc)
                problem = r8_prompts.get(key,{})
            else:problem = {}
            origin = str(problem.get('source') or meta.get('source') or meta.get('domain') or doc.split(':')[0])
            origins[origin] += 1
            stats.update(features); reasoning_lengths += meaningful
            if doc in visits:
                selected_stats.update(features); selected_origins[origin] += 1
                selected_reasoning_lengths += meaningful
            if source=='teacher_code' and doc in visits:
                tests = problem.get('tests','')
                code_prompt_test_names[doc] = dict(functions=len(re.findall(r'\bdef\s+test_',tests)),
                    assertions=len(re.findall(r'\bassert\b',tests)),parametrize='parametrize' in tests,
                    edge_name_hint=bool(re.search(r'empty|bound|duplicate|negative|zero|single',tests,re.I)))
            if features['capture_truncated'] and source!='frontier_code_raw':
                flags.append(dict(doc_id=doc,reason='conversation_capture_truncated',selected=doc in visits))
            text = problem.get('question') or problem.get('prompt') or ''
            if text and doc in visits:
                hits = set().union(*(bank.get(g,set()) for g in grams(text)))
                hits |= exact.get(signature(text),set())
                if hits:overlaps.append(dict(source=name,doc_id=doc,matches=sorted(hits),method='exact full prompt or 13-word stem containment; flag for review, not verdict'))
            if tokenizer is not None:
                cached_text = tokenizer.decode(ids.tolist(),skip_special_tokens=False)
                normalized = signature(cached_text)
                cached_grams = grams(cached_text)
                possible = set().union(*(bank.get(g,set()) for g in cached_grams))
                # Include short MBPP stems below the shingle width, requiring
                # multiple words; short generic matches still need manual review.
                possible |= {tid for tid,stem in benchmark_text.items() if len(stem.split())>=8 and stem in normalized}
                for tid in sorted(possible):
                    shared = len(cached_grams & benchmark_grams[tid])
                    cache_overlaps.append(dict(source=name,doc_id=doc,benchmark=tid,
                        exact_full_stem=benchmark_text[tid] in normalized,shared_13word_grams=shared,
                        benchmark_13word_grams=len(benchmark_grams[tid]),selected=doc in visits))
        cache.close()
        def distribution(lengths):
            return dict(turns=len(lengths),median=float(np.median(lengths)) if lengths else None,
                        p90=float(np.percentile(lengths,90)) if lengths else None)
        report[name] = dict(cached_documents=len(manifest['documents']),retained_train=dict(stats),
            recent_prefix=dict(selected_stats),recent_visits=sum(visits[d['doc_id']] for d in retained),
            available_outside_recent_prefix=sum(d['doc_id'] not in visits for d in retained),
            origins=dict(origins),recent_origins=dict(selected_origins),
            reasoning_lengths=distribution(reasoning_lengths),recent_reasoning_lengths=distribution(selected_reasoning_lengths),
            flags=flags,metadata_missing=sum(d['doc_id'] not in metadata[source] for d in retained))
        print(name,json.dumps(dict(retained=stats['documents'],selected=selected_stats['documents'],
                                  thinking=selected_stats['nonempty_thinking'],truncated=stats['capture_truncated'])),flush=True)
    result = dict(sources=report,active_verified_code_test_structure=code_prompt_test_names,
        active_verified_prompt_overlap_flags=overlaps,retained_cache_code_overlap_flags=cache_overlaps,
        full_retained_token_screen=options.screen_code,source_sha256=hashes,
        caveats=['Test names and assertions do not prove coverage or correctness.',
                 'Outside recent prefix does not mean never trained on before.',
                 'Reasoning lengths count native token boundaries, not judge correctness.',
                 'Code stem overlap is only a flag; no exclusion list or training data changed.'])
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
