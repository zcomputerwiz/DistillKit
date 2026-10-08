# Assisted-by: Codex
"""CPU review of saved paired retention and budget diagnostics; never promotes."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import argparse
import time
from pathlib import Path
import numpy as np
from atlas import ledger_rows, paired_delta

OUT = Path(__file__).resolve().parent/'phase3-masking'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def retention():
    result = {}
    for bank in ('retention','atlas'):
        root = OUT/'eval'/bank
        metadata = read(root/'token_evidence.json')
        pooled = read(root/'nll.json')
        with np.load(root/'token_evidence.npz') as arrays:
            domains = {}
            for domain, count in metadata['documents'].items():
                labels = [arrays[f'documents/{domain}/{i}/roles'] for i in range(count)]
                ledgers = {}
                for arm in ('base','masked'):
                    observations = [(arrays[f'observations/{arm}/{domain}/{i}/nll'],
                                     arrays[f'observations/{arm}/{domain}/{i}/hit']) for i in range(count)]
                    ledgers[arm] = ledger_rows(observations,labels)
                groups = {}
                for group in ('own-turns','tool-call','thinking','plain'):
                    if group not in ledgers['base']:
                        continue
                    left,right = ledgers['base'][group],ledgers['masked'][group]
                    if not np.array_equal(left[:,2],right[:,2]):
                        raise ValueError('paired token inventory mismatch')
                    delta,low,high = paired_delta(left,right)
                    expected = pooled['masked'][domain][group]['nll']-pooled['base'][domain][group]['nll']
                    if abs(delta-expected)>1e-6:
                        raise ValueError('saved evidence does not match pooled NLL')
                    groups[group] = dict(targets=int(left[:,2].sum()),delta=delta,
                                         ci95_document_bootstrap=[low,high])
                domains[domain] = dict(documents=count,groups=groups)
        result[bank] = dict(domains=domains,evidence_sha256=hashlib.sha256(
            (root/'token_evidence.npz').read_bytes()).hexdigest())
    return result


def budgets():
    from transformers import AutoTokenizer
    from math_dev_eval import boxed, correct
    tokenizer = AutoTokenizer.from_pretrained(OUT.parent/'merges-long1/u50')
    result = {}
    for arm in ('base','masked'):
        root = OUT/'diagnostics'/f'{arm}-math-full-s1'
        if not (root/'summary.json').exists():
            result[arm] = dict(status='pending')
            continue
        manifest = read(root/'manifest.json')
        if manifest['status']!='complete' or manifest['budget']!=4096 or manifest['batch_size']!=8 or manifest['seed']!=1:
            raise ValueError('diagnostic runtime contract mismatch')
        short_root = OUT/'eval'/arm/'math-s1'
        short = {r['id']:r for r in map(json.loads,(short_root/'cases.jsonl').read_text().splitlines())}
        long = list(map(json.loads,(root/'cases.jsonl').read_text().splitlines()))
        if len(long)!=len(short) or {r['id'] for r in long}!=short.keys():
            raise ValueError('budget case inventory mismatch')
        rows=[]
        for row in long:
            before=short[row['id']]
            if row['prompt_sha256']!=before['prompt_sha256'] or row['seed']!=before['seed']:
                raise ValueError('budget prompt or RNG mismatch')
            ids=row['token_ids'][:2048]
            stopped=any(t in (248044,248046) for t in ids)
            text=tokenizer.decode([t for t in ids if t not in (248044,248046)],skip_special_tokens=False)
            answer=boxed(text.split('</think>')[-1])
            rows.append(dict(id=row['id'],subject=row['subject'],short_correct=before['correct'],
                             prefix_correct=bool(correct(answer,row['reference'])),full_correct=row['correct'],
                             prefix_stopped=stopped,full_truncated=row['truncated'],
                             short_prefix_identical=before['token_ids']==row['token_ids'][:len(before['token_ids'])]))
        summary={s:dict(cases=sum(r['subject']==s for r in rows),
            **{k:sum(bool(r[k]) for r in rows if r['subject']==s) for k in
               ('short_correct','prefix_correct','full_correct','prefix_stopped','full_truncated','short_prefix_identical')})
                 for s in ('gsm8k','math')}
        result[arm]=dict(status='complete',summary=summary,rows=rows,
            note='First 2048 tokens of the same 4096-token trajectory isolate the cap. Cross-run differences may also reflect static-cache shape arithmetic.')
    return result


def repeatability():
    root=OUT/'diagnostics/base-math-full-s1-repeat'
    if not (root/'summary.json').exists():
        return dict(status='pending')
    original=OUT/'diagnostics/base-math-full-s1'
    first=read(original/'manifest.json')
    second=read(root/'manifest.json')
    for key in ('checkpoint','bank_sha256','eos','seed','budget','batch_size','protocol','status'):
        if first[key]!=second[key]:
            raise ValueError('repeatability runtime mismatch: '+key)
    left={r['id']:r for r in map(json.loads,(original/'cases.jsonl').read_text().splitlines())}
    right={r['id']:r for r in map(json.loads,(root/'cases.jsonl').read_text().splitlines())}
    if left.keys()!=right.keys():
        raise ValueError('repeatability case inventory mismatch')
    summaries={}
    for subject in ('gsm8k','math'):
        rows=[r for r in left.values() if r['subject']==subject]
        for row in rows:
            if any(row[k]!=right[row['id']][k] for k in ('prompt_sha256','seed','reference')):
                raise ValueError('repeatability prompt/RNG mismatch')
        summaries[subject]=dict(cases=len(rows),
            exact_tokens=sum(r['token_ids']==right[r['id']]['token_ids'] for r in rows),
            first_correct=sum(r['correct'] for r in rows),
            repeat_correct=sum(right[r['id']]['correct'] for r in rows),
            gained=[r['id'] for r in rows if not r['correct'] and right[r['id']]['correct']],
            lost=[r['id'] for r in rows if r['correct'] and not right[r['id']]['correct']])
    return dict(status='complete',summary=summaries)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-repeat',action='store_true')
    args=parser.parse_args()
    status=OUT/'diagnostics/status.json'
    if args.wait_for_repeat:
        status.write_text(json.dumps(dict(stage='repeatability_running',promotion='not performed')))
        deadline=time.monotonic()+3600
        while True:
            path=OUT/'runs/diagnostic-base-math-full-s1-repeat.json'
            receipt=read(path) if path.exists() else {}
            if receipt.get('status')=='complete':
                break
            if receipt.get('status')=='failed' or time.monotonic()>deadline:
                status.write_text(json.dumps(dict(stage='repeatability_failed_or_timed_out',promotion='not performed')))
                raise RuntimeError('repeatability run failed or exceeded one-hour bound')
            time.sleep(10)
    result=dict(retention=retention(),budget=budgets(),repeatability=repeatability(),promotion='not performed',
                interval_note='2000 paired document bootstrap draws, seed 0; small fixed development banks, not powered noninferiority.')
    (OUT/'diagnostics').mkdir(exist_ok=True)
    (OUT/'diagnostics/review.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({arm:stats.get('summary',stats) for arm,stats in result['budget'].items()},indent=2))
    print(json.dumps(result['repeatability'],indent=2))
    if args.wait_for_repeat:
        status.write_text(json.dumps(dict(stage='repeatability_analysis_complete',promotion='not performed',
            next='Review diagnostics/review.json before freezing another training recipe.'),indent=2))
