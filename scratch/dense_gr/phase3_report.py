# Assisted-by: Codex
"""CPU-only summaries and paired cases for the frozen phase-3 evaluations."""
import argparse
from collections import Counter
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'phase3-masking'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def agent(path):
    rows=read(path)['records']
    counts=Counter()
    families={}
    reviews=[]
    for row in rows:
        grade=row.get('grade') or {}
        counts['cases']+=1
        counts['success']+=bool(row['success'])
        counts['clean_success']+=bool(grade.get('clean_success',row['success'] and not row['errors']))
        counts['environment_outcome']+=bool(grade.get('outcome',False))
        counts['unauthorized_cases']+=bool('unauthorized_write' in row['errors'])
        counts['false_completion_cases']+=bool(grade.get('false_completion') or (
            row['kind'] in ('pagination','indirect_id','recover')
            and grade.get('report_verified') and not grade.get('outcome')))
        counts['invalid_attempts']+=len(row['errors'])
        counts['unfinished']+=row.get('grade') is None
        counts['announcement_flags']+=len(grade.get('unbacked_announcements',[]))
        counts['review_cases']+=bool(grade.get('review',[]))
        family=families.setdefault(row['kind'],dict(cases=0,success=0,clean_success=0))
        family['cases']+=1
        family['success']+=bool(row['success'])
        family['clean_success']+=bool(grade.get('clean_success',row['success'] and not row['errors']))
        if grade.get('review') or grade.get('unbacked_announcements') or row.get('grade') is None:
            reviews.append(dict(id=row['id'],grade=grade,errors=row['errors'],
                                final=row['messages'][-1],
                                assistant=[m for m in row['messages'] if m['role']=='assistant']))
    return dict(counts=counts,families=families,review=reviews)


def math_summary(path):
    manifest=read(path/'manifest.json')
    if manifest['status']!='complete':
        raise ValueError('incomplete math run: '+str(path))
    rows=[json.loads(line) for line in (path/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    return dict(summary=read(path/'summary.json'),cases=len(rows),
                budget_prefix=read(path/'budget-prefix.json')['summary'] if (path/'budget-prefix.json').exists() else None,
                stop_tokens=dict(Counter(str(r['stop_token']) for r in rows)),
                mean_generated_tokens=sum(len(r['token_ids']) for r in rows)/len(rows))


def paired_rows(base,candidate,key,metric):
    left={r[key]:r for r in base}
    right={r[key]:r for r in candidate}
    if len(left)!=len(base) or len(right)!=len(candidate) or left.keys()!=right.keys():
        raise ValueError('case inventory mismatch')
    gained,lost=[],[]
    for ident,row in left.items():
        other=right[ident]
        if 'prompt_sha256' in row and row['prompt_sha256']!=other['prompt_sha256']:
            raise ValueError('prompt mismatch: '+ident)
        if not row[metric] and other[metric]:
            gained.append(ident)
        if row[metric] and not other[metric]:
            lost.append(ident)
    return dict(cases=len(left),base=sum(bool(r[metric]) for r in left.values()),
                masked=sum(bool(r[metric]) for r in right.values()),gained=gained,lost=lost)


def main():
    result=dict(arms={},paired={},note='Development summaries; review flags are unresolved, not definitive capability failures.')
    for arm in ('base','masked'):
        root=OUT/'eval'/arm
        if not root.exists():
            continue
        result['arms'][arm]={}
        for path in sorted(root.glob('*.json')):
            result['arms'][arm][path.name]=agent(path)
        for path in sorted(root.glob('math-*')):
            if (path/'summary.json').exists():
                result['arms'][arm][path.name]=math_summary(path)
    base=OUT/'eval/base'
    candidate=OUT/'eval/masked'
    for path in sorted(base.glob('*.json')):
        other=candidate/path.name
        if other.exists():
            result['paired'][path.name]=paired_rows(read(path)['records'],read(other)['records'],'id','success')
    for path in sorted(base.glob('math-*')):
        other=candidate/path.name
        if (path/'summary.json').exists() and (other/'summary.json').exists():
            left=[json.loads(l) for l in (path/'cases.jsonl').read_text().splitlines()]
            right=[json.loads(l) for l in (other/'cases.jsonl').read_text().splitlines()]
            result['paired'][path.name]={s:paired_rows([r for r in left if r['subject']==s],
                [r for r in right if r['subject']==s],'id','correct') for s in ('gsm8k','math')}
    target=OUT/'comparison.json'
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Wrote',target,flush=True)
    for arm,by_test in result['arms'].items():
        for name,stats in by_test.items():
            print(arm,name,json.dumps(stats.get('counts',stats.get('summary'))),flush=True)


if __name__=='__main__':
    main()
