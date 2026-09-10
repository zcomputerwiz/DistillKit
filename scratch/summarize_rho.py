"""Token-weighted rho summaries and paired document bootstrap, from raw JSONL."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    docs = [r for r in rows if r['kind']=='document']
    groups = {c:{r['doc_id']:r for r in docs if r['control']==c} for c in ('real','shuffled')}
    ids = sorted(set(groups['real']) & set(groups['shuffled']))
    assert ids
    rng = np.random.default_rng(20260910)
    draw = rng.integers(len(ids),size=(20000,len(ids)))
    keys = sorted(set().union(*(groups['real'][i]['analytic'].keys() for i in ids)))
    result = {'source':str(path),'documents':len(ids),'doc_ids':ids,
              'length':sum(groups['real'][i]['length'] for i in ids),
              'endpoints':[r for r in rows if r['kind']=='endpoints'],
              'complete':rows[-1]['kind']=='complete','by_term':{}}
    for key in keys:
        role,term = key.split('/')
        weights = np.array([groups['real'][i]['counts'].get(role,0) for i in ids])
        total = weights.sum()
        if total == 0: continue
        entry = {'positions':int(total)}
        arrays = {}
        for control in groups:
            data = [groups[control][i] for i in ids]
            a = np.array([r['analytic'].get(key,0) for r in data]); arrays[control] = a
            bootstrap = (a[draw]*weights[draw]).sum(1)/np.maximum(weights[draw].sum(1),1)
            entry[control] = {'analytic':float(a@weights/total),
                'ci95':np.quantile(bootstrap,[.025,.975]).tolist(),
                'positive_documents':int(((a>0)&(weights>0)).sum()),
                'loss':float(sum(r['losses'].get(key,0)*w for r,w in zip(data,weights))/total),
                'finite':{}}
            for step in data[0]['finite']:
                fd = np.array([r['finite'][step]['derivative'].get(key,0) for r in data])
                entry[control]['finite'][step] = {'derivative':float(fd@weights/total),
                    'absolute_difference':float(abs((fd-a)@weights/total)),
                    'document_weighted_absolute_difference':float(abs(fd-a)@weights/total)}
        d = arrays['real']-arrays['shuffled']
        boot = (d[draw]*weights[draw]).sum(1)/np.maximum(weights[draw].sum(1),1)
        entry['real_minus_shuffled'] = {'analytic':float(d@weights/total),
            'ci95':np.quantile(boot,[.025,.975]).tolist()}
        result['by_term'][key]=entry
    # Trainer's cosine term averages its two anchors before the weight 0.3.
    result['weighted_kd'] = {}
    roles = sorted({k.split('/')[0] for k in keys})
    for role in roles:
        entry = {}
        for control in groups:
            weights=np.array([groups[control][i]['counts'].get(role,0) for i in ids])
            a=np.array([sum(w*groups[control][i]['analytic'].get(role+'/'+t,0)
                           for w,t in ((.7,'kl'),(.15,'cosine_4'),(.15,'cosine_32'))) for i in ids])
            boot=(a[draw]*weights[draw]).sum(1)/np.maximum(weights[draw].sum(1),1)
            entry[control]={'analytic':float(a@weights/weights.sum()),
                            'ci95':np.quantile(boot,[.025,.975]).tolist()}
        result['weighted_kd'][role]=entry
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('inputs',nargs='+'); ap.add_argument('--output',required=True)
    args=ap.parse_args()
    results=[summarize(p) for p in args.inputs]
    Path(args.output).write_text(json.dumps(results,indent=2),encoding='utf-8')
    lines=[]
    for result in results:
        lines += [f"**{result['source']} — {result['documents']} documents, {result['length']} tokens**",'',
                  '| Role / term | N | Real analytic | Real FD (step) | Shuffled analytic | Shuffled FD (step) |',
                  '|---|---:|---:|---|---:|---|']
        for key,e in result['by_term'].items():
            def fd(c): return '; '.join(f"{v['derivative']:+.8f} ({s})" for s,v in e[c]['finite'].items())
            lines.append(f"| {key} | {e['positions']} | {e['real']['analytic']:+.8f} | {fd('real')} | {e['shuffled']['analytic']:+.8f} | {fd('shuffled')} |")
        lines += ['', 'The 95% intervals below resample whole documents (20,000 paired draws); losses and derivatives are token-weighted.', '',
                  '| Role / term | Real 95% CI | Shuffled 95% CI | Real minus shuffled [95% CI] |','|---|---|---|---|']
        for key,e in result['by_term'].items():
            def ci(c):return '['+', '.join(f'{v:+.8f}' for v in e[c]['ci95'])+']'
            lines.append(f"| {key} | {ci('real')} | {ci('shuffled')} | {e['real_minus_shuffled']['analytic']:+.8f} {ci('real_minus_shuffled')} |")
        lines += ['', '| Weighted KD role | Real analytic [95% CI] | Shuffled analytic [95% CI] |','|---|---|---|']
        for role,e in result['weighted_kd'].items():
            lines.append('| '+role+' | '+' | '.join(f"{e[c]['analytic']:+.8f} [{e[c]['ci95'][0]:+.8f}, {e[c]['ci95'][1]:+.8f}]" for c in ('real','shuffled'))+' |')
        lines += ['']
    Path(args.output).with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    for r in results:
        print(r['source'],r['documents'], 'documents')
        for key in ('assistant/ce','assistant/kl','assistant/cosine_4','assistant/cosine_32'):
            print(key,r['by_term'][key])


if __name__=='__main__':main()
