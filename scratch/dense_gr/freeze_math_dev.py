# Assisted-by: Codex
"""Conservative local-capture overlap screen; never claim foundation pretraining is clean."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def words(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def fragments(value, decoder):
    if isinstance(value,dict):
        for k,v in value.items():
            if k == 'input_ids' and isinstance(v,list) and v and isinstance(v[0],int):
                yield decoder.decode(v,skip_special_tokens=False)
            elif isinstance(v,(str,dict,list)):
                yield from fragments(v,decoder)
    elif isinstance(value,list):
        for v in value:
            if isinstance(v,(str,dict,list)):
                yield from fragments(v,decoder)
    elif isinstance(value,str):
        yield value


def main(output, candidates=None):
    if output.exists():
        raise ValueError('refuse to overwrite frozen development bank')
    bank = json.loads((candidates or HERE/'evaluation-audit/math-bank.json').read_text())
    categories=bank.pop('_categories',{})
    candidate_note=bank.pop('_note',None)
    rows, index, seen_questions = {}, defaultdict(set), set()
    short_sizes = defaultdict(set)
    for subject, samples in bank.items():
        for n,(q,a) in enumerate(samples):
            ident = f'{subject}:{n}'
            w = words(q)
            normalized=' '.join(w)
            if normalized in seen_questions:
                continue
            seen_questions.add(normalized)
            rows[ident] = dict(id=ident,subject=subject,question=q,answer=a,
                              question_sha256=hashlib.sha256(q.encode()).hexdigest(),
                              category=categories.get(hashlib.sha256(q.encode()).hexdigest(),subject))
            # Short questions use their entire normalized text.
            size = min(13,len(w))
            if size < 13:
                short_sizes[w[0]].add(size)
            for at in range(len(w)-size+1):
                index[' '.join(w[at:at+size])].add(ident)
    hit, files = {}, []
    decoder = Tokenizer.from_file(str(HERE/'merges-long1/u50/tokenizer.json'))
    for path in sorted((ROOT/'capture-data').glob('*.jsonl')):
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for line_no,line in enumerate(f,1):
                digest.update(line)
                row = json.loads(line)
                for fragment in fragments(row,decoder):
                    w = words(fragment)
                    for at,word in enumerate(w):
                        for size in (13,*short_sizes.get(word,())):
                            if at+size>len(w):
                                continue
                            for ident in index.get(' '.join(w[at:at+size]), ()):
                                if ident not in hit:
                                    hit[ident] = dict(file=path.name,line=line_no)
        files.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest()))
        print(f'{path.name}: {len(hit)} excluded questions',flush=True)
    rng = random.Random(20261007)
    selected, counts = [], {}
    for subject in bank:
        eligible = [r for k,r in rows.items() if r['subject']==subject and k not in hit]
        rng.shuffle(eligible)
        counts[subject] = len(eligible)
        strata=defaultdict(list)
        for r in eligible:
            strata[r['category']].append(r)
        chosen=[]
        while len(chosen)<64 and any(strata.values()):
            for category in sorted(strata):
                if strata[category] and len(chosen)<64:
                    chosen.append(strata[category].pop())
        selected += chosen
    result = dict(version=1,method='case/whitespace/punctuation-normalized 13-word overlap (whole question if shorter) against every JSONL field and decoded input_ids in capture-data',
                  candidate_sha256=hashlib.sha256((candidates or HERE/'evaluation-audit/math-bank.json').read_bytes()).hexdigest(),
                  candidate_note=candidate_note,
                  limitation='Local capture coverage only. No claim about foundation pretraining or unrecorded training data. Any shared shingle is conservatively excluded.',
                  source_files=files,excluded=hit,eligible_counts=counts,rows=selected,
                  ready=all(v>=64 for v in counts.values()))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Eligible:',counts,'ready:',result['ready'],flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'phase3-eval/math-dev.json')
    p.add_argument('--candidates',type=Path)
    a=p.parse_args()
    main(a.output,a.candidates)
