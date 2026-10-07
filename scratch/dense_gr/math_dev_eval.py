# Assisted-by: Codex
"""Evaluate a frozen screened bank, retaining case-level evidence and stable row RNGs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'downstream/code_bench'))
sys.path.insert(0,str(HERE.parent/'downstream/math_bench'))
sys.path.insert(0,str(HERE.parents[1]))
from generate import CompiledGreedy, stop_mask
from run_math import PROMPT, boxed, correct
import torch


def case_seed(ident, seed):
    return int.from_bytes(hashlib.sha256(f'{ident}:{seed}'.encode()).digest()[:8],'little') % (2**63-1)


def main(a):
    if a.output.exists():
        raise ValueError('refuse to overwrite evaluation')
    bank=json.loads(a.bank.read_text())
    if not bank['ready']:
        raise ValueError('screened bank is incomplete')
    from transformers import AutoTokenizer
    from distillkit.models import Qwen35WidenedForCausalLM
    tok=AutoTokenizer.from_pretrained(a.checkpoint)
    tok.padding_side='left'
    tok.pad_token_id=248044
    model=Qwen35WidenedForCausalLM.from_pretrained(a.checkpoint,dtype=torch.bfloat16).to('cuda:0').eval()
    model.config.use_cache=True
    a.output.mkdir(parents=True)
    manifest=dict(checkpoint=str(a.checkpoint.resolve()),bank_sha256=hashlib.sha256(a.bank.read_bytes()).hexdigest(),
                  eos=[248044,248046],seed=a.seed,per_case_rng=True,budget=a.budget,
                  batch_size=a.batch,protocol='gsm8k greedy nonthinking; math sampled thinking T=.6 p=.95 k=20',
                  status='running')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[]
    for subject in ('gsm8k','math'):
        rows=[r for r in bank['rows'] if r['subject']==subject]
        if a.limit_per_subject:
            rows=rows[:a.limit_per_subject]
        prompts=[tok.apply_chat_template([dict(role='user',content=PROMPT.format(problem=r['question']))],
                   tokenize=False,add_generation_prompt=True,enable_thinking=subject=='math') for r in rows]
        width=64*((max(len(tok.encode(p,add_special_tokens=False)) for p in prompts)+63)//64)
        runner=CompiledGreedy(model,a.batch,width,a.budget,[248044,248046],
                             sampling=None if subject=='gsm8k' else (.6,.95,20))
        for start in range(0,len(rows),a.batch):
            chunk=rows[start:start+a.batch]
            texts=prompts[start:start+a.batch]
            seeds=[case_seed(r['id'],a.seed) for r in chunk]
            filled=texts+[texts[0]]*(a.batch-len(texts))
            tokens=tok(filled,return_tensors='pt',padding='max_length',max_length=width,add_special_tokens=False).to('cuda:0')
            output=runner(tokens['input_ids'],tokens['attention_mask'],seeds=seeds+[seeds[0]]*(a.batch-len(seeds)))
            for offset,row in enumerate(chunk):
                new=output[offset,width:]
                stop=stop_mask(new,[248044,248046]).nonzero()
                end=int(stop[0]) if stop.numel() else len(new)
                ids=new[:end+(1 if stop.numel() else 0)].tolist()
                text=tok.decode(new[:end],skip_special_tokens=False)
                answer=boxed(text.split('</think>')[-1])
                record=dict(id=row['id'],subject=subject,prompt_sha256=hashlib.sha256(texts[offset].encode()).hexdigest(),
                            question_sha256=row['question_sha256'],seed=seeds[offset],token_ids=ids,raw=text,
                            truncated=not bool(stop.numel()),stop_token=int(new[end]) if stop.numel() else None,
                            answer=answer,reference=row['answer'],format_ok=answer is not None,
                            correct=bool(correct(answer,row['answer'])))
                records.append(record)
                with (a.output/'cases.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(record)+'\n')
            print(subject,min(start+a.batch,len(rows)),'/',len(rows),flush=True)
        del runner
        torch.cuda.empty_cache()
    manifest['status']='complete'
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summary={s:dict(count=sum(r['subject']==s for r in records),correct=sum(r['correct'] for r in records if r['subject']==s),
                    truncated=sum(r['truncated'] for r in records if r['subject']==s),unboxed=sum(not r['format_ok'] for r in records if r['subject']==s))
             for s in ('gsm8k','math')}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,default=HERE/'phase3-eval/math-dev-v3.json')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget',type=int,default=2048)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--limit-per-subject',type=int,default=0,
                   help='frozen bank prefix for a budget-sensitivity subset; 0 uses all')
    a=p.parse_args()
    if a.batch<1 or a.budget<1 or a.limit_per_subject<0:
        p.error('batch and budget must be positive')
    main(a)
