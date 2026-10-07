# Assisted-by: Codex
"""Freeze all eligible eval documents, avoiding lexicographic source sampling."""
import hashlib
import json
from pathlib import Path
import torch
torch.cuda.is_available=lambda:False
torch.cuda.device_count=lambda:0
from transformers import AutoTokenizer
from teacher_kl import CachedTeacher
from smoke_train import ANSWER_MARKER,EFFORT_PROMPT
from atlas import roles_of,ROLES

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'phase3-eval/retention'
    if out.exists():
        raise ValueError('refuse to overwrite retention bank')
    out.mkdir(parents=True)
    tok=AutoTokenizer.from_pretrained(HERE/'merges-long1/u50')
    enc=lambda s:tok.encode(s,add_special_tokens=False)
    header=enc('<|im_start|>system\n')
    whole=enc('<|im_start|>system\n'+EFFORT_PROMPT+'<|im_end|>\n')
    heading=header+enc(EFFORT_PROMPT)+enc('\n\n')
    domains,roles,manifest={},{},{}
    for name,source in [('code','expand-code'),('thinking','thinking')]:
        cache=HERE.parents[2]/('teacher-cache-'+source)
        teacher=CachedTeacher(cache,'eval',device='cpu',max_length=4096,
            answer_marker=enc(ANSWER_MARKER),min_answer_tokens=2,
            strip_prefix=[(whole,0,len(whole)),(heading,len(header),len(heading))],
            strip_nonthinking=(enc(ANSWER_MARKER),enc('\n<think>\n\n</think>')))
        documents=[]
        domains[name],roles[name]=[],[]
        for doc in sorted(set(teacher.ids)):
            ids=teacher.read(doc)['input_ids'][0].int()
            labels=torch.from_numpy(roles_of(ids.numpy()))
            domains[name].append(ids)
            roles[name].append(labels)
            documents.append(dict(id=doc,length=len(ids),at_cap=len(ids)==4096,
                token_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                roles=dict(zip(ROLES,torch.bincount(labels.long()[1:],minlength=len(ROLES)).tolist()))))
        manifest[name]=dict(cache=str(cache),documents=documents,
             cache_manifest_sha256=hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest())
        teacher.close()
    path=out/'domains.pt'
    torch.save(dict(domains=domains,roles=roles,length=4096),path)
    (out/'manifest.json').write_text(json.dumps(dict(version=1,selection='all eligible eval documents; no first-N source bias',
        limitation='held-out capture split, not guaranteed absent from foundation pretraining; NLL is diagnostic, not execution correctness',
        domains=manifest,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
    print({k:len(v) for k,v in domains.items()},flush=True)


if __name__=='__main__':
    main()
