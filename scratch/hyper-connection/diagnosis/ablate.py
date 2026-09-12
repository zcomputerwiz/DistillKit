"""Inference-only causal routing ablations. No production mutations or training."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
import json
import sys
import threading
import time
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoConfig
from distillkit.models import Qwen35WidenedForCausalLM
from distillkit.borrowed_routing import initialise_widened_residual
from distillkit.hyper_connection import HyperConnection
from distillkit.tp_model import shard_model
from distillkit.widened_residual import collapse_residual
from distillkit.independent_eval import make_collator, score_sequences

parser = argparse.ArgumentParser()
parser.add_argument('--confirm', action='store_true')
parser.add_argument('--modes', nargs='+')
parser.add_argument('--tag')
args = parser.parse_args()
OUT = Path('scratch/hyper-connection/diagnosis')
OUT.mkdir(parents=True, exist_ok=True)
tag = args.tag or ('confirm' if args.confirm else 'screen')
log = (OUT / f'{tag}.jsonl').open('w', encoding='utf-8')
start = time.monotonic()
def emit(kind, **kw):
    row = dict(kind=kind, elapsed=time.monotonic()-start, **kw)
    line = json.dumps(row)
    print(line, flush=True); log.write(line+'\n'); log.flush()
def timeout():
    emit('timeout'); os._exit(124)
watchdog = threading.Timer(900, timeout); watchdog.daemon=True; watchdog.start()
torch.set_num_threads(8)
torch.manual_seed(42)
cfg = AutoConfig.from_pretrained('../student-hf')
cfg.residual_stream_routing = 'flash_next'
cfg.residual_stream_num_branches = 4
cfg.residual_stream_lowrank = 320
cfg.residual_stream_blend = 1.
cfg.residual_stream_sidecar = False
model = Qwen35WidenedForCausalLM.from_pretrained('../student-hf', config=cfg,
    dtype=torch.bfloat16, attn_implementation='sdpa').eval()
emit('loaded', donor=initialise_widened_residual(model, '../flash-next-hc'))
shard_model(model, ['cuda:0', 'cuda:1'])
bundle = json.loads(Path('scratch/independent-eval/reply-bundle-384.json').read_text())
pool = bundle['splits']['screen']['nll']
short = sorted(pool,key=lambda f:(len(f['ids']),f['id']))[:8]
if args.confirm:
    excluded = {f['id'] for f in short}
    candidates = [f for f in pool if f['id'] not in excluded]
    indices = np.random.default_rng(20260911).choice(len(candidates),32,replace=False)
    features = [candidates[i] for i in indices]
else:
    features = short
collator = make_collator(bundle['pad_token_id'])
emit('documents', ids=[f['id'] for f in features], lengths=[len(f['ids']) for f in features])
context = dict(mode='baseline', doc=0)
clean_states = {}
diagnostic = []
profiles = []

def compare(a,b):
    a,b=a.float(),b.float()
    ar=a.square().mean(-1).sqrt(); br=b.square().mean(-1).sqrt()
    cos=F.cosine_similarity(a,b,dim=-1)
    scalar=(a*b).sum(-1)/(b.square().sum(-1).clamp_min(1e-20))
    return dict(rms_a=float(ar.mean()), rms_b=float(br.mean()),
                rms_ratio=float((br/ar.clamp_min(1e-20)).mean()),
                cosine=float(cos.mean()), relative_error=float(((a-b).norm(dim=-1)/a.norm(dim=-1).clamp_min(1e-20)).mean()),
                best_scalar_residual=float(((a-scalar[...,None]*b).norm(dim=-1)/a.norm(dim=-1).clamp_min(1e-20)).mean()))

@torch.no_grad()
def read(route, states, norm):
    mode = context['mode']
    original = norm(states[...,route.read_index,:])
    if mode.startswith('layer24') and route.layer_index != 24:
        return original.contiguous(), None
    if mode=='last8' and route.layer_index<24: return original.contiguous(),None
    if mode=='first8' and route.layer_index>=8: return original.contiguous(),None
    if mode=='attn_only' and route.kind!='attn': return original.contiguous(),None
    if mode=='mlp_only' and route.kind!='mlp': return original.contiguous(),None
    if mode=='half_read': return (.5*original).contiguous(),None
    if mode=='baseline' and (context['doc']!=0 or args.confirm):
        return original.contiguous(),None
    x=states.float()
    unit=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+route.norm_eps)
    z=(unit*(1+route.branch_gain_delta.float())).to(states.dtype)
    if mode in ('student_norm','student_norm_double_gate'):
        z=torch.stack([norm(branch) for branch in states.unbind(-2)],-2)
    flat=z.flatten(-2)
    g=torch.sigmoid(route.W_up(F.silu(route.W_down(flat)/route.num_branches))).reshape_as(z)
    donor=(g*z).mean(-2)
    weights=2*torch.sigmoid(route.W_write(flat)/route.num_branches)
    if mode in ('normalized_read','normalized_donor_write','normalized_unitmean_write'):
        student=torch.stack([norm(branch) for branch in states.unbind(-2)],-2)
        mixing=g.float()/g.float().sum(-2,keepdim=True).clamp_min(1e-20)
        ref=student[...,route.read_index,:]
        value=ref+(mixing*(student.float()-ref.float().unsqueeze(-2))).sum(-2).to(ref.dtype)
        if mode=='normalized_read':weights=None
        elif mode=='normalized_unitmean_write':weights=weights/weights.mean(-1,keepdim=True)
        return value.contiguous(),weights
    if context['doc']==0 and mode=='baseline':
        row=dict(layer=route.layer_index,sublayer=route.kind,read=compare(original,donor),
                 gate_mean=float(g.float().mean()),gate_std=float(g.float().std()),
                 write_mean=float(weights.float().mean()),write_std=float(weights.float().std()),
                 student_gain_rms=float((1+norm.weight.float()).square().mean().sqrt()),
                 donor_gain_rms=float((1+route.branch_gain_delta.float()).square().mean().sqrt()))
        if route.kind=='mlp' and route.layer_index in (0,8,16,24,31):
            block=model.model.layers[route.layer_index].mlp
            rms=lambda t:t.float().square().mean(-1,keepdim=True).sqrt()
            matched=(donor.float()*rms(original)/rms(donor).clamp_min(1e-20)).to(donor.dtype)
            y=block(original)
            row['mlp_output']=compare(y,block(donor))
            row['mlp_output_rms_corrected_read']=compare(y,block(matched))
        diagnostic.append(row)
        return original.contiguous(),None
    if mode=='write_only': return original.contiguous(),weights
    if mode=='norm_only': return z.mean(-2).contiguous(),None
    if mode=='norm_half': return (.5*z.mean(-2)).contiguous(),None
    if mode=='gate_only':
        student=torch.stack([norm(branch) for branch in states.unbind(-2)],-2)
        return (g*student).mean(-2).contiguous(),None
    if mode=='student_norm_double_gate': return (2*donor).contiguous(),None
    if mode=='double_read': return (2*donor).contiguous(),None
    if mode in ('rms_read','rms_full','amplitude_only'):
        ri=original.float().square().mean(-1,keepdim=True).sqrt()
        rd=donor.float().square().mean(-1,keepdim=True).sqrt()
        value=(original.float()*rd/ri.clamp_min(1e-20) if mode=='amplitude_only'
               else donor.float()*ri/rd.clamp_min(1e-20)).to(original.dtype)
        return value.contiguous(),weights if mode=='rms_full' else None
    return donor.contiguous(), None if mode in ('read_only','student_norm') else weights

for i, layer in enumerate(model.model.layers):
    for kind,route in [('attn',layer.attn_residual),('mlp',layer.mlp_residual)]:
        route.layer_index=i; route.kind=kind
        route.read=types.MethodType(read,route)
    def hook(module,inputs,output,index=i):
        if context['doc']!=0:return
        collapsed=collapse_residual(output.detach())
        if context['mode']=='baseline':clean_states[index]=collapsed.cpu()
        else:
            profiles.append(dict(layer=index,**compare(clean_states[index].to(collapsed.device),collapsed)))
    layer.register_forward_hook(hook)

modes=args.modes or ['baseline','donor','read_only','write_only','norm_only','norm_half','half_read',
    'gate_only','student_norm','student_norm_double_gate','double_read','rms_read','rms_full',
    'amplitude_only','layer24','first8','last8','attn_only','mlp_only']
assert modes[0]=='baseline'
with torch.inference_mode():
    for mode in modes:
        context['mode']=mode
        rows=[];profiles.clear()
        for i,f in enumerate(features):
            context['doc']=i
            result=score_sequences(model,[f],collator,'bypassed','cuda:0')[0]['by_role']['assistant']
            rows.append(dict(id=f['id'],**result))
        nll=sum(r['sum_nll'] for r in rows)/sum(r['tokens'] for r in rows)
        if mode=='baseline':
            baseline=nll
            (OUT/f'{tag}-local-arithmetic.json').write_text(json.dumps(diagnostic,indent=2))
        emit('arm',mode=mode,nll=nll,delta=nll-baseline,rows=rows,profile=list(profiles))
        torch.cuda.empty_cache()
emit('complete')
watchdog.cancel()
