import ast, pathlib, types, torch, sys, json
sys.path.insert(0,'.')
from torch import nn
from torch.nn import functional as F
from distillkit.hyper_connection import HyperConnection
p=pathlib.Path('scratch/hyper-connection/modeling_qwen4_exp.py')
t=ast.parse(p.read_text(encoding='utf-8'))
nodes=[n for n in t.body if isinstance(n,ast.ClassDef) and n.name in ('Qwen4ExpTextRMSNorm','Qwen4ExpTextGatedResidual')]
ns=dict(torch=torch,nn=nn,F=F,Qwen4ExpTextConfig=types.SimpleNamespace)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
rows=[]
for dtype in (torch.float32,torch.bfloat16):
 torch.manual_seed(19)
 cfg=types.SimpleNamespace(hc_count=4,hidden_size=32,hc_lowrank=8,rms_norm_eps=1e-6)
 ref=ns['Qwen4ExpTextGatedResidual'](cfg).to(dtype)
 route=HyperConnection(32,4,8,blend=1).to(dtype)
 with torch.no_grad():
  ref.hc_norm.weight.uniform_(-.5,.5)
  for ours,theirs in [('W_down.weight','input_mix_weight_down.weight'),('W_up.weight','input_mix_weight_up.weight'),('W_write.weight','block_inject_weight.weight'),('branch_gain_delta','hc_norm.weight')]:
   route.get_parameter(ours).copy_(ref.get_parameter(theirs).reshape_as(route.get_parameter(ours)))
  x=torch.randn(2,19,4,32,dtype=dtype)
  a,_,w=ref(x.flatten(-2)); b,v=route.read(x,nn.Identity())
  torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6); assert torch.equal(w,v)
  rows.append(dict(dtype=str(dtype),read_max_error=float((a-b).abs().max()),read_bitwise=torch.equal(a,b),write_bitwise=True))
print(rows)
pathlib.Path('scratch/hyper-connection/source-oracle.json').write_text(json.dumps(rows,indent=2))
