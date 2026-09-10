"""Bounded three-step instrumentation of the production trainer; no model export.

Run from repo root with .venv/Scripts/python.exe scratch/gate_update_diagnosis.py.
All mutations are process-local or under scratch/gate-update-diagnosis/.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_DATASETS_CACHE'] = os.path.abspath('scratch/gate-update-diagnosis/datasets-cache')
import sys
import json
import time
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

START = time.monotonic()
OUT = Path('scratch/gate-update-diagnosis')
OUT.mkdir(parents=True, exist_ok=True)
LOG = (OUT / 'measurements.jsonl').open('w', encoding='utf-8')
def emit(kind, **data):
    row = dict(kind=kind, elapsed=round(time.monotonic()-START, 3), **data)
    line = json.dumps(row)
    print('PROBE ' + line, flush=True)
    LOG.write(line+'\n'); LOG.flush()

def deadline():
    emit('timeout', limit_seconds=540)
    os._exit(124)
watchdog = threading.Timer(540, deadline)
watchdog.daemon = True
watchdog.start()

import torch
import yaml
from safetensors import safe_open
from distillkit.configuration import DistillationRunConfig
import distillkit.main as entry
import distillkit.tp_model as tp
from distillkit.trainer import HybridDistillationTrainer
from distillkit.ple_gated_sidecar import DirectionGatedPLESidecar

def stats(t):
    if t is None: return None
    x = t.detach().float()
    d = dict(dtype=str(t.dtype), norm=x.norm().item(), max_abs=x.abs().max().item(),
             nonzero=int(torch.count_nonzero(x)), count=x.numel(), finite=bool(torch.isfinite(x).all()))
    if x.numel() <= 4: d['values'] = x.cpu().tolist()
    return d

CKPT = Path('../runs/widened-plegated-stage1-1m/checkpoint-72')
checkpoint = {}
with safe_open(str(CKPT/'model.safetensors'), framework='pt', device='cpu') as f:
    for name in f.keys():
        if '.sidecar.' in name and any(name.endswith(n) for n in ('gate', 'sharpness', 'value_proj.weight', 'conv1d.weight')):
            checkpoint[name] = f.get_tensor(name)
            emit('checkpoint_parameter', name=name, **stats(checkpoint[name]))
saved = torch.load(CKPT/'optimizer.pt', map_location='cpu', weights_only=True)
for group in saved['param_groups']:
    for name, pid in zip(group.get('param_names', []), group['params']):
        if name in checkpoint:
            state = saved['state'].get(pid, {})
            emit('checkpoint_optimizer', name=name, lr=group['lr'], weight_decay=group['weight_decay'],
                 optimizer_kind=group['optimizer_kind'], state={k: stats(v) if torch.is_tensor(v) else v for k,v in state.items()})
del saved

class ProbeTrainer(HybridDistillationTrainer):
    def __init__(self, *args, **kwargs):
        ds = kwargs['train_dataset']
        # Complete, short training documents; retain cache alignment/truncation rules.
        lengths = ds['length'] if 'length' in ds.column_names else [len(x) for x in ds['input_ids']]
        candidates = [i for i,n in enumerate(lengths) if 128 <= n <= 384]
        selected = sorted(candidates, key=lambda i:(lengths[i],i))[:3]
        assert len(selected) == 3
        kwargs['train_dataset'] = ds.select(selected)
        kwargs['eval_dataset'] = None
        emit('documents', indices=selected, lengths=[lengths[i] for i in selected],
             doc_ids=[str(ds[i].get('doc_id')) for i in selected])
        super().__init__(*args, **kwargs)
        self.watched = {n:p for n,p in self.model.named_parameters() if n in checkpoint}
        assert len(self.watched) == 4
        self.initial = {n:p.detach().float().cpu().clone() for n,p in self.watched.items()}
        self.shadow_params = {n:torch.nn.Parameter(p.detach().float().clone())
                              for n,p in self.watched.items() if n.endswith(('gate','sharpness'))}
        self.shadow = None
        for n,p in self.watched.items():
            emit('initial_parameter', name=n, requires_grad=p.requires_grad, device=str(p.device),
                 checkpoint_delta=stats(checkpoint[n].float()-self.initial[n]), **stats(p))
        self.probe_step = 0
        original_sync = tp.sync_replicated_gradients
        def sync(model):
            before = {n:p.grad.detach().clone() if p.grad is not None else None for n,p in self.watched.items()}
            for n,p in self.watched.items(): emit('grad_before_sync', step=self.probe_step+1, name=n, grad=stats(p.grad))
            result = original_sync(model)
            for n,p in self.watched.items():
                emit('grad_after_sync', step=self.probe_step+1, name=n, grad=stats(p.grad),
                     identical=before[n] is None if p.grad is None else bool(torch.equal(before[n],p.grad)))
            return result
        tp.sync_replicated_gradients = sync

    def create_optimizer(self):
        fresh = self.optimizer is None
        result = super().create_optimizer()
        if not fresh: return result
        for group in result.param_groups:
            for p in group['params']:
                for n,w in self.watched.items():
                    if p is w:
                        emit('live_group', name=n, optimizer_kind=group['optimizer_kind'], lr=group['lr'], weight_decay=group['weight_decay'])
        self.shadow = torch.optim.AdamW(list(self.shadow_params.values()), lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay, betas=(self.args.adam_beta1,self.args.adam_beta2),
            eps=self.args.adam_epsilon, foreach=False)
        def before_step(opt, args, kwargs):
            self.before = {n:p.detach().clone() for n,p in self.watched.items()}
            for n,p in self.watched.items():
                group = next(g for g in opt.param_groups if any(w is p for w in g['params']))
                emit('grad_pre_optimizer', step=self.probe_step+1, name=n, grad=stats(p.grad), lr=group['lr'])
                if n in self.shadow_params:
                    self.shadow_params[n].grad = None if p.grad is None else p.grad.detach().float().clone()
                    self.shadow.param_groups[0]['lr'] = group['lr']
            self.shadow_before = {n:p.detach().clone() for n,p in self.shadow_params.items()}
            self.shadow.step()
        def after_step(opt, args, kwargs):
            self.probe_step += 1
            for n,p in self.watched.items():
                delta = p.detach().float()-self.before[n].float()
                state = opt.state[p]
                emit('update', step=self.probe_step, name=n, delta=stats(delta), parameter=stats(p),
                     total_delta=stats(p.detach().float().cpu()-self.initial[n]),
                     moment=stats(state.get('exp_avg')), variance=stats(state.get('exp_avg_sq')),
                     adam_step=float(state['step']) if 'step' in state else None,
                     fp32_same_grad_delta=stats(self.shadow_params[n].detach()-self.shadow_before[n]) if n in self.shadow_params else None,
                     fp32_parameter=stats(self.shadow_params[n]) if n in self.shadow_params else None)
        result.register_step_pre_hook(before_step)
        result.register_step_post_hook(after_step)
        return result

    def save_model(self, *args, **kwargs):
        emit('export_suppressed')

entry.HybridDistillationTrainer = ProbeTrainer
raw = yaml.safe_load(Path('examples/qwen35_widened_plegated_stage1_1m.yml').read_text())
raw['output_path'] = str(OUT/'trainer-output')
raw['project_name'] = ''
raw['training_args'].update(max_steps=3, per_device_train_batch_size=1, gradient_accumulation_steps=1,
    eval_strategy='no', save_strategy='no', report_to='none', logging_steps=1,
    warmup_steps=0, lr_scheduler_type='constant', disable_tqdm=True)
raw['optimizer']['log_every_n_steps'] = 1
emit('configuration', torch_version=torch.__version__, modifications={
    'max_steps':3, 'batch':1, 'accumulation':1, 'warmup_steps':0, 'lr':1e-4,
    'documents':'three complete short documents from training cache', 'checkpointing':True,
    'tensor_parallel':True, 'loss':'unchanged 0.7 sparse KL + 0.3 two-anchor cosine'})
torch.manual_seed(42)
entry.do_distill(DistillationRunConfig.model_validate(raw))
emit('complete')
watchdog.cancel()
