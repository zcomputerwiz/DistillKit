"""Bounded real-model/table/cache checks; no checkpoints or full training run.

Run from repo root. Default: donor sweep + three short optimizer steps.
--long: two optimizer steps at batch 2 x 4096 and blend .01, including live Adam state.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_DATASETS_CACHE'] = os.path.abspath('scratch/hyper-connection/datasets-cache')
import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml
from transformers import Qwen3_5ForCausalLM
from distillkit.configuration import DistillationRunConfig
from distillkit.hyper_connection import HyperConnection
from distillkit.independent_eval import make_collator, score_sequences
from distillkit.tp_model import shard_model
from distillkit.widened_residual import collapse_residual
from distillkit.trainer import HybridDistillationTrainer
import distillkit.main as entry

parser = argparse.ArgumentParser()
parser.add_argument('--long', action='store_true')
args = parser.parse_args()
OUT = Path('scratch/hyper-connection')
TAG = 'long' if args.long else 'short'
START = time.monotonic()
log = (OUT / f'{TAG}-measurements.jsonl').open('w', encoding='utf-8')

def emit(kind, **data):
    line = json.dumps(dict(kind=kind, elapsed=round(time.monotonic()-START, 3), **data))
    print('PROBE ' + line, flush=True)
    log.write(line + '\n'); log.flush()

def deadline():
    emit('timeout', seconds=540)
    os._exit(124)

watchdog = threading.Timer(540, deadline)
watchdog.daemon = True
watchdog.start()

def memory():
    return [dict(device=i, allocated_gib=torch.cuda.memory_allocated(i)/2**30,
                 peak_gib=torch.cuda.max_memory_allocated(i)/2**30,
                 reserved_gib=torch.cuda.memory_reserved(i)/2**30,
                 peak_reserved_gib=torch.cuda.max_memory_reserved(i)/2**30) for i in range(2)]

def set_blend(model, value):
    for module in model.modules():
        if isinstance(module, HyperConnection):
            module.set_blend(value)


class ProbeTrainer(HybridDistillationTrainer):
    def __init__(self, *a, **kw):
        ds = kw['train_dataset']
        lengths = ds['length'] if 'length' in ds.column_names else [len(x) for x in ds['input_ids']]
        candidates = [i for i, n in enumerate(lengths) if n == 4096] if args.long else [
            i for i, n in enumerate(lengths) if 128 <= n <= 384]
        selected = sorted(candidates, key=lambda i: (lengths[i], i))[:2 if args.long else 3]
        assert len(selected) == (2 if args.long else 3)
        kw['train_dataset'] = ds.select(selected)
        kw['eval_dataset'] = None
        emit('documents', indices=selected, lengths=[lengths[i] for i in selected],
             ids=[str(ds[i].get('doc_id')) for i in selected])
        super().__init__(*a, **kw)
        self.steps = 0
        self.profile = None
        self.hooks = []
        for i, layer in enumerate(self.model.model.layers):
            def capture(module, inputs, output, index=i):
                if self.profile is not None:
                    x = collapse_residual(output.detach()).float()
                    rms = x.square().mean().sqrt().item()
                    assert torch.isfinite(x).all(), f'nonfinite layer {index}'
                    self.profile[index] = rms
            self.hooks.append(layer.register_forward_hook(capture))
        if not args.long:
            self.preflight()
        for i in range(2):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(i)

    @torch.no_grad()
    def preflight(self):
        self.model.eval()
        bundle = json.loads(Path('scratch/independent-eval/reply-bundle-384.json').read_text())
        # Fixed smallest eight held-out documents, preserving all role spans/tokens.
        self.features = sorted(bundle['splits']['screen']['nll'], key=lambda f: (len(f['ids']), f['id']))[:8]
        self.collator = make_collator(bundle['pad_token_id'], self.data_collator.table)
        plain = make_collator(bundle['pad_token_id'])
        emit('screen', ids=[f['id'] for f in self.features], lengths=[len(f['ids']) for f in self.features])
        # Separate pre-retrofit student, same TP kernels. Its scores are the absolute control.
        stock = Qwen3_5ForCausalLM.from_pretrained('../student-hf', dtype=torch.bfloat16,
                                                attn_implementation='sdpa')
        shard_model(stock, ['cuda:0', 'cuda:1'])
        stock.eval()
        self.baseline = [score_sequences(stock, [f], plain, 'enabled', 'cuda:0')[0]['by_role']['assistant']
                         for f in self.features]
        first = {k: v.to('cuda:0') for k, v in plain(self.features[:1]).items()}
        expected = stock(**first, logits_to_keep=1, output_hidden_states=True)
        expected_logits = expected.logits.cpu()
        expected_hidden = [x.cpu() for x in expected.hidden_states]
        del expected, stock
        gc.collect(); torch.cuda.empty_cache()
        set_blend(self.model, 0)
        actual = self.model(**first, sidecar_enabled=False, logits_to_keep=1, output_hidden_states=True)
        assert torch.equal(expected_logits, actual.logits.cpu())
        assert all(torch.equal(x, y.cpu()) for x, y in zip(expected_hidden, actual.hidden_states))
        emit('real_identity', logits_bitwise=True, hidden_states_bitwise=True, states=len(expected_hidden))
        del actual, expected_logits, expected_hidden
        self.baseline_nll = sum(x['sum_nll'] for x in self.baseline) / sum(x['tokens'] for x in self.baseline)
        self.rms_baseline = None
        for alpha in (0, .01, .05, .1, .25, .5, 1):
            set_blend(self.model, alpha)
            rows = []
            profiles = []
            for f in self.features:
                self.profile = {}
                rows.append(score_sequences(self.model, [f], self.collator, 'bypassed', 'cuda:0')[0]['by_role']['assistant'])
                assert len(self.profile) == 32
                profiles.append([self.profile[i] for i in range(32)])
            layer_rms = torch.tensor(profiles).mean(0)
            if alpha == 0:
                self.rms_baseline = layer_rms
                assert rows == self.baseline
            ratio = layer_rms / self.rms_baseline
            nll = sum(x['sum_nll'] for x in rows) / sum(x['tokens'] for x in rows)
            # Predeclared safety screen, not a quality acceptance criterion.
            bounded = bool((ratio <= 4).all() and (layer_rms <= 5).all())
            emit('sweep', blend=alpha, assistant_nll=nll, baseline_nll=self.baseline_nll,
                 delta_nll=nll-self.baseline_nll, tokens=sum(x['tokens'] for x in rows),
                 layer_rms=layer_rms.tolist(), layer_ratio=ratio.tolist(), bounded=bounded,
                 rows=rows)
            assert bounded, f'blend {alpha} failed residual scale guard'
        self.profile = None
        set_blend(self.model, 0)
        self.model.train()

    def training_step(self, *a, **kw):
        self.profile = {}
        loss = super().training_step(*a, **kw)
        assert torch.isfinite(loss).all()
        assert len(self.profile) == 32
        assert max(self.profile.values()) < 5
        emit('training_forward_backward', step=self.steps+1, loss=float(loss),
             blend=float(self.model.model.layers[0].attn_residual.blend),
             layer_rms=[self.profile[i] for i in range(32)], memory=memory())
        self.profile = None
        return loss

    def create_optimizer(self):
        fresh = self.optimizer is None
        opt = super().create_optimizer()
        if not fresh:
            return opt
        watched = {'down': self.model.model.layers[24].attn_residual.W_down.weight,
                   'up': self.model.model.layers[24].attn_residual.W_up.weight,
                   'write': self.model.model.layers[24].attn_residual.W_write.weight,
                   'norm': self.model.model.layers[24].attn_residual.branch_gain_delta,
                   'value': self.model.model.layers[24].sidecar.ple.value_proj.weight}
        def before(opt, a, kw):
            self.before = {n: p.detach().clone() for n, p in watched.items()}
            emit('gradients', step=self.steps+1, gradients={n: None if p.grad is None else float(p.grad.float().norm()) for n,p in watched.items()})
        def after(opt, a, kw):
            self.steps += 1
            emit('optimizer_step', step=self.steps,
                 updates={n: dict(norm=float((p.detach().float()-self.before[n].float()).norm()),
                                  changed=int((p.detach()!=self.before[n]).sum())) for n,p in watched.items()},
                 memory=memory())
        opt.register_step_pre_hook(before)
        opt.register_step_post_hook(after)
        return opt

    def save_model(self, *a, **kw):
        emit('export_suppressed', steps=self.steps)
        if not args.long:
            self.model.eval()
            for mode in ('enabled', 'bypassed'):
                self.profile = {}
                rows = [score_sequences(self.model, [f], self.collator, mode, 'cuda:0')[0]['by_role']['assistant'] for f in self.features]
                nll = sum(x['sum_nll'] for x in rows) / sum(x['tokens'] for x in rows)
                emit('after_training_screen', mode=mode, assistant_nll=nll, baseline_nll=self.baseline_nll,
                     delta_nll=nll-self.baseline_nll, blend=float(self.model.model.layers[0].attn_residual.blend),
                     rows=rows, layer_rms=[self.profile[i] for i in range(32)])
            self.profile = None


entry.HybridDistillationTrainer = ProbeTrainer
raw = yaml.safe_load(Path('scratch/borrowed-L24-stage1-1m.yml').read_text())
raw['output_path'] = str(OUT / f'{TAG}-trainer-output')
raw['project_name'] = ''
raw['residual_stream'].update(routing='flash_next', blend=.01 if args.long else 0,
                              blend_target=.015, blend_warmup_steps=0 if args.long else 3)
raw['training_args'].update(max_steps=2 if args.long else 3,
    per_device_train_batch_size=2 if args.long else 1, gradient_accumulation_steps=1,
    eval_strategy='no', save_strategy='no', report_to='none', logging_steps=1,
    warmup_steps=0, lr_scheduler_type='constant', disable_tqdm=True)
raw['optimizer']['log_every_n_steps'] = 1
(OUT/f'{TAG}-config.yml').write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
emit('configuration', tag=TAG, torch_version=torch.__version__, routing=raw['residual_stream'],
     steps=raw['training_args']['max_steps'])
torch.manual_seed(42)
entry.do_distill(DistillationRunConfig.model_validate(raw))
emit('complete', memory=memory())
watchdog.cancel()
