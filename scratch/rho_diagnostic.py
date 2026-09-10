"""Analysis only: frozen checkpoint, whole-increment rho, no optimizer or training.

Run with .venv/Scripts/python.exe scratch/rho_diagnostic.py --docs 12.
Every loss partition uses the SAME predictor positions t whose target t+1 has
that role. Also report KD on its original unshifted all-position training mask.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRITON_CACHE_DIR'] = os.path.abspath('scratch/rho-diagnostic/triton-cache')
import argparse
import json
import time
import sys
from pathlib import Path
from types import SimpleNamespace, MethodType
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from distillkit.models.qwen35_widened import Qwen35WidenedForCausalLM
from distillkit.independent_eval import role_spans, make_collator, complete_bypass
from distillkit.signals import OfflineHiddenStateSignalSource
from distillkit.ngram_table import GGUFNGramTable
from distillkit.tp_model import shard_model
from distillkit.anchor_tap import AnchorTap
from distillkit.chunked_head import HeadContext
from distillkit.lossfuncs.kl import sparse_kl_div_inner
from distillkit.lossfuncs.hidden_state import compute_hs_loss


class ScaleDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, stream, output, rho):
        delta = output.float() - stream.float()
        ctx.save_for_backward(delta, rho)
        # Exact endpoint values; double interpolation avoids avoidable cancellation.
        return torch.lerp(stream.double(), output.double(), rho.double()).to(output.dtype)

    @staticmethod
    def backward(ctx, grad):
        delta, rho = ctx.saved_tensors
        return grad * (1-rho), grad * rho, (grad.float()*delta).sum().reshape_as(rho)


def ce_sum(logits, ids, values, mask):
    if hasattr(logits, 'sparse_logprobs'):
        nll = -logits.sparse_logprobs(ids).squeeze(-1)
    else:
        nll = -logits.float().log_softmax(-1).gather(-1, ids).squeeze(-1)
    return (nll * mask.squeeze(-1)).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', type=int, default=12)
    ap.add_argument('--dtype', choices=['bf16', 'fp32'], default='bf16')
    ap.add_argument('--steps', type=float, nargs='+', default=[0.01, 0.05])
    ap.add_argument('--output', default='scratch/rho-diagnostic/bf16.jsonl')
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--reference-recurrence', action='store_true')
    ap.add_argument('--checkpoint-above', type=int, default=700)
    args = ap.parse_args()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    log = out.open('w', encoding='utf-8'); started = time.monotonic()
    def emit(kind, **kw):
        row = dict(kind=kind, elapsed=time.monotonic()-started, **kw)
        s = json.dumps(row, default=lambda x: sorted(x) if isinstance(x,set) else str(x)); log.write(s+'\n'); log.flush(); print(s, flush=True)
    torch.set_num_threads(8)
    torch.manual_seed(20260910)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ckpt = Path('../runs/widened-plegated-stage1-1m/checkpoint-72')
    config = AutoConfig.from_pretrained(ckpt)
    config = getattr(config, 'text_config', config)
    model, info = Qwen35WidenedForCausalLM.from_pretrained(
        ckpt, config=config, dtype=torch.bfloat16, attn_implementation='sdpa',
        output_loading_info=True)
    saved = {}
    with safe_open(str(ckpt/'model.safetensors'), framework='pt') as f:
        for k in f.keys():
            if any(s in k for s in ('.sidecar.', '.attn_residual.', '.mlp_residual.', 'distillation_projections.')):
                saved[k] = f.get_tensor(k)
    old = 'model.layers.1.sidecar.ple.sharpness'
    new = old + '_delta'
    assert torch.equal(saved[old], torch.ones_like(saved[old]))
    assert set(info['missing_keys']) == {new}, info
    assert set(info['unexpected_keys']) == {old, 'distillation_projections.0.weight', 'distillation_projections.1.weight'}, info
    assert not info.get('mismatched_keys') and not info.get('error_msgs')
    model.model.layers[1].sidecar.ple.sharpness_delta.data.zero_()
    actual = model.state_dict()
    for k, v in saved.items():
        if k == old or k.startswith('distillation_projections.'): continue
        assert torch.equal(actual[k].cpu(), v.to(actual[k].dtype)), k
    shard_model(model, ['cuda:0', 'cuda:1'])
    if args.dtype == 'fp32':
        model.float()
    if args.reference_recurrence:
        # Optional pure torch audit of the recurrence (much larger autograd graph).
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        for name in ('torch_chunk_gated_delta_rule', 'causal_conv1d_fn'):
            op = getattr(qwen, name)
            while hasattr(op, '__wrapped__'): op = op.__wrapped__
            setattr(qwen, name, op)
    model.requires_grad_(False).eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        'use_reentrant':False, 'preserve_rng_state':False})
    assert getattr(model.config, 'attention_dropout', 0.) == 0.
    projections = []
    for i in range(2):
        w = saved[f'distillation_projections.{i}.weight']
        p = torch.nn.Linear(w.shape[1], w.shape[0], bias=False,
                            device='cuda:0', dtype=next(model.parameters()).dtype)
        p.weight.data.copy_(w); p.requires_grad_(False); projections.append(p)
    del actual, saved
    source = OfflineHiddenStateSignalSource('../teacher-cache-1m')
    tokenizer = AutoTokenizer.from_pretrained('../student-hf')
    raw_config = yaml.safe_load(Path('examples/qwen35_widened_plegated_stage1_1m.yml').read_text())
    table = GGUFNGramTable(raw_config['sidecar']['table_path'])
    collator = make_collator(tokenizer.pad_token_id, table)
    records = [r for r in source.cache.manifest['documents'] if r['split']=='eval']
    # Fixed seeded selection, independent of lengths, roles and measured losses.
    order = np.random.default_rng(20260910).permutation(len(records))
    records = [records[i] for i in order[args.offset:args.offset+args.docs]]
    emit('setup', pid=os.getpid(), args=vars(args), checkpoint=str(ckpt.resolve()), loading_info=info,
         migration='sharpness=1 -> sharpness_delta=0, all other adapter tensors exact',
         records=records, torch_version=torch.__version__,
         device_memory=[torch.cuda.memory_allocated(i) for i in range(2)])
    ple = model.model.layers[1].sidecar.ple
    original = ple.forward
    rho = torch.tensor(1., device='cuda:0', requires_grad=True)
    def scaled(self, stream, features):
        return ScaleDelta.apply(stream, original(stream, features), rho)
    def forward(batch, scaled_on=True, bypass=False):
        ple.forward = MethodType(scaled, ple) if scaled_on else original
        with complete_bypass(model, bypass), AnchorTap(model, [4, 32]) as tap:
            output = model(**batch, use_cache=False, logits_to_keep=1, return_dict=True)
        output.hidden_states = tap.states()
        return output
    for idx, record in enumerate(records):
        cached = source.cache.read_document(record['doc_id'], include_hidden_states=False)
        ids = cached['input_ids'].tolist()
        # Activate recomputation on long documents. All parameters remain frozen;
        # there are no optimizer steps, and this checkpoint's dropout is zero.
        model.train(len(ids) > args.checkpoint_above)
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        assert encoded['input_ids'] == ids, 'role retokenization mismatch; refuse misaligned masks'
        spans = role_spans(text, encoded['offset_mapping'])
        roles = torch.full((len(ids),), -1, dtype=torch.long)
        role_names = ['system', 'user', 'assistant', 'template']
        for name, ranges in spans.items():
            for a,b in ranges: roles[a:b] = role_names.index(name)
        assert (roles >= 0).all(), 'unassigned role tokens'
        batch = {k:v.to('cuda:0') for k,v in collator([{'ids':ids}]).items()}
        signal = source.get_signal(dict(batch, doc_id=[record['doc_id']]), return_hidden_states=True)
        targets = torch.tensor(ids[1:]+[0], device='cuda:0').reshape(1,-1,1)
        masks = {'matched_all': torch.arange(len(ids),device='cuda:0').lt(len(ids)-1).reshape(1,-1,1),
                 'training_all': torch.ones(1,len(ids),1,device='cuda:0',dtype=torch.bool)}
        for r,name in enumerate(role_names):
            masks[name] = torch.cat((roles[1:].eq(r), torch.tensor([False]))).to('cuda:0').reshape(1,-1,1)
        masks = {k:v for k,v in masks.items() if v.any()}
        if idx == 0:
            with torch.no_grad():
                a = forward(batch, scaled_on=False)
                rho.fill_(1); b = forward(batch)
                enabled = {str(k):float((a.hidden_states[k]-b.hidden_states[k]).abs().max()) for k in (4,32)}
                enabled['logits'] = float((a.logits-b.logits).abs().max())
                rho.fill_(0); a = forward(batch); b = forward(batch, bypass=True)
                bypass = {str(k):float((a.hidden_states[k]-b.hidden_states[k]).abs().max()) for k in (4,32)}
                bypass['logits'] = float((a.logits-b.logits).abs().max())
                emit('endpoints', enabled_max_abs=enabled, bypass_max_abs=bypass)
                assert max(enabled.values()) == 0 and max(bypass.values()) == 0
                del a,b
        def terms(output):
            context = HeadContext(output.hidden_states[32], model.get_output_embeddings(),
                                  signal.vocab_size, 64)
            values = {}
            for name, mask in masks.items():
                denominator = mask.sum()
                if name != 'training_all':
                    values[name+'/ce'] = context.accumulate(ce_sum, targets, targets, mask)/denominator
                values[name+'/kl'] = context.accumulate(sparse_kl_div_inner,
                    signal.sparse_ids, signal.sparse_values, mask)/denominator
                for i,layer in enumerate((4,32)):
                    mapping = SimpleNamespace(layer_mapping=[(layer,i)], projections=[projections[i]])
                    values[name+f'/cosine_{layer}'] = compute_hs_loss('cosine',output,signal,mask,mapping)
            return values
        raw = batch['ngram_raw']
        generator = torch.Generator().manual_seed(20260910+int(order[args.offset+idx]))
        permutation = torch.randperm(len(ids), generator=generator).to(raw.device)
        for control in ('real','shuffled'):
            emit('begin_control', index=idx, doc_id=record['doc_id'], control=control,
                 checkpoint_recomputation=model.training)
            batch['ngram_raw'] = raw if control=='real' else raw[:,permutation]
            with torch.no_grad(): rho.fill_(1)
            # Backpropagate each scalar separately, without accumulating parameter grads.
            output = forward(batch)
            losses = terms(output)
            analytic = {}
            # Cold FLA autotuners share state between TP worker threads. The
            # trainer serializes its first backward for the same reason.
            with torch.autograd.set_multithreading_enabled(False):
                for name, loss in losses.items():
                    analytic[name] = float(torch.autograd.grad(loss, rho, retain_graph=True)[0])
            base = {k:float(v.detach()) for k,v in losses.items()}
            del losses,output
            finite = {}
            with torch.no_grad():
                for step in args.steps:
                    rho.fill_(1-step); output = forward(batch)
                    minus = {k:float(v) for k,v in terms(output).items()}; del output
                    rho.fill_(1+step); output = forward(batch)
                    plus = {k:float(v) for k,v in terms(output).items()}; del output
                    finite[str(step)] = {'minus':minus,'plus':plus,
                        'derivative':{k:(plus[k]-minus[k])/(2*step) for k in plus}}
            emit('document', index=idx, doc_id=record['doc_id'], length=len(ids), control=control,
                 counts={k:int(v.sum()) for k,v in masks.items()}, role_roundtrip_exact=True,
                 permutation_fixed_fraction=float((permutation==torch.arange(len(ids),device=raw.device)).float().mean()),
                 losses=base, analytic=analytic, finite=finite,
                 peak_memory=[torch.cuda.max_memory_allocated(i) for i in range(2)])
        del signal,batch
    emit('complete')


if __name__ == '__main__': main()
