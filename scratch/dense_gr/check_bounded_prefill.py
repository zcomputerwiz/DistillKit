# Assisted-by: Codex
"""Compare original and bounded cached prefill, then measure a long batch."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
import smoke_train
from transformers import AutoTokenizer
from distillkit.models import Qwen35WidenedForCausalLM
from distillkit.models.qwen35.csa2 import Qwen35SparseLatentAttention
from bounded_cached_prefill import install


def main(args):
    torch.cuda.set_per_process_memory_fraction(.90)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16).to('cuda:0').eval()
    model.config.use_cache = True
    layers = [m for m in model.modules() if isinstance(m, Qwen35SparseLatentAttention)]
    text = 'Inspect the available records and report the total. Archived record: units 17, status pending. '
    source = tok(text * 2000, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda:0')
    cases = [(str(n), source[:, :n], torch.ones_like(source[:, :n])) for n in (512,1024,2048)]
    bank = json.loads((Path(__file__).parent/'phase3-eval/agent-scenarios.json').read_text())['rows']
    pair = [next(r for r in bank if r['kind']==kind and r['padding_lines']==padding)
            for kind,padding in [('recover',0),('injection',128)]]
    tok.padding_side, tok.pad_token_id = 'left',248044
    rendered = [tok.apply_chat_template(r['messages'],tools=r['tools'],tokenize=False,
                add_generation_prompt=True,enable_thinking=False) for r in pair]
    padded = tok(rendered,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda:0')
    cases.append(('padded-agent-pair',padded.input_ids,padded.attention_mask))
    references = []
    with torch.inference_mode():
        for label,ids,mask in cases:
            output = model(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
            logits = output.logits.float().cpu()
            selected = [m.last_allowed.cpu().clone() for m in layers]
            del output
            tokens = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=16,
                                    do_sample=False, eos_token_id=[248044,248046], pad_token_id=248044)
            references.append((label,ids,mask, logits, selected, tokens.cpu()))
        install(model, args.query_chunk)
        results = []
        for label,ids,mask,logits,selected,tokens in references:
            output = model(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
            delta = (output.logits.float().cpu() - logits).abs()
            results.append(dict(label=label,length=ids.shape[1], max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                                top1_equal=bool((output.logits.cpu().argmax(-1)==logits.argmax(-1)).all()),
                                selected_sets_equal=[bool(torch.equal(m.last_allowed.cpu(),s))
                                                     for m,s in zip(layers,selected)]))
            del output
            following = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=16,
                                       do_sample=False, eos_token_id=[248044,248046], pad_token_id=248044)
            results[-1]['greedy_tokens_equal'] = bool(torch.equal(tokens, following.cpu()))
        del references, tokens, following
        for m in layers:
            m.last_allowed = None
        model.model.csa2_bus.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = source[:, :args.long_length].expand(2,-1).contiguous()
        mask = torch.ones_like(ids)
        start = time.perf_counter()
        output = model(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        memory = dict(length=ids.shape[1], batch=2, seconds=time.perf_counter()-start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    report = dict(query_chunk=args.query_chunk, parity=results, long_prefill=memory)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not all(r['top1_equal'] and all(r['selected_sets_equal']) and r['greedy_tokens_equal'] for r in results):
        raise RuntimeError('bounded prefill parity failed; inspect report before deployment')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--query-chunk',type=int,default=256)
    parser.add_argument('--long-length',type=int,default=16384)
    main(parser.parse_args())
