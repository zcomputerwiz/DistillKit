"""Arms A, S and M of the two-stream co-adaptation pilot.

The offload branch answered one question and closed: taking whitespace-selection work
*away* from an already-trained single residual stream does not improve content. That
rules out loss removal as the mechanism. It leaves the representational hypothesis
standing -- PLE may only pay when local information has a separately maintained state, so
it never contaminates the residual the backbone uses for content, with joint training
learning when to read that state.

This is the smallest test of it. Three arms, everything else identical -- same documents,
token budget, window, schedule, seed, optimiser, and plain cross entropy throughout. No
whitespace masking; the objective is not what varies here.

    A   stock continued training. No sidecar, no lane.

    S   the known retrofit. The PLE sidecar writes straight into the ordinary residual
        at its own layer, exactly as C1 does.

    M   the same sidecar writing into a *private* lane. The backbone never sees it
        except through per-layer zero-initialised reads:

            m  =  PLE(h_l, features)
            h  <- h + s_l * g_l(h) * R_l(m)      for each layer in the window

Every arm is exactly stock Qwen3.5 at initialisation: S's ``value_proj`` and ``conv1d``
are zero, and M's read scales are zero on top of that.

The question is not only whether M beats A on content. The informative intermediate
result is **M keeping S's layout benefit without S's content damage**, which is what the
representational-interference hypothesis predicts and what a single stream cannot give.

Reads never see a token class or a target -- ``g_l`` is a function of the current stream
alone -- so any class structure in the learned gates is something joint training found,
not something this harness supplied.

**No gradient checkpointing here**, unlike ``offload_arms.py``. Arm M's lane is written
at one layer and read at nine others, and under non-reentrant checkpointing each layer is
recomputed independently during backward: the read layers would recompute against a lane
tensor whose graph has already been freed. Holding the activations costs a few GB and
keeps the three arms running through the identical code path.

    python scratch/ple_forensics/costream_arms.py --arm M --lr 1e-5 --output ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch import nn

from scratch.ple_forensics.offload_arms import DOCUMENTS, MANIFESTS, STUDENT, assistant_targets
from scratch.ple_forensics.router_check import whitespace_vocabulary
from scratch.ple_forensics.token_classes import CLASSES, class_of

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GGUF = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--unsloth--Qwen3.8-Flash-Next-GGUF", "snapshots",
    "38bb39ee97821de2c9009abb7e93950eec396e66", "UD-IQ4_XS",
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
)


class Wiring:
    """How an arm gets the n-gram table into the model, and how it is torn down.

    Arms A and S need no hooks at all: A runs with the sidecar bypassed, S runs it in
    place. Only M needs the lane, which is one pre-hook at the sidecar's layer that
    *captures* the write instead of letting it land, and one pre-hook at each read layer
    that adds a gated read back in.
    """

    def __init__(self, model, arm, read_layers, project):
        self.model = model
        self.arm = arm
        self.box = {"ngram_raw": None, "memory": None, "enabled": True}
        self.reads = nn.ModuleDict()
        self.handles = []
        self.read_layers = list(read_layers) if arm == "M" else []
        if arm != "M":
            return

        from distillkit.memory_lane import MemoryRead

        layer_index = model.config.sidecar_layer_index
        sidecar = model.model.layers[layer_index].sidecar

        def write_hook(module, args, kwargs):
            stream = args[0] if args else kwargs["hidden_states"]
            self.box["memory"] = sidecar.write(stream, self.box["ngram_raw"])
            return None                      # the stream itself is left untouched

        self.handles.append(model.model.layers[layer_index].register_forward_pre_hook(
            write_hook, with_kwargs=True))

        for index in self.read_layers:
            layer = model.model.layers[index]
            device = next(layer.parameters()).device
            read = MemoryRead(model.config.hidden_size, project=project).to(
                device=device, dtype=torch.float32)
            self.reads[str(index)] = read

            def read_hook(module, args, kwargs, read=read):
                if not self.box["enabled"] or self.box["memory"] is None:
                    return None
                stream = args[0] if args else kwargs["hidden_states"]
                stream = stream + read(stream, self.box["memory"])
                if args:
                    return (stream,) + tuple(args[1:]), kwargs
                kwargs["hidden_states"] = stream
                return args, kwargs

            self.handles.append(layer.register_forward_pre_hook(read_hook, with_kwargs=True))

    def parameters(self):
        return list(self.reads.parameters())

    def forward(self, ids, ngram_raw, device, output_hidden_states=False):
        """One document's model output, under this arm's wiring."""
        self.box["ngram_raw"] = ngram_raw
        self.box["memory"] = None
        inputs = torch.tensor([ids], device=device)
        mask = torch.ones(1, len(ids), dtype=torch.long, device=device)
        # input_ids first, always. Accelerate's io_same_device hook takes the device to
        # return outputs on from the first tensor it finds in the call's inputs, and
        # `ngram_raw` is a CPU uint8 tensor -- leading with it silently moves the whole
        # forward's output to the host, where the evaluation then runs at CPU speed or
        # dies on a device mismatch.
        if self.arm == "S":
            return self.model(input_ids=inputs, attention_mask=mask,
                              output_hidden_states=output_hidden_states,
                              ngram_raw=ngram_raw)
        return self.model(input_ids=inputs, attention_mask=mask,
                          output_hidden_states=output_hidden_states,
                          sidecar_enabled=False)

    def logits(self, ids, ngram_raw, device):
        """One document's full logits, under this arm's wiring."""
        return self.forward(ids, ngram_raw, device).logits[0]

    def diagnostics(self, positions):
        """Per-token read strength and read-to-stream norm ratio at `positions`."""
        out = {}
        for index, read in self.reads.items():
            if read.last_alpha is None:
                continue
            rows = torch.as_tensor(positions, device=read.last_alpha.device)
            out["alpha_L%s" % index] = read.last_alpha[0].index_select(0, rows).cpu().numpy()
            out["ratio_L%s" % index] = read.last_ratio[0].index_select(0, rows).cpu().numpy()
        return out

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def ngram_raw_of(ids, hasher, table):
    """The document's IQ4_NL rows, `[1, seq, heads, 90]` uint8, as the collator makes them."""
    rows = hasher.row_indices(torch.tensor([ids], dtype=torch.long))
    return torch.from_numpy(np.ascontiguousarray(table.gather_raw(rows)))


@torch.inference_mode()
def evaluate(wiring, documents, tokenizer, hasher, table, whitespace_index, limit, device,
             collect_reads=False):
    """Per-token NLL in document order, plus the whitespace factorisation and read stats."""
    out = {key: [] for key in ("nll", "target", "document", "detect", "select")}
    reads = {}
    for number, text in enumerate(documents):
        encoding = tokenizer(text, return_offsets_mapping=True)
        ids = encoding["input_ids"][:limit]
        targets = assistant_targets(text, encoding, len(ids))
        if not targets:
            continue
        raw = ngram_raw_of(ids, hasher, table)
        everything_logits = wiring.logits(ids, raw, device)
        positions = np.array(targets) - 1
        rows = torch.as_tensor(positions, device=everything_logits.device)
        logits = everything_logits.index_select(0, rows).float()
        del everything_logits
        target = torch.as_tensor([ids[i] for i in targets], device=logits.device)
        everything = torch.logsumexp(logits, dim=-1)
        within = torch.logsumexp(logits[:, whitespace_index.to(logits.device)], dim=-1)
        picked = logits.gather(1, target.unsqueeze(1)).squeeze(1)
        out["nll"].append((everything - picked).cpu().numpy())
        out["detect"].append((everything - within).cpu().numpy())
        out["select"].append((within - picked).cpu().numpy())
        out["target"].append(target.cpu().numpy())
        out["document"].append(np.full(len(targets), number, dtype=np.int32))
        if collect_reads:
            for key, value in wiring.diagnostics(positions).items():
                reads.setdefault(key, []).append(value)
        del logits
    packed = {key: np.concatenate(value) for key, value in out.items()}
    packed.update({key: np.concatenate(value) for key, value in reads.items()})
    return packed


def by_class(values, targets, tokenizer):
    labels = class_of(targets, tokenizer)
    return {name: float(values[labels == name].mean()) if (labels == name).any() else float("nan")
            for name in CLASSES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["A", "S", "M"], required=True)
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--train-docs", type=int, default=512)
    parser.add_argument("--eval-docs", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--window", type=int, nargs=2, default=(20, 28))
    parser.add_argument("--read-layers", type=int, nargs="+",
                        help="layers M reads the lane at; the trainable window by default")
    parser.add_argument("--project", action="store_true",
                        help="give each read its own identity-initialised matrix")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--trajectory-docs", type=int, default=48)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--baseline", action="store_true",
                        help="evaluate before training and exit; the shared reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from distillkit.independent_eval import unseen_records
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.ngram_hash import NGramHasher
    from distillkit.ngram_table import GGUFNGramTable

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    unseen = unseen_records(DOCUMENTS, MANIFESTS)
    if len(unseen) < args.train_docs + args.eval_docs:
        raise SystemExit("only %d unseen documents" % len(unseen))
    train_docs = [record["text"] for record in unseen[:args.train_docs]]
    eval_docs = [record["text"] for record in unseen[-args.eval_docs:]]
    print("%d train, %d eval, from %d unseen" % (len(train_docs), len(eval_docs), len(unseen)))

    table = GGUFNGramTable(args.gguf)
    hasher = NGramHasher()
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(STUDENT, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.sidecar_variant = "ple"
    model = Qwen35SidecarForCausalLM.from_pretrained(
        STUDENT, config=config, dtype=torch.bfloat16, local_files_only=True,
        device_map="auto")
    model.config.use_cache = False
    model.requires_grad_(False)

    low, high = args.window
    read_layers = args.read_layers if args.read_layers else list(range(low, high + 1))
    wiring = Wiring(model, args.arm, read_layers, args.project)

    trainable = []
    for index in range(low, high + 1):
        for parameter in model.model.layers[index].parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    backbone = sum(p.numel() for p in trainable)
    sidecar = model.model.layers[model.config.sidecar_layer_index].sidecar
    if args.arm in ("S", "M"):
        for parameter in sidecar.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    trainable.extend(wiring.parameters())
    print("arm %s: %.3fB backbone + %.2fM sidecar/lane trainable"
          % (args.arm, backbone / 1e9, (sum(p.numel() for p in trainable) - backbone) / 1e6))
    if args.arm == "M":
        print("lane written at layer %d, read at %s"
              % (model.config.sidecar_layer_index, read_layers))

    device = model.get_input_embeddings().weight.device
    whitespace_index = torch.as_tensor(
        whitespace_vocabulary(tokenizer, model.config.vocab_size),
        device=model.get_output_embeddings().weight.device)

    def score(documents, collect_reads=False):
        return evaluate(wiring, documents, tokenizer, hasher, table, whitespace_index,
                        args.tokens, device, collect_reads)

    if args.baseline:
        model.eval()
        scores = score(eval_docs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **scores)
        classes = by_class(scores["nll"], scores["target"], tokenizer)
        print("baseline %s" % json.dumps(classes))
        return 0

    initial = [parameter.detach().to("cpu", torch.float32).clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(args.warmup, 1)))

    model.train()
    started, step, seen = time.perf_counter(), 0, 0
    trajectory = []
    for start in range(0, len(train_docs), args.batch):
        prepared = []
        for text in train_docs[start:start + args.batch]:
            encoding = tokenizer(text, return_offsets_mapping=True)
            ids = encoding["input_ids"][:args.tokens]
            targets = assistant_targets(text, encoding, len(ids))
            if targets:
                prepared.append((ids, np.array(targets, dtype=np.int64)))
        if not prepared:
            continue
        optimizer.zero_grad(set_to_none=True)
        total, counted = 0.0, 0
        for ids, targets in prepared:
            raw = ngram_raw_of(ids, hasher, table)
            everything_logits = wiring.logits(ids, raw, device)
            rows = torch.as_tensor(targets - 1, device=everything_logits.device)
            logits = everything_logits.index_select(0, rows).float()
            target = torch.as_tensor([ids[i] for i in targets], device=logits.device)
            # Plain cross entropy, identical in all three arms. The objective is not the
            # variable here; where the sidecar writes is.
            loss = (torch.logsumexp(logits, dim=-1)
                    - logits.gather(1, target.unsqueeze(1)).squeeze(1)).sum()
            counted += len(targets)
            total += float(loss.detach())
            (loss / max(len(targets), 1)).backward()
            del everything_logits, logits, loss
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        step += 1
        seen += counted
        if step % 20 == 0:
            print("  step %4d  loss/token %.5f  %d targets  %.1fs"
                  % (step, total / max(counted, 1), seen, time.perf_counter() - started),
                  flush=True)
        if args.eval_every and step % args.eval_every == 0:
            model.eval()
            sampled = score(eval_docs[:args.trajectory_docs])
            model.train()
            point = {"step": step, **by_class(sampled["nll"], sampled["target"], tokenizer)}
            if args.arm == "M":
                point["read_scale"] = {index: float(read.scale.item())
                                       for index, read in wiring.reads.items()}
            trajectory.append(point)
            print("    [%4d] lexical %.5f  whitespace %.5f  punctuation %.5f  control %.5f"
                  % (step, point["lexical"], point["whitespace"],
                     point["punctuation"], point["control"]), flush=True)

    print("trained %d steps over %d targets in %.1fs"
          % (step, seen, time.perf_counter() - started))
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores = score(eval_docs, collect_reads=args.arm == "M")
    np.savez(args.output, **scores)
    summary = {"arm": args.arm, "lr": args.lr, "steps": step, "train_targets": seen,
               "eval_tokens": int(len(scores["nll"])),
               "assistant_nll": float(scores["nll"].mean()),
               "by_class": by_class(scores["nll"], scores["target"], tokenizer)}
    if args.arm in ("S", "M"):
        # Whether the write ever grew. A lane the backbone ignores and a lane that was
        # never written look identical in the NLL, and they are different failures: the
        # first is a routing result, the second is an optimisation one. `value_proj` and
        # `conv1d` start at exactly zero, so any norm here is movement.
        summary["sidecar"] = {name: float(parameter.detach().float().norm())
                              for name, parameter in sidecar.named_parameters()}

    if args.arm == "M":
        # The causal ablation: the same trained model with every read forced to zero.
        # Anything the lane bought has to disappear here, or it was never the lane.
        wiring.box["enabled"] = False
        off = score(eval_docs)
        off_path = args.output.with_name(args.output.stem + "-off.npz")
        np.savez(off_path, **off)
        wiring.box["enabled"] = True
        summary["memory_off"] = by_class(off["nll"], off["target"], tokenizer)
        summary["read_scale"] = {index: float(read.scale.item())
                                 for index, read in wiring.reads.items()}
        torch.save({index: read.state_dict() for index, read in wiring.reads.items()},
                   args.output.with_name(args.output.stem + "-reads.pt"))

    moved = float(np.sqrt(sum(
        float((parameter.detach().to("cpu", torch.float32) - begin).pow(2).sum())
        for parameter, begin in zip(trainable, initial))))
    reference = float(np.sqrt(sum(float(begin.pow(2).sum()) for begin in initial)))
    summary.update({"displacement": moved, "initial_norm": reference,
                    "relative_displacement": moved / max(reference, 1e-30)})
    if args.trajectory:
        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
        args.trajectory.write_text(
            json.dumps({"summary": summary, "trajectory": trajectory}, indent=2),
            encoding="utf-8")
    wiring.close()
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
