"""Does the model still use long context? NLL by position, and a pass-key at depth.

Every round so far trained at 1,536 tokens or less. The source is natively 262K (Gated
DeltaNet plus full attention), but conversion put CSA2 -- an indexer that picks 256
positions per query -- on its full-attention layers, and that indexer has only ever
chosen within short sequences; short training can also shift DeltaNet's decay gates
toward fast forgetting (SpectralShift, arXiv 2609.14320). This measures, per model:

* NLL by position bucket on long documents -- agent traces rendered with the chat
  template (SmolDataEnvs: Claude Code and Codex) and llama.cpp source concatenated per
  directory. A model that uses its context gets better further in; one that does not
  flattens out or gets worse.
* Pass-key retrieval: a five-digit key at 10/50/90% depth in filler text of 4K-32K
  tokens, scored as the key's log-probability and whether greedy decoding gets it.

Logits are projected a chunk at a time: at 32K positions the full [seq, 248320] row
would be 16 GB.

    python scratch/dense_gr/long_context_probe.py --arm source=../student-2b-hf \\
        --arm u50=scratch/dense_gr/merges-r6r8b/u50 --output long-context.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

# The served template (the teacher's, which the student adopted); same vocabulary as the source.
TOKENIZER = "D:/DeepThought/Projects/HybridModel/DistillKit/scratch/dense_gr/merges-r6r8b/u50"
AGENT = Path("D:/DeepThought/Projects/HybridModel/agent-data/smoldataenvs/conversations")
CODE = Path("D:/DeepThought/Projects/HybridModel/llama.cpp")
BUCKETS = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. "
          "There and back again. ")


def load(path):
    config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    if "Qwen35WidenedForCausalLM" in config.get("architectures", []):
        from distillkit.models import Qwen35WidenedForCausalLM as cls
        kwargs = {}
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as cls
        kwargs = {"attn_implementation": "flash_attention_2"}
    return cls.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True,
                               **kwargs).to("cuda").eval()


def plain(value):
    """Parquet hands back numpy arrays for lists and NaN for nulls; the template wants
    lists and None. Only a column itself is JSON text -- a tool output that happens to
    parse ("123") must stay a string."""
    if isinstance(value, str):
        value = json.loads(value)

    def walk(v):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, float) and v != v:
            return None
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v
    return walk(value)


def messages_of(row):
    messages = plain(row["prompt"]) + plain(row["completion"])
    for m in messages:
        if isinstance(m.get("content"), (dict, list)) and m["role"] == "tool":
            m["content"] = json.dumps(m["content"])  # mini-swe-agent's {returncode, output}
    return messages


def agent_documents(tokenizer, length, count, seed=0):
    import pandas as pd

    docs = []
    for harness in ("claude-code", "codex"):
        frame = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(AGENT / harness / "*.parquet"))))
        # One row per rollout, its last turn: the longest history.
        frame = frame.sort_values("turn_id").groupby("rollout_id").tail(1)
        for _, row in frame.sample(frac=1.0, random_state=seed).iterrows():
            messages = messages_of(row)
            text = tokenizer.apply_chat_template(messages, tools=plain(row["tools"]), tokenize=False)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= length:
                docs.append(("agent:" + harness, ids[:length]))
            if sum(d[0] == "agent:" + harness for d in docs) == count // 2:
                break
    return docs


def code_documents(tokenizer, length, count):
    docs = []
    folders = sorted({p.parent for p in CODE.glob("**/*.c*") if ".git" not in p.parts})
    for folder in folders:
        text = "".join("// %s\n%s\n" % (p.name, p.read_text(encoding="utf-8", errors="replace"))
                       for p in sorted(folder.glob("*")) if p.suffix in (".c", ".cpp", ".h", ".cu", ".cuh"))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= length:
            docs.append(("code", ids[:length]))
        if len(docs) == count:
            break
    return docs


@torch.inference_mode()
def position_nll(model, ids, chunk=2048):
    ids = torch.tensor([ids], device="cuda")
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    out = []
    for start in range(0, ids.shape[1] - 1, chunk):
        stop = min(start + chunk, ids.shape[1] - 1)
        logits = model.lm_head(hidden[start:stop]).float()
        out.append(F.cross_entropy(logits, ids[0, start + 1:stop + 1], reduction="none"))
    return torch.cat(out).cpu().numpy()


def passkey_cases(tokenizer, lengths, depths, keys=4, seed=0):
    rng = random.Random(seed)
    filler = tokenizer(FILLER, add_special_tokens=False)["input_ids"]
    cases = []
    for length in lengths:
        for depth in depths:
            for _ in range(keys):
                key = str(rng.randint(10000, 99999))
                needle = tokenizer(" The pass key is %s. Remember it. %s is the pass key. " % (key, key),
                                   add_special_tokens=False)["input_ids"]
                question = tokenizer(" What is the pass key? The pass key is", add_special_tokens=False)["input_ids"]
                answer = tokenizer(" " + key, add_special_tokens=False)["input_ids"]
                room = length - len(needle) - len(question) - len(answer)
                body = (filler * (room // len(filler) + 1))[:room]
                at = int(room * depth)
                cases.append(dict(length=length, depth=depth, key=key,
                                  prompt=body[:at] + needle + body[at:] + question, answer=answer))
    return cases


@torch.inference_mode()
def passkey(model, case):
    ids = torch.tensor([case["prompt"] + case["answer"]], device="cuda")
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    start = len(case["prompt"]) - 1
    logits = model.lm_head(hidden[start:start + len(case["answer"])]).float()
    target = torch.tensor(case["answer"], device="cuda")
    logp = -F.cross_entropy(logits, target, reduction="sum")
    return float(logp), bool((logits.argmax(-1) == target).all())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--documents", type=int, default=24, help="per kind: agent traces, code")
    parser.add_argument("--passkey-lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    docs = agent_documents(tokenizer, args.length, args.documents) + code_documents(tokenizer, args.length, args.documents)
    kinds = sorted({k for k, _ in docs})
    print("documents: %s at %d tokens" % ({k: sum(d[0] == k for d in docs) for k in kinds}, args.length), flush=True)
    cases = passkey_cases(tokenizer, args.passkey_lengths, [0.1, 0.5, 0.9])
    edges = [b for b in BUCKETS if b < args.length] + [args.length]

    report = {}
    for label, path in (item.split("=", 1) for item in args.arm):
        model = load(path)
        nll = {kind: np.mean([position_nll(model, ids) for k, ids in docs if k == kind], axis=0) for kind in kinds}
        buckets = {kind: [float(values[lo:hi].mean()) for lo, hi in zip(edges, edges[1:])] for kind, values in nll.items()}
        keys = [passkey(model, case) for case in cases]
        table = {}
        for case, (logp, hit) in zip(cases, keys):
            table.setdefault("%d" % case["length"], []).append((logp, hit))
        report[label] = dict(buckets=buckets, passkey={n: dict(logp=float(np.mean([v[0] for v in rows])),
                                                               hit=float(np.mean([v[1] for v in rows])))
                                                        for n, rows in table.items()})
        print("== %s" % label, flush=True)
        print("  bucket   " + "".join("%9s" % ("%dk" % (hi // 1024) if hi >= 1024 else hi) for hi in edges[1:]))
        for kind, row in buckets.items():
            print("  %-15s" % kind + "".join("%9.4f" % v for v in row), flush=True)
        print("  pass-key " + "  ".join("%s: hit %.0f%% logp %.2f" % (n, 100 * r["hit"], r["logp"])
                                        for n, r in report[label]["passkey"].items()), flush=True)
        del model
        torch.cuda.empty_cache()
    report["edges"] = edges
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
