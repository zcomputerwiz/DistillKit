"""Independent recomputation of Codex's assistant-only widened-minus-control claim."""
import json
import numpy as np

def role_sums(path, role="assistant"):
    d = json.load(open(path, encoding="utf-8"))
    recs = d["records"]["nll"]
    ids = [r["id"] for r in recs]
    return ids, {mode: np.array([r["modes"][mode]["by_role"][role]["sum_nll"] for r in recs])
                 for mode in recs[0]["modes"]}, \
           np.array([recs[0]["modes"]["enabled"]["by_role"][role]["tokens"] for _ in recs]) * 0 + \
           np.array([r["modes"]["enabled"]["by_role"][role]["tokens"] for r in recs])

def paired(a, b, tokens, draws=10000, seed=0):
    rng = np.random.default_rng(seed)
    delta = a - b
    idx = rng.integers(0, len(delta), (draws, len(delta)))
    samples = delta[idx].sum(1) / tokens[idx].sum(1)
    return delta.sum() / tokens.sum(), np.quantile(samples, [0.025, 0.975])

base = "scratch/independent-eval/"
pairs = [("stage1", "reply-widened-ple-stage1-1m.json", "reply-ple-stage1-1m.json"),
         ("stage2", "reply-widened-ple-stage2-5m.json", "reply-ple-control-stage2-5m.json")]
for stage, wide, ctrl in pairs:
    iw, w, tw = role_sums(base + wide)
    ic, c, tc = role_sums(base + ctrl)
    assert iw == ic, "unpaired documents"
    assert (tw == tc).all(), "assistant token counts differ between arms"
    for mode in ("enabled", "bypassed"):
        est, ci = paired(w[mode], c[mode], tw)
        print("%s %-9s widened - control  %+.6f  [%+.6f, %+.6f]  %d docs, %d tokens"
              % (stage, mode, est, ci[0], ci[1], len(tw), tw.sum()))
