"""The atlas's estimates on a tiny widened hybrid: the change map sums to the measured loss
change, and the first-order unit importances track real mean-ablations."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas import Taps, accumulate, path_scores, unit_importance, units_of  # noqa: E402
from test_widened_residual import tiny_config  # noqa: E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402


def setup(seed=0):
    torch.manual_seed(seed)
    model = Qwen35WidenedForCausalLM(tiny_config()).float().eval()
    docs = [torch.randint(4, 64, (32,), dtype=torch.int32) for _ in range(3)]
    return model, docs


def test_trapezoid_change_map_sums_to_the_measured_change():
    model, docs = setup()
    params = list(model.named_parameters())
    base = {n: p.detach().clone() for n, p in params}
    # A 10% relative change, larger than a round's; trapezoid error there is ~1.5%.
    tuned = {n: p + 0.1 * (p.abs().mean() + 1e-3) * torch.randn_like(p) for n, p in base.items()}
    delta = {n: tuned[n] - base[n] for n in base}
    sums = {}
    for which, weights in (("base", base), ("tuned", tuned)):
        with torch.no_grad():
            for n, p in params:
                p.copy_(weights[n])
        loss, acc = path_scores(model, params, delta, docs)
        sums[which] = (loss, sum(float(r.sum()) for r, _ in acc.values()))
        # Rows and columns of a matrix are two views of the same total.
        for r, c in acc.values():
            if c is not None:
                assert abs(float(r.sum()) - float(c.sum())) < 1e-4 + 1e-3 * abs(float(r.sum()))
    measured = sums["tuned"][0] - sums["base"][0]
    trapezoid = 0.5 * (sums["base"][1] + sums["tuned"][1])
    assert abs(trapezoid - measured) < 0.05 * abs(measured)
    # One endpoint's gradient alone is the cruder estimate.
    assert abs(trapezoid - measured) <= abs(sums["tuned"][1] - measured)


def test_accumulate_sums_documents_in_fp32():
    p = torch.nn.Parameter(torch.ones(2, 3))
    acc = {}
    for _ in range(2):
        p.grad = torch.full((2, 3), 0.5)
        accumulate(acc, [("w", p)], {"w": torch.full((2, 3), 2.0)})
    rows, cols = acc["w"]
    assert rows.tolist() == [6.0, 6.0] and cols.tolist() == [4.0, 4.0, 4.0]


def test_unit_importance_tracks_exact_mean_ablation():
    model, docs = setup(1)
    units = units_of(model)
    names = {u[0] for u in units}
    assert {"L0.mixer", "L0.mlp", "L0.heads", "L1.mixer", "L2.mlp"} <= names
    clean, estimate, exact = unit_importance(model, units, docs, confirm=10)
    assert len(exact) == 10 and all(k in estimate for k in exact)
    # Heads are keyed one by one.
    assert any(k.startswith("L0.heads.") for k in estimate)
    est, real = zip(*[(estimate[k], exact[k]) for k in exact])
    assert np.corrcoef(est, real)[0, 1] > 0.7
    # Ablation hooks are removed afterwards: the clean loss comes back.
    with torch.inference_mode():
        from atlas import domain_loss
        assert abs(domain_loss(model, docs) - clean) < 1e-5


def test_taps_replace_one_head_only():
    model, docs = setup(2)
    units = units_of(model)
    seen = {}
    with Taps(units) as taps:
        taps.means = {u[0]: torch.zeros(32) for u in units}
        taps.ablate = ("L0.heads", 1)
        proj = dict((u[0], u[1]) for u in units)["L0.heads"]
        handle = proj.register_forward_pre_hook(lambda m, i: seen.setdefault("x", i[0].detach().clone()))
        with torch.inference_mode():
            model(input_ids=docs[0].long()[None])
        handle.remove()
    width = 32 // units[2][3]
    x = seen["x"]
    assert torch.all(x[..., width:2 * width] == 0) and not torch.all(x[..., :width] == 0)
