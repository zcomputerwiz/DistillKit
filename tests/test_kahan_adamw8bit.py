"""Kahan-compensated AdamW8bit against the two things it has to agree with.

bitsandbytes rounds each update into a bf16 weight to nearest, so an update smaller
than half an ulp is lost every step and never accumulates. On this model at lr 7.3e-6
that froze every weight with |w| >= 0.002 -- 87% of the elements over a 2,000-step run.
`KahanAdamW8bit` keeps the rounded-away part in a bf16 buffer and runs bitsandbytes'
own fp32 kernel on the compensated value, so:

* against stock AdamW8bit on bf16, it must move weights the stock optimizer freezes;
* against stock AdamW8bit on a float32 copy of the same weights -- the same kernel,
  fed the same gradients -- its compensated value must track the float32 trajectory.

Both the 8-bit path and the float32-state path bitsandbytes takes for tensors under
`min_8bit_size` (norms, `A_log`) are exercised, since the norms were the other thing
that never moved.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes 8-bit kernels need CUDA")

LR, STEPS = 1e-5, 200


def weights(numel, device="cuda"):
    """Magnitudes from well under to well over the rounding cliff at this learning rate."""
    torch.manual_seed(0)
    return (torch.logspace(-5, 0, numel, device=device)
            * torch.where(torch.rand(numel, device=device) < 0.5, -1.0, 1.0))


def gradients(numel, steps, device="cuda"):
    torch.manual_seed(1)
    return [torch.randn(numel, device=device) * 1e-3 + 1e-3 for _ in range(steps)]


def run(optimizer_class, start, dtype, grads, **kwargs):
    param = torch.nn.Parameter(start.clone().to(dtype))
    optimizer = optimizer_class([param], lr=LR, betas=(0.9, 0.95), weight_decay=0.1, **kwargs)
    for grad in grads:
        param.grad = grad.to(dtype)
        optimizer.step()
    return param, optimizer


@cuda
@pytest.mark.parametrize("numel", [8192, 1024], ids=["8-bit-state", "fp32-state"])
def test_stock_bf16_freezes_what_kahan_moves(numel):
    import bitsandbytes as bnb
    from training_step import KahanAdamW8bit

    start = weights(numel)
    grads = gradients(numel, STEPS)
    stock, _ = run(bnb.optim.AdamW8bit, start, torch.bfloat16, grads)
    kahan, _ = run(KahanAdamW8bit, start, torch.bfloat16, grads)

    large = start.abs() > 0.01                      # well past the cliff
    initial = start.to(torch.bfloat16)
    assert (stock.data[large] == initial[large]).all(), (
        "stock AdamW8bit moved a large bf16 weight; the premise of this test is wrong")
    moved = (kahan.data[large] != initial[large]).float().mean().item()
    # 200 steps of ~1e-5 is ~2e-3, more than an ulp for |w| up to about 0.5.
    assert moved > 0.5, "Kahan moved only %.1f%% of large weights" % (100 * moved)


@cuda
@pytest.mark.parametrize("numel", [8192, 1024], ids=["8-bit-state", "fp32-state"])
def test_kahan_tracks_the_float32_trajectory(numel):
    import bitsandbytes as bnb
    from training_step import KahanAdamW8bit

    start = weights(numel)
    grads = gradients(numel, STEPS)
    reference, _ = run(bnb.optim.AdamW8bit, start.to(torch.bfloat16).float(), torch.float32,
                       [g.to(torch.bfloat16).float() for g in grads])
    kahan, optimizer = run(KahanAdamW8bit, start, torch.bfloat16, grads)

    base = start.to(torch.bfloat16).float()
    compensated = kahan.data.float() + optimizer.state[kahan]["compensation"].float()
    # The buffer is itself bf16. The residual it holds can grow to half an ulp of the
    # weight, so for |w| near 1 its own ulp is comparable to one step and it carries
    # a few percent of noise; for the matrix-sized weights that make up the model it is
    # a fraction of a percent. What must not happen is *bias*: movement systematically
    # lost is the defect being fixed, so the fit of the compensated movement to the
    # fp32 movement has to be 1, band by band, and the scatter around it bounded.
    for low, high, spread in ((1e-5, 1e-1, 2e-3), (1e-1, 1.0, 5e-2)):
        band = (start.abs() >= low) & (start.abs() < high)
        want = (reference.data - base)[band]
        got = (compensated - base)[band]
        scale = ((got * want).sum() / (want * want).sum()).item()
        error = ((got - want).norm() / want.norm()).item()
        assert abs(scale - 1) < 1e-2, (
            "|w| in [%g, %g): compensated movement is %.4f of the fp32 movement"
            % (low, high, scale))
        assert error < spread, (
            "|w| in [%g, %g): %.2f%% RMS error against the fp32 trajectory"
            % (low, high, 100 * error))
    # And the stored bf16 weight is the fp32 trajectory rounded, to within one ulp.
    ulp = kahan.data.float().abs() * 2 ** -7
    assert ((kahan.data.float() - reference.data).abs() <= ulp + 1e-12).all()


@cuda
def test_the_compensation_survives_a_save_and_reload():
    from training_step import KahanAdamW8bit

    numel = 8192
    start = weights(numel)
    grads = gradients(numel, 40)

    through, _ = run(KahanAdamW8bit, start, torch.bfloat16, grads)

    half, optimizer = run(KahanAdamW8bit, start, torch.bfloat16, grads[:20])
    saved = {"weight": half.data.clone(), "optimizer": optimizer.state_dict()}
    resumed = torch.nn.Parameter(saved["weight"].clone())
    fresh = KahanAdamW8bit([resumed], lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    fresh.load_state_dict(saved["optimizer"])
    assert "compensation" in fresh.state[resumed]
    for grad in grads[20:]:
        resumed.grad = grad.to(torch.bfloat16)
        fresh.step()

    torch.testing.assert_close(resumed.data, through.data, rtol=0, atol=0)


@cuda
def test_chunked_update_is_bit_identical_to_a_single_pass():
    """Chunking is only allowed to change memory, never a single bit of the result.

    The 8-bit state is quantized in independent 256-element blocks, each with its own
    absmax, so slicing at block boundaries must reproduce the one-pass update exactly --
    including a final chunk that is not a whole number of chunks long.
    """
    from training_step import KahanAdamW8bit

    numel = 256 * 37 + 256 * 5            # not a multiple of the chunk below
    start = weights(numel)
    grads = gradients(numel, 25)

    whole, whole_opt = run(KahanAdamW8bit, start, torch.bfloat16, grads)

    class Chunked(KahanAdamW8bit):
        chunk = 256 * 4                    # forces 11 chunks, the last one ragged

    sliced, sliced_opt = run(Chunked, start, torch.bfloat16, grads)
    torch.testing.assert_close(sliced.data, whole.data, rtol=0, atol=0)
    for key in ("compensation", "state1", "state2", "absmax1", "absmax2"):
        torch.testing.assert_close(sliced_opt.state[sliced][key],
                                   whole_opt.state[whole][key], rtol=0, atol=0)
    assert sliced_opt.state[sliced]["step"] == whole_opt.state[whole]["step"] == 25
