"""The two collectives tensor parallelism needs, and their backward passes.

Everything else in a TP layer is ordinary matmul; these two primitives are where a
mistake is silent. A wrong backward on an all-reduce does not raise -- it trains
against gradients that are a fraction or a multiple of the truth, and the run just
learns more slowly. So both directions are checked against autograd's own answer for
the equivalent single-device computation.
"""

import pytest
import torch

from distillkit.tensor_parallel import all_reduce, peer_capable, replicate

TWO_GPUS = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two CUDA devices"
)


def test_all_reduce_of_one_shard_copies_nothing():
    """Degenerate case: nothing to reduce, so nothing should move.

    `Function.apply` always wraps outputs in a fresh autograd node, so identity is
    checked by storage rather than by `is`.
    """
    x = torch.randn(4, 8, requires_grad=True)
    (out,) = all_reduce([x])
    torch.testing.assert_close(out, x)
    assert out.data_ptr() == x.data_ptr(), "single-shard reduce made a copy"


def test_all_reduce_value_and_gradient_on_cpu():
    """Same algebra, no devices involved: y_i = x0 + x1 for every i."""
    a = torch.randn(4, 8, requires_grad=True)
    b = torch.randn(4, 8, requires_grad=True)
    out_a, out_b = all_reduce([a, b])
    torch.testing.assert_close(out_a, a + b)
    torch.testing.assert_close(out_b, a + b)

    # Each input feeds both outputs, so its gradient is the sum of both.
    ga, gb = torch.randn(4, 8), torch.randn(4, 8)
    torch.autograd.backward([out_a, out_b], [ga, gb])
    torch.testing.assert_close(a.grad, ga + gb)
    torch.testing.assert_close(b.grad, ga + gb)


def test_replicate_value_and_gradient_on_cpu():
    x = torch.randn(4, 8, requires_grad=True)
    first, second = replicate(x, ["cpu", "cpu"])
    torch.testing.assert_close(first, x)
    torch.testing.assert_close(second, x)

    g1, g2 = torch.randn(4, 8), torch.randn(4, 8)
    torch.autograd.backward([first, second], [g1, g2])
    torch.testing.assert_close(x.grad, g1 + g2)


@TWO_GPUS
def test_peer_access_is_available():
    assert peer_capable(["cuda:0", "cuda:1"]), (
        "tensor parallelism here assumes peer access; without it every reduction "
        "would stage through host memory"
    )


@TWO_GPUS
def test_all_reduce_across_devices_matches_single_device():
    a = torch.randn(64, 128, device="cuda:0", requires_grad=True)
    b = torch.randn(64, 128, device="cuda:1", requires_grad=True)
    out_a, out_b = all_reduce([a, b])

    reference = a + b.to("cuda:0")
    torch.testing.assert_close(out_a, reference)
    torch.testing.assert_close(out_b.to("cuda:0"), reference)
    assert out_a.device == a.device and out_b.device == b.device

    grad = torch.randn(64, 128, device="cuda:0")
    torch.autograd.backward([out_a, out_b], [grad, grad.to("cuda:1")])
    torch.testing.assert_close(a.grad, 2 * grad)
    torch.testing.assert_close(b.grad.to("cuda:0"), 2 * grad)


@TWO_GPUS
def test_column_then_row_parallel_matmul_matches_unsharded():
    """The pattern a TP MLP uses: split output channels, then input channels.

    y = (x @ W1) @ W2 with W1 split by columns and W2 by rows is exactly
    sum_i (x @ W1_i) @ W2_i -- one all-reduce and no gather. This is the whole
    arithmetic case for tensor parallelism, so it is checked end to end including
    both weight gradients.
    """
    torch.manual_seed(0)
    rows, hidden, inner = 32, 64, 96
    x = torch.randn(rows, hidden)
    w1 = torch.randn(hidden, inner) * 0.1
    w2 = torch.randn(inner, hidden) * 0.1

    reference_x = x.clone().to("cuda:0").requires_grad_(True)
    reference_w1 = w1.clone().to("cuda:0").requires_grad_(True)
    reference_w2 = w2.clone().to("cuda:0").requires_grad_(True)
    reference = (reference_x @ reference_w1) @ reference_w2
    reference.sum().backward()

    half = inner // 2
    shards = []
    for index, device in enumerate(("cuda:0", "cuda:1")):
        piece = slice(index * half, (index + 1) * half)
        shards.append((
            w1[:, piece].clone().to(device).requires_grad_(True),
            w2[piece, :].clone().to(device).requires_grad_(True),
        ))
    sharded_x = x.clone().to("cuda:0").requires_grad_(True)
    copies = replicate(sharded_x, ["cuda:0", "cuda:1"])
    partials = [(copy @ s1) @ s2 for copy, (s1, s2) in zip(copies, shards)]
    out_a, _ = all_reduce(partials)
    out_a.sum().backward()

    torch.testing.assert_close(out_a, reference, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sharded_x.grad, reference_x.grad, rtol=1e-4, atol=1e-5)
    for index, (s1, s2) in enumerate(shards):
        piece = slice(index * half, (index + 1) * half)
        torch.testing.assert_close(
            s1.grad.to("cuda:0"), reference_w1.grad[:, piece], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            s2.grad.to("cuda:0"), reference_w2.grad[piece, :], rtol=1e-4, atol=1e-5)


@TWO_GPUS
def test_reduction_costs_one_transfer_per_extra_device():
    """An all-pairs sum would move n^2 tensors; this must move 2(n-1)."""
    import distillkit.tensor_parallel as module

    moved = []
    real_to = torch.Tensor.to

    def counting_to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if isinstance(target, (str, torch.device)) and torch.device(target) != self.device:
            moved.append(1)
        return real_to(self, *args, **kwargs)

    torch.Tensor.to = counting_to
    try:
        a = torch.randn(8, 8, device="cuda:0")
        b = torch.randn(8, 8, device="cuda:1")
        module._sum_to_each((a, b))
    finally:
        torch.Tensor.to = real_to
    assert len(moved) == 2, f"expected 2 cross-device copies, made {len(moved)}"


def test_reduce_to_matches_all_reduce_but_returns_one_tensor():
    """Same sum, without copying the total back out to devices that never read it."""
    from distillkit.tensor_parallel import reduce_to

    a = torch.randn(4, 8, requires_grad=True)
    b = torch.randn(4, 8, requires_grad=True)
    total = reduce_to([a, b], "cpu")
    torch.testing.assert_close(total, a + b)

    grad = torch.randn(4, 8)
    total.backward(grad)
    # Each shard contributed additively, so each receives the whole gradient.
    torch.testing.assert_close(a.grad, grad)
    torch.testing.assert_close(b.grad, grad)


@TWO_GPUS
def test_reduce_to_costs_half_the_transfers_of_all_reduce():
    from distillkit.tensor_parallel import reduce_to
    import distillkit.tensor_parallel as module

    def count(fn):
        moved = []
        real_to = torch.Tensor.to

        def counting_to(self, *args, **kwargs):
            target = args[0] if args else kwargs.get("device")
            if isinstance(target, (str, torch.device)) and torch.device(target) != self.device:
                moved.append(1)
            return real_to(self, *args, **kwargs)

        torch.Tensor.to = counting_to
        try:
            fn()
        finally:
            torch.Tensor.to = real_to
        return len(moved)

    a = torch.randn(8, 8, device="cuda:0")
    b = torch.randn(8, 8, device="cuda:1")
    assert count(lambda: module._sum_to_each((a, b))) == 2
    assert count(lambda: reduce_to([a, b], "cuda:0")) == 1
