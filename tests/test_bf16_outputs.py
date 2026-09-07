"""accelerate's fp32 output upcast must go, and going must change no numbers.

`Accelerator.prepare_model` wraps a mixed-precision forward in
`convert_outputs_to_fp32`, which calls `.float()` on every bf16 tensor the model
returns. Over a 248,320-wide head that is a 3.79 GiB logits allocation at sequence
4096 plus another for its gradient -- it OOM'd the 1M control arm and the first
stage-2 attempt -- and it allocates exactly the full-vocabulary fp32 tensor that
`sparse_chunk_length` exists to avoid.

Removing it is only safe because widening bf16 to fp32 is exact, so every consumer
that upcasts what it needs gets identical numbers. These pin that.
"""

from types import MethodType

import pytest
import torch
from accelerate.utils.operations import ConvertOutputsToFp32, convert_outputs_to_fp32

from distillkit.chunked_ce import keep_bf16_forward_outputs
from distillkit.lossfuncs.common import accumulate_over_chunks
from distillkit.lossfuncs.kl import sparse_kl_div_inner


class _Model(torch.nn.Module):
    """Returns a structure shaped like CausalLMOutput: a tensor plus a tuple."""

    def forward(self, x):
        hidden = x * 2
        return {"logits": hidden, "hidden_states": (hidden, hidden + 1)}


def _bf16_input():
    return torch.tensor([[1.5, -2.25]], dtype=torch.bfloat16)


def test_wrapper_is_removed_and_outputs_stay_bf16():
    model = _Model()
    model.forward = MethodType(
        convert_outputs_to_fp32(model.forward.__func__), model
    )
    assert model(_bf16_input())["logits"].dtype is torch.float32

    assert keep_bf16_forward_outputs(model) is True
    output = model(_bf16_input())
    assert output["logits"].dtype is torch.bfloat16
    assert all(h.dtype is torch.bfloat16 for h in output["hidden_states"])


def test_values_are_unchanged_by_the_removal():
    """The upcast was exact, so removing it must not move a single number."""
    model = _Model()
    reference = model(_bf16_input())["logits"].float()

    model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
    keep_bf16_forward_outputs(model)
    torch.testing.assert_close(model(_bf16_input())["logits"].float(), reference, rtol=0, atol=0)


def test_removal_is_idempotent_and_safe_on_an_unwrapped_model():
    model = _Model()
    assert keep_bf16_forward_outputs(model) is False
    model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
    assert keep_bf16_forward_outputs(model) is True
    assert keep_bf16_forward_outputs(model) is False
    assert model(_bf16_input())["logits"].dtype is torch.bfloat16


def test_plain_function_forward_form_is_handled():
    """accelerate takes a different branch when model.forward has no __func__."""

    class _Callable:
        def forward(self, x):
            return {"logits": x * 2}

        def __call__(self, x):
            return self.forward(x)

    model = _Callable()
    model.forward = convert_outputs_to_fp32(model.forward)
    assert model(_bf16_input())["logits"].dtype is torch.float32
    assert keep_bf16_forward_outputs(model) is True
    assert model(_bf16_input())["logits"].dtype is torch.bfloat16


def test_autocast_is_preserved_not_stripped_with_it():
    """Only the output conversion goes; the autocast context must survive.

    Removing autocast as well would silently change every matmul in the forward.
    """
    entered = []

    def _autocast_marker(func):
        def wrapper(self, x):
            entered.append(True)
            return func(self, x)

        return wrapper

    model = _Model()
    inner = _autocast_marker(model.forward.__func__)
    model.forward = MethodType(inner, model)
    model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)

    keep_bf16_forward_outputs(model)
    model(_bf16_input())
    assert entered, "the autocast wrapper was removed along with the fp32 conversion"


@pytest.mark.parametrize("chunk", [None, 4, 16])
def test_sparse_kl_matches_whether_logits_arrive_bf16_or_prefloated(chunk):
    """The KL loss is the consumer that made the upcast look necessary.

    It upcasts each chunk itself, and widening bf16 is exact, so feeding it bf16
    logits must give the same number as feeding it the fp32 copy accelerate used to
    hand over.
    """
    generator = torch.Generator().manual_seed(0)
    batch, seq, vocab, top_k = 1, 32, 512, 8
    logits = torch.randn(batch, seq, vocab, generator=generator).to(torch.bfloat16)
    ids = torch.randint(0, vocab, (batch, seq, top_k), generator=generator)
    values = torch.log_softmax(torch.randn(batch, seq, top_k, generator=generator), -1)

    def run(tensor):
        return accumulate_over_chunks(
            tensor, ids, values, None, chunk, sparse_kl_div_inner
        )

    # Exact, not merely close: widening bf16 to fp32 loses nothing, so the chunked
    # upcast reproduces the old fp32-input numbers bit for bit.
    torch.testing.assert_close(run(logits.float()), run(logits), rtol=0, atol=0)
