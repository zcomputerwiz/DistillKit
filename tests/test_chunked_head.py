"""Folding the head into the chunk loop must change the number, not just the memory.

The point is to stop materializing `[batch, seq, 248320]` logits and their gradient --
1.89 GiB each at sequence 4096, the allocation in every OOM this project has hit. That
is only worth anything if the loss and its gradients are identical to projecting the
whole sequence first, so these compare against exactly that.
"""

import pytest
import torch

from distillkit.chunked_head import chunked_head_loss
from distillkit.lossfuncs.common import accumulate_over_chunks
from distillkit.lossfuncs.kl import sparse_kl_div_inner

VOCAB, HIDDEN, TOP_K = 512, 32, 8


def _fixture(batch=1, seq=64, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, seq, HIDDEN, generator=generator, dtype=dtype)
    head = torch.nn.Linear(HIDDEN, VOCAB, bias=False, dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(torch.randn(VOCAB, HIDDEN, generator=generator, dtype=dtype) * 0.05)
    ids = torch.randint(0, VOCAB, (batch, seq, TOP_K), generator=generator)
    values = torch.log_softmax(torch.randn(batch, seq, TOP_K, generator=generator), -1).to(dtype)
    return hidden, head, ids, values


@pytest.mark.parametrize("chunk", [None, 1, 7, 16, 64, 1000])
def test_matches_projecting_the_whole_sequence_first(chunk):
    hidden, head, ids, values = _fixture()
    reference = accumulate_over_chunks(head(hidden), ids, values, None, None, sparse_kl_div_inner)
    got = chunked_head_loss(hidden, head, ids, values, None, chunk, sparse_kl_div_inner)
    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_head_on_another_card_gives_the_same_loss_and_gradients():
    """Tensor parallelism parks the head off the home card; the loss must follow the
    weight there and the hidden-state gradient must come back."""
    hidden, head, ids, values = _fixture()
    reference_hidden = hidden.clone().requires_grad_(True)
    chunked_head_loss(reference_hidden, head, ids, values, None, 16, sparse_kl_div_inner).backward()
    reference_weight_grad = head.weight.grad.clone()
    head.weight.grad = None

    remote_head = head.to("cuda:1")
    remote_hidden = hidden.to("cuda:0").requires_grad_(True)
    got = chunked_head_loss(
        remote_hidden, remote_head, ids.to("cuda:1"), values.to("cuda:1"), None, 16,
        sparse_kl_div_inner,
    )
    assert got.device == torch.device("cuda", 1)
    got.backward()
    assert remote_hidden.grad.device == torch.device("cuda", 0)
    torch.testing.assert_close(remote_hidden.grad.cpu(), reference_hidden.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(remote_head.weight.grad.cpu(), reference_weight_grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("chunk", [None, 8, 16])
def test_gradients_match_through_both_the_head_and_the_hidden_state(chunk):
    """The recompute has to rebuild the projection, not just the reduction."""
    hidden, head, ids, values = _fixture()

    reference_hidden = hidden.clone().requires_grad_(True)
    accumulate_over_chunks(
        head(reference_hidden), ids, values, None, None, sparse_kl_div_inner
    ).backward()
    reference_weight_grad = head.weight.grad.clone()
    head.weight.grad = None

    chunked_hidden = hidden.clone().requires_grad_(True)
    chunked_head_loss(
        chunked_hidden, head, ids, values, None, chunk, sparse_kl_div_inner
    ).backward()

    torch.testing.assert_close(chunked_hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(head.weight.grad, reference_weight_grad, rtol=1e-5, atol=1e-7)


def test_mask_is_sliced_with_the_chunk():
    hidden, head, ids, values = _fixture(seq=32)
    mask = torch.zeros(1, 32, 1)
    mask[:, :10] = 1.0
    reference = accumulate_over_chunks(head(hidden), ids, values, mask, None, sparse_kl_div_inner)
    for chunk in (1, 3, 8, 32):
        got = chunked_head_loss(hidden, head, ids, values, mask, chunk, sparse_kl_div_inner)
        torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


def test_padded_head_is_truncated_to_the_signal_vocabulary():
    """The student's head is padded wider than the teacher's signal (248320 vs 248077).

    The trainer truncates the materialized logits; chunking has to do the same per
    chunk, or the log-sum-exp normalizes over columns the teacher never scored.
    """
    hidden, head, ids, values = _fixture()
    true_vocab = VOCAB - 7
    ids = ids.clamp(max=true_vocab - 1)
    reference = accumulate_over_chunks(
        head(hidden)[..., :true_vocab], ids, values, None, None, sparse_kl_div_inner
    )
    got = chunked_head_loss(
        hidden, head, ids, values, None, 16, sparse_kl_div_inner, vocab_size=true_vocab
    )
    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_never_materializes_the_full_vocabulary_logits():
    """The whole point: peak memory must not contain a [seq, vocab] tensor."""
    seq, vocab = 512, 32_000
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(1, seq, HIDDEN, generator=generator).cuda().requires_grad_(True)
    head = torch.nn.Linear(HIDDEN, vocab, bias=False).cuda()
    ids = torch.randint(0, vocab, (1, seq, TOP_K), generator=generator).cuda()
    values = torch.log_softmax(torch.randn(1, seq, TOP_K, generator=generator), -1).cuda()

    def peak(fn):
        head.weight.grad = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        fn().backward()
        return (torch.cuda.max_memory_allocated() - before) / 1024**2

    materialized = peak(
        lambda: accumulate_over_chunks(head(hidden), ids, values, None, 64, sparse_kl_div_inner)
    )
    folded = peak(
        lambda: chunked_head_loss(hidden, head, ids, values, None, 64, sparse_kl_div_inner)
    )
    full_logits_mb = seq * vocab * 4 / 1024**2
    print(f"\nfull logits {full_logits_mb:.0f} MB | materialized {materialized:.0f} MB "
          f"-> folded {folded:.0f} MB")
    # Materializing costs the logits and their gradient; folding should cost neither.
    assert materialized > full_logits_mb, "test is not measuring the logits allocation"
    assert folded < materialized - full_logits_mb, (
        f"folding saved only {materialized - folded:.0f} MB of the "
        f"{full_logits_mb:.0f} MB logits tensor"
    )


def test_kl_loss_matches_with_and_without_head_context():
    """The trainer hands KLDLoss a HeadContext instead of full logits.

    Both routes must produce the same number, or the memory saving is a silent
    change to what the run optimizes.
    """
    from types import SimpleNamespace

    from distillkit.chunked_head import HeadContext
    from distillkit.lossfuncs.kl import KLDLoss
    from distillkit.signals import SparseSignal

    hidden, head, ids, values = _fixture(seq=48)
    signal = SparseSignal(
        sparse_ids=ids, sparse_values=values, log_values=True,
        generation_temperature=1.0, hidden_states=None, vocab_size=VOCAB,
    )
    mask = torch.ones(1, 48, 1, dtype=torch.bool)
    loss_fn = KLDLoss(temperature=1.0, sparse_chunk_length=16)
    assert loss_fn.accepts_head_context()

    materialized = loss_fn(SimpleNamespace(logits=head(hidden)), signal, mask=mask)
    folded = loss_fn(
        SimpleNamespace(logits=head(hidden[:, -1:])), signal, mask=mask,
        head_context=HeadContext(hidden, head, vocab_size=VOCAB, chunk_length=16),
    )
    torch.testing.assert_close(folded, materialized, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_kl_loss_divides_on_the_heads_card():
    """The mask arrives on the batch's card; the divisor built from it must follow the
    result to the head's card, or two 0-dim CUDA tensors refuse to divide."""
    from types import SimpleNamespace

    from distillkit.chunked_head import HeadContext
    from distillkit.lossfuncs.kl import KLDLoss
    from distillkit.signals import SparseSignal

    hidden, head, ids, values = _fixture(seq=48)
    mask = torch.ones(1, 48, 1, dtype=torch.bool)
    mask[0, :5] = False  # so the divisor is not just the sequence length
    loss_fn = KLDLoss(temperature=1.0, sparse_chunk_length=16)

    def signal(device):
        return SparseSignal(
            sparse_ids=ids.to(device), sparse_values=values.to(device), log_values=True,
            generation_temperature=1.0, hidden_states=None, vocab_size=VOCAB,
        )

    reference = loss_fn(SimpleNamespace(logits=head(hidden)), signal("cpu"), mask=mask)
    head.to("cuda:1")
    remote = loss_fn(
        SimpleNamespace(logits=None), signal("cuda:0"), mask=mask.to("cuda:0"),
        head_context=HeadContext(hidden.to("cuda:0"), head, vocab_size=VOCAB, chunk_length=16),
    )
    assert remote.device == torch.device("cuda", 1)
    torch.testing.assert_close(remote.cpu(), reference, rtol=1e-4, atol=1e-6)


def test_config_rejects_chunked_head_with_cross_entropy(tmp_path):
    """cross_entropy reads the model's own loss over the full head.

    Under chunked_head the forward runs with logits_to_keep=1, so that loss cannot
    be computed -- fail at config time rather than at step 0 of a long run.
    """
    from distillkit.configuration import DistillationRunConfig

    payload = {
        "model": "x", "dataset": {}, "chunked_head": True, "sequence_length": 8,
        "teacher": {"kind": "dataset", "cache_path": str(tmp_path)},
        "output_path": str(tmp_path / "out"),
        "loss_functions": [{"function": "cross_entropy", "weight": 1.0}],
    }
    with pytest.raises(ValueError, match="cross_entropy"):
        DistillationRunConfig.model_validate(payload)

    payload["loss_functions"] = [{"function": "hs_cosine", "weight": 1.0}]
    with pytest.raises(ValueError, match="sparse divergence"):
        DistillationRunConfig.model_validate(payload)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_chunk_memory_is_budgeted_in_rows_not_positions(batch):
    """A chunk's logits are [batch, chunk_length, vocab].

    Reading the configured chunk_length as positions makes its memory scale with the
    batch: [4, 256, 248320] in fp32 is 970 MiB against 242 MiB at batch 1. The value
    is a row budget at batch 1, so the peak must stay flat as the batch grows.
    """
    hidden, head, ids, values = _fixture(batch=batch, seq=64)
    calls = []
    real_head = head.forward

    def counting(x):
        calls.append(x.shape[0] * x.shape[1])
        return real_head(x)

    head.forward = counting
    chunked_head_loss(hidden, head, ids, values, None, 16, sparse_kl_div_inner)
    head.forward = real_head
    assert max(calls) <= 16, f"chunk grew to {max(calls)} rows at batch {batch}"


def test_row_budget_does_not_change_the_value():
    """Rescaling the chunk must not move the number it computes."""
    hidden, head, ids, values = _fixture(batch=4, seq=64)
    reference = accumulate_over_chunks(head(hidden), ids, values, None, None, sparse_kl_div_inner)
    got = chunked_head_loss(hidden, head, ids, values, None, 16, sparse_kl_div_inner)
    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


class _CharacterTokenizer:
    """Explicit offsets, so role and causal-shift expectations are hand checkable."""
    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)),
                "offset_mapping": [(i, i+1) for i in range(len(text))]}


def test_assistant_spans_exclude_headers_empty_think_user_and_padding():
    from distillkit.lossfuncs.cross_entropy import assistant_token_mask
    text = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n</think>\n\nABC<|im_end|>\n"
            "<|im_start|>user\nV<|im_end|>\n<|im_start|>assistant\nDE")
    ids = torch.tensor([[0, 0] + list(map(ord, text)) + [0]])
    attention = ids.ne(0)
    got = assistant_token_mask(ids, attention, _CharacterTokenizer())
    expected = torch.zeros_like(attention)
    start = 2 + text.index("ABC")
    stop = 2 + text.index("<|im_start|>user\nV")
    expected[:, start:stop] = True
    expected[:, -3:-1] = True  # DE, excluding final padding
    assert torch.equal(got, expected)


def test_assistant_mask_refuses_inexact_roundtrip():
    from distillkit.lossfuncs.cross_entropy import assistant_token_mask
    class Wrong(_CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            result['input_ids'][0] += 1
            return result
    with pytest.raises(ValueError, match="match input_ids exactly"):
        assistant_token_mask(torch.tensor([[65, 66]]), None, Wrong())


@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_assistant_ce_matches_dense_value_and_gradients(chunk):
    from distillkit.chunked_head import HeadContext
    from distillkit.lossfuncs.cross_entropy import AssistantCrossEntropyLoss
    torch.manual_seed(83)
    head = torch.nn.Linear(5, 11, bias=False)
    hidden = torch.randn(2, 6, 5, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 3, -100, 5], [6, 7, 8, 9, 10, 0]])
    assistant = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 1, 1]], dtype=torch.bool)
    attention = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
    # Explicit predictor positions: first row 1,2,4; second row 0,1,3.
    expected = torch.nn.functional.cross_entropy(
        head(hidden)[[0,0,0,1,1,1], [1,2,4,0,1,3]], torch.tensor([2,3,5,7,8,10]))
    ref_grads = torch.autograd.grad(expected, (hidden,head.weight))
    loss_fn = AssistantCrossEntropyLoss(sparse_chunk_length=chunk)
    got = loss_fn(None, None, head_context=HeadContext(hidden,head,vocab_size=9),
                  labels=labels, assistant_mask=assistant, attention_mask=attention)
    grads = torch.autograd.grad(got, (hidden,head.weight))
    torch.testing.assert_close(got, expected)
    for a,b in zip(grads,ref_grads): torch.testing.assert_close(a,b)
    assert not loss_fn.requires_model_loss()
    assert loss_fn.requires_token_targets() and loss_fn.accepts_head_context()


def test_assistant_ce_empty_region_is_differentiable_zero_without_head():
    from distillkit.chunked_head import HeadContext
    from distillkit.lossfuncs.cross_entropy import AssistantCrossEntropyLoss
    hidden = torch.randn(1, 4, 5, requires_grad=True)
    head = torch.nn.Linear(5, 11, bias=False)
    def forbidden(x): raise AssertionError("empty assistant region projected the head")
    head.forward = forbidden
    result = AssistantCrossEntropyLoss()(None,None,head_context=HeadContext(hidden,head),
        labels=torch.ones(1,4,dtype=torch.long), assistant_mask=torch.zeros(1,4,dtype=torch.bool))
    assert result.item() == 0
    result.backward()
    assert torch.count_nonzero(hidden.grad) == 0


def test_assistant_ce_configuration_is_opt_in_and_uses_loss_registry(tmp_path):
    from distillkit.configuration import DistillationRunConfig
    from distillkit.trainer import create_loss_func
    payload = dict(model="x",dataset={},chunked_head=True,sequence_length=8,
        teacher={"kind":"dataset","cache_path":str(tmp_path)},output_path=str(tmp_path/'out'),
        loss_functions=[{"function":"assistant_cross_entropy","weight":1.,"sparse_chunk_length":4}])
    config = DistillationRunConfig.model_validate(payload)
    assert create_loss_func(config.loss_functions[0]).name() == 'assistant_cross_entropy'
    payload['chunked_head'] = False
    with pytest.raises(ValueError,match='requires chunked_head'):
        DistillationRunConfig.model_validate(payload)
    payload['chunked_head'] = True
    payload['training_args'] = {'packing':True}
    with pytest.raises(ValueError,match='unpacked'):
        DistillationRunConfig.model_validate(payload)


def test_trainer_assistant_ce_shares_chunked_forward_and_preserves_existing_kl():
    import threading
    from types import SimpleNamespace
    from test_sidecar_model import tiny_config
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.trainer import DistillationTrainer, create_loss_func
    from distillkit.configuration import LossFunctionConfig
    from distillkit.signals import SparseSignal
    from distillkit.lossfuncs.cross_entropy import assistant_token_mask
    torch.manual_seed(31)
    config = tiny_config()
    config.vocab_size = 128
    model = Qwen35SidecarForCausalLM(config).eval()
    text = '<|im_start|>user\nU<|im_start|>assistant\nAB'
    ids = torch.tensor([list(map(ord,text))])
    mask = torch.ones_like(ids)
    signal = SparseSignal(sparse_ids=torch.ones(1,len(text),1,dtype=torch.long),
        sparse_values=torch.zeros(1,len(text),1), log_values=True,
        generation_temperature=1., hidden_states=None,vocab_size=128)
    cfgs = [LossFunctionConfig(function='kl',weight=.7,temperature=1.,sparse_chunk_length=4)]
    trainer = SimpleNamespace(model=model, need_hidden_states=False, need_model_loss=False,
        need_token_targets=False, _kept_bf16_outputs=True, chunked_head=True,
        true_vocab_size=128, hidden_state_mapping=None, _head_chunk_length=4,
        processing_class=_CharacterTokenizer(), accelerator=SimpleNamespace(unwrap_model=lambda x:x),
        signal_source=SimpleNamespace(get_signal=lambda *a,**kw:signal),
        _loss_log_local=threading.local(),log=lambda *a:None,
        config=SimpleNamespace(dataset=SimpleNamespace(eos_label_token_ids=[]),sidecar=SimpleNamespace(enabled=False),
                               loss_functions=cfgs),loss_functions=[create_loss_func(cfgs[0])])
    trainer.total_distillation_loss = lambda *a,**kw: DistillationTrainer.total_distillation_loss(trainer,*a,**kw)
    calls = []
    original = model.forward
    def observed(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    model.forward = observed
    inputs = dict(input_ids=ids,attention_mask=mask)
    before = DistillationTrainer.compute_loss(trainer,model,dict(inputs))
    repeated = DistillationTrainer.compute_loss(trainer,model,dict(inputs))
    torch.testing.assert_close(repeated,before,rtol=0,atol=0)
    cfgs.append(LossFunctionConfig(function='assistant_cross_entropy',weight=.3,sparse_chunk_length=4))
    trainer.loss_functions.append(create_loss_func(cfgs[1]))
    trainer.need_token_targets = True
    combined = DistillationTrainer.compute_loss(trainer,model,dict(inputs))
    assert all(c['logits_to_keep']==1 and 'labels' not in c for c in calls)
    dense = original(**inputs, use_cache=False, sidecar_enabled=False).logits
    selected = assistant_token_mask(ids,mask,_CharacterTokenizer())[:,1:]
    ce = torch.nn.functional.cross_entropy(dense[:,:-1][selected],ids[:,1:][selected])
    torch.testing.assert_close(combined,.7*before+.3*ce,rtol=1e-5,atol=1e-6)
    combined.backward()
    assert torch.isfinite(model.get_input_embeddings().weight.grad).all()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_assistant_ce_vocab_shards_match_dense():
    from distillkit.chunked_head import HeadContext
    from distillkit.lossfuncs.cross_entropy import AssistantCrossEntropyLoss
    from distillkit.tp_vocab import VocabParallelEmbedding, VocabParallelHead
    torch.manual_seed(91)
    embedding = torch.nn.Embedding(64, 16).cuda()
    hidden = torch.randn(1, 9, 16,device='cuda:0',requires_grad=True)
    labels = torch.tensor([[0,3,31,32,63,1,44,9,21]],device='cuda:0')
    mask = torch.tensor([[0,1,1,0,1,1,0,1,1]],device='cuda:0',dtype=torch.bool)
    dense = torch.nn.Linear(16,64,bias=False).cuda()
    dense.weight.data.copy_(embedding.weight)
    fn = AssistantCrossEntropyLoss(3)
    expected = fn(None,None,head_context=HeadContext(hidden,dense),labels=labels,assistant_mask=mask)
    ref_h,ref_w = torch.autograd.grad(expected,(hidden,dense.weight))
    head = VocabParallelHead(VocabParallelEmbedding(embedding,['cuda:0','cuda:1']))
    got = fn(None,None,head_context=HeadContext(hidden,head),labels=labels,assistant_mask=mask)
    grads = torch.autograd.grad(got,(hidden,*head.shards))
    torch.testing.assert_close(got,expected)
    torch.testing.assert_close(grads[0],ref_h)
    torch.testing.assert_close(torch.cat([g.to('cuda:0') for g in grads[1:]]),ref_w)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_assistant_ce_peak_memory_is_bounded_across_vocabulary_sizes():
    from distillkit.chunked_head import HeadContext
    from distillkit.lossfuncs.cross_entropy import AssistantCrossEntropyLoss
    peaks = []
    for vocab in (16_000,64_000):
        hidden = torch.randn(1,513,16,device='cuda',requires_grad=True)
        head = torch.nn.Linear(16,vocab,bias=False,device='cuda').requires_grad_(False)
        labels = torch.randint(vocab,(1,513),device='cuda')
        assistant = torch.ones_like(labels,dtype=torch.bool)
        def measure(fn):
            hidden.grad = None
            torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            fn().backward(); torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated()-before
        dense = measure(lambda: torch.nn.functional.cross_entropy(
            head(hidden[:,:-1]).flatten(0,1),labels[:,1:].flatten()))
        # A row budget plus a byte ceiling limits projected activation memory.
        chunked = measure(lambda: AssistantCrossEntropyLoss(32)(None,None,
            head_context=HeadContext(hidden,head), labels=labels,assistant_mask=assistant))
        full = 512*vocab*4
        print(f"vocab={vocab}: dense peak={dense/2**20:.2f} MiB, "
              f"assistant chunked peak={chunked/2**20:.2f} MiB")
        assert dense >= full
        assert chunked < dense-full
        assert chunked < full/2
        peaks.append((dense,chunked))
        del hidden,head,labels,assistant
    # Increasing vocab 4x must not add a sequence-wide logits allocation.
    assert peaks[1][1]-peaks[0][1] < (peaks[1][0]-peaks[0][0])/2
