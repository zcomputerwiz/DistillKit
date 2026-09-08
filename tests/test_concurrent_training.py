"""Numerical and failure gates for bounded threaded training."""
import threading

import pytest
import torch

from distillkit.concurrent_training import ConcurrentMicrobatches
from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.anchor_tap import AnchorTap
from distillkit.lossfuncs.hidden_state import compute_hs_loss
from distillkit.lossfuncs.kl import KLDLoss
from test_sharding import _config, _split_map, _batch


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
@pytest.mark.parametrize("checkpoint", [False, True])
def test_threaded_sidecar_gradients_and_update_match_serial(tmp_path, checkpoint):
    config = _config()
    path = tmp_path / "model"
    torch.manual_seed(7)
    Qwen35SidecarForCausalLM(config).save_pretrained(path)
    models = [Qwen35SidecarForCausalLM.from_pretrained(path, device_map=_split_map()) for _ in range(2)]
    mappings = []
    for model in models:
        torch.manual_seed(9)
        mappings.append(HiddenStateMapping(model, 48, [(1, 0), (4, 1)]))
        model.train()
        if checkpoint:
            model.gradient_checkpointing_enable({"use_reentrant": False, "preserve_rng_state": False})
    for dst, src in zip(mappings[1].projections, mappings[0].projections):
        dst.weight.data.copy_(src.weight.data)

    # An odd window exercises a final lone worker. Different sequences detect tap
    # cross-contamination that identical microbatches would conceal.
    batches = [_batch(64, 48, batch=1, seq=8+i, seed=i) for i in range(3)]
    def loss_for(model, mapping, batch):
        ids, mask, signal = batch
        sidecar = model.model.layers[1].sidecar
        raw = torch.zeros(*ids.shape, sidecar.num_heads, sidecar.bytes_per_head, dtype=torch.uint8, device=ids.device)
        raw[..., 1::18] = 60  # finite fp16 scale of 1 in each IQ4_NL block
        with AnchorTap(model, [1, 4]) as tap:
            outputs = model(input_ids=ids, ngram_raw=raw, return_dict=True)
        outputs.hidden_states = tap.states()
        kl = KLDLoss(temperature=1.0)(outputs, signal, mask=mask, hidden_state_mapping=mapping)
        hs = compute_hs_loss("cosine", outputs, signal, mask, mapping)
        return kl.to("cuda:0") + hs.to("cuda:0"), {}

    serial = 0.0
    for batch in batches:
        loss, _ = loss_for(models[0], mappings[0], batch)
        (loss / len(batches)).backward()
        serial += loss.detach().item() / len(batches)
    # Prime lazy CUDA/Triton initialization outside worker threads.
    with torch.no_grad():
        loss_for(models[1], mappings[1], batches[0])
    result, _ = ConcurrentMicrobatches(models[1]).run(
        batches, lambda batch: loss_for(models[1], mappings[1], batch),
    )
    assert result == pytest.approx(serial, rel=2e-4, abs=2e-5)
    for (name, ref), (other_name, actual) in zip(models[0].named_parameters(), models[1].named_parameters()):
        assert name == other_name
        assert (ref.grad is None) == (actual.grad is None), name
        if ref.grad is not None:
            torch.testing.assert_close(actual.grad, ref.grad, rtol=3e-3, atol=3e-5, msg=name)
    for model in models:
        torch.optim.AdamW(model.parameters(), lr=1e-4).step()
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    for (name, ref), (_, actual) in zip(models[0].named_parameters(), models[1].named_parameters()):
        torch.testing.assert_close(actual, ref, rtol=3e-3, atol=3e-5, msg=name)

    # Fail after both workers enter their callbacks, including a waiter at the head.
    barrier = threading.Barrier(2, timeout=10)
    def fail(batch):
        barrier.wait()
        if batch == 0:
            raise ValueError("injected worker failure")
        return loss_for(models[1], mappings[1], batches[1])
    hooks_before = len(models[1].lm_head._forward_pre_hooks)
    with pytest.raises((ValueError, RuntimeError), match="injected|cancelled"):
        ConcurrentMicrobatches(models[1]).run([0, 1], fail)
    assert all(p.grad is None for p in models[1].parameters())
    assert len(models[1].lm_head._forward_pre_hooks) == hooks_before


def test_tap_does_not_retain_foreign_thread_graphs():
    model = Qwen35SidecarForCausalLM(_config()).eval()
    ids = torch.ones(1, 8, dtype=torch.long)
    with AnchorTap(model, [1]) as tap:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(model, input_ids=ids, sidecar_enabled=False).result()
        with pytest.raises(RuntimeError, match="never produced"):
            tap.states()
        assert not tap._captured


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
@pytest.mark.parametrize("bf16", [False, True])
def test_trainer_short_accumulation_window_matches_serial(tmp_path, bf16):
    import numpy as np
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from trl import SFTConfig
    from distillkit.configuration import DistillationRunConfig
    from distillkit.data import CachedBatchCollator
    from distillkit.offline_cache import OfflineCacheWriter
    from distillkit.signals import OfflineHiddenStateSignalSource
    from distillkit.trainer import HybridDistillationTrainer

    cache = tmp_path / "cache"
    writer = OfflineCacheWriter(cache, tokenizer_hash="ab"*32, anchor_layers=[1, 4],
                                hidden_size=48, vocab_size=64, sequence_length=12, top_k=4)
    for i in range(5):
        n = 7+i
        writer.append(str(i), np.arange(1, n+1, dtype=np.uint32),
                      np.tile(np.arange(1, 5, dtype=np.uint32), (n, 1)),
                      np.full((n, 4), -3.0, dtype=np.float16),
                      np.full((n, 2, 48), 56, dtype=np.uint8), split="train")
    writer.close()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0")),
        pad_token="t0", eos_token="t1", unk_token="t0",
    )
    path = tmp_path / "initial"
    torch.manual_seed(42)
    Qwen35SidecarForCausalLM(_config()).save_pretrained(path)
    results = []
    def check_parameter(actual, expected, name):
        # FP32 is the real correctness gate and stays tight: a dropped, doubled or
        # mis-scaled microbatch shows up there as a relative error far above 3e-3.
        if actual.dtype is not torch.bfloat16:
            torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-5,
                                       msg=lambda message: f"{name}: {message}")
            return
        # BF16 keeps 8 mantissa bits, so one Adam step at lr 1e-4 on a weight of
        # ~3e-3 spans only about 8 representable values. A last-bit difference in the
        # accumulated gradient therefore re-rounds a large fraction of the tensor --
        # measured at up to 37% of elements, every one of them within a couple of ulp.
        # Counting differing elements is uninformative here; bounding the magnitude
        # is the check that means something, and FP32 above is the real gate.
        torch.testing.assert_close(actual, expected, rtol=8 * torch.finfo(torch.bfloat16).eps,
                                   atol=1e-4, msg=lambda message: f"{name}: {message}")

    for workers in (1, 2):
        model = Qwen35SidecarForCausalLM.from_pretrained(
            path, device_map=_split_map(), dtype=torch.bfloat16 if bf16 else torch.float32,
        )
        model.disable_sidecar_projection()
        torch.manual_seed(12)
        mapping = HiddenStateMapping(model, 48, [(1, 0), (4, 1)], force_projection=True)
        output = tmp_path / f"workers-{workers}"
        training = dict(output_dir=str(output), report_to=[], num_train_epochs=1,
                        per_device_train_batch_size=1, gradient_accumulation_steps=3,
                        gradient_checkpointing=True,
                        gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": False},
                        max_length=12, save_strategy="steps", save_steps=1, save_total_limit=2,
                        eval_strategy="no", optim="adamw_torch",
                        learning_rate=1e-4, lr_scheduler_type="constant", warmup_steps=0,
                        max_grad_norm=0, remove_unused_columns=False, disable_tqdm=True,
                        bf16=bf16, fp16=False, seed=17, dataset_kwargs={"skip_prepare_dataset": True})
        config = DistillationRunConfig.model_validate(dict(
            model=str(path), dataset={}, teacher={"kind": "dataset", "cache_path": str(cache)},
            sequence_length=12, output_path=str(output), use_flash_attention=False,
            optimizer={"strategy": "adamw", "freeze_backbone": False},
            sidecar={"enabled": False}, concurrent_microbatches=workers,
            training_args=training, loss_functions=[{"function": "kl", "weight": 0.7, "temperature": 1.0}, {"function": "hs_cosine", "weight": 0.3}],
            layer_mapping=[(1, 0), (4, 1)],
        ))
        source = OfflineHiddenStateSignalSource(str(cache))
        trainer = HybridDistillationTrainer(model=model, config=config, signal_source=source,
            true_vocab_size=64, hidden_state_mapping=mapping, args=SFTConfig(**training),
            train_dataset=source.cache.to_dataset("train"), processing_class=tokenizer,
            data_collator=CachedBatchCollator(0))
        result = trainer.train()
        assert trainer.state.global_step == 2
        assert not trainer._concurrent_pending
        results.append((result.training_loss, {n: p.detach().cpu().clone() for n, p in model.named_parameters()}))
        # Exercise HF saving after the worker join, including the shared embedding.
        trainer.save_model(output)
        assert (output / "config.json").exists()
        if workers == 2:
            trainer.train(resume_from_checkpoint=str(output / "checkpoint-1"))
            assert trainer.state.global_step == 2
            assert not trainer._concurrent_pending
            for name, p in model.named_parameters():
                check_parameter(p.detach().cpu(), results[-1][1][name], name)
        source.cache.close()
    assert results[1][0] == pytest.approx(results[0][0], rel=2e-4, abs=2e-5)
    for name, ref in results[0][1].items():
        check_parameter(results[1][1][name], ref, name)
