"""Multiple optimizer steps, accumulation, evaluation and resumable TP checkpoints."""
import numpy as np
import pytest
import torch

from test_sharding import _config, _split_map
from distillkit.configuration import DistillationRunConfig
from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
from distillkit.tp_checkpoint import consolidated_state_dict
from distillkit.tp_model import shard_model


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_tp_trainer_trajectory_and_fresh_resume(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from trl import SFTConfig
    from distillkit.data import CachedBatchCollator
    from distillkit.offline_cache import OfflineCacheWriter
    from distillkit.signals import OfflineHiddenStateSignalSource
    from distillkit.hsd_mapping import HiddenStateMapping
    from distillkit.trainer import HybridDistillationTrainer

    cache = tmp_path / "cache"
    with OfflineCacheWriter(cache, tokenizer_hash="ab"*32, anchor_layers=[1, 4],
                            hidden_size=48, vocab_size=64, sequence_length=12, top_k=4) as writer:
        for i in range(7):
            n = 6+i
            writer.append(str(i), np.arange(1, n+1, dtype=np.uint32),
                          np.tile(np.arange(1, 5, dtype=np.uint32), (n, 1)),
                          np.full((n, 4), -3, dtype=np.float16),
                          np.full((n, 2, 48), 56, dtype=np.uint8), split="train")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0")),
        pad_token="t0", eos_token="t1", unk_token="t0")
    torch.manual_seed(42)
    initial = tmp_path / "initial"
    Qwen35SidecarForCausalLM(_config()).save_pretrained(initial)

    def make(tp, output):
        model = Qwen35SidecarForCausalLM.from_pretrained(initial, **({} if tp else {"device_map": _split_map()}))
        if tp:
            shard_model(model, ["cuda:0", "cuda:1"])
        model.disable_sidecar_projection()
        torch.manual_seed(12)
        mapping = HiddenStateMapping(model, 48, [(1, 0), (4, 1)])
        training = dict(output_dir=str(output), report_to=[], num_train_epochs=1,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=2, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, max_length=12,
            save_strategy="steps", save_steps=2, eval_strategy="steps", eval_steps=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            optim="adamw_torch", learning_rate=2e-5, lr_scheduler_type="constant",
            warmup_steps=0, max_grad_norm=0.1, remove_unused_columns=False,
            disable_tqdm=True, bf16=False, fp16=False, seed=17,
            dataset_kwargs={"skip_prepare_dataset": True})
        config = DistillationRunConfig.model_validate(dict(
            model=str(initial), dataset={}, teacher={"kind": "dataset", "cache_path": str(cache)},
            sequence_length=12, output_path=str(output), use_flash_attention=False,
            optimizer={"strategy": "adamw", "freeze_backbone": False},
            sidecar={"enabled": False}, tensor_parallel=tp, training_args=training,
            loss_functions=[{"function": "kl", "weight": 0.7, "temperature": 1.0},
                            {"function": "hs_cosine", "weight": 0.3}], layer_mapping=[(1, 0), (4, 1)]))
        source = OfflineHiddenStateSignalSource(str(cache))
        dataset = source.cache.to_dataset("train")
        trainer = HybridDistillationTrainer(model=model, config=config, signal_source=source,
            true_vocab_size=64, hidden_state_mapping=mapping, args=SFTConfig(**training),
            train_dataset=dataset, eval_dataset=dataset.select([0, 1]),
            processing_class=tokenizer, data_collator=CachedBatchCollator(0))
        return trainer

    serial = make(False, tmp_path / "serial")
    serial_result = serial.train()
    tp = make(True, tmp_path / "tp")
    assert tp.is_model_parallel and not tp.place_model_on_device
    tp_result = tp.train()
    assert tp.state.global_step == serial.state.global_step == 4
    assert tp_result.training_loss == pytest.approx(serial_result.training_loss, rel=3e-4, abs=2e-5)
    for a, b in zip([x["eval_loss"] for x in tp.state.log_history if "eval_loss" in x],
                    [x["eval_loss"] for x in serial.state.log_history if "eval_loss" in x]):
        assert a == pytest.approx(b, rel=3e-4, abs=2e-5)
    canonical = consolidated_state_dict(tp.model)
    for name, expected in serial.model.state_dict().items():
        torch.testing.assert_close(canonical[name], expected.cpu(), rtol=3e-3, atol=3e-5, msg=name)
    # Reload the export as an ordinary model; no TP implementation needed for inference.
    tp.save_model(tmp_path / "portable")
    reloaded = Qwen35SidecarForCausalLM.from_pretrained(tmp_path / "portable")
    for name, p in reloaded.named_parameters():
        torch.testing.assert_close(p, canonical[name], rtol=0, atol=0, msg=name)

    resumed = make(True, tmp_path / "resumed")
    resumed.train(resume_from_checkpoint=str(tmp_path / "tp" / "checkpoint-2"))
    assert resumed.state.global_step == 4
    for name, value in consolidated_state_dict(resumed.model).items():
        torch.testing.assert_close(value, canonical[name], rtol=3e-3, atol=3e-5, msg=name)
    # Prevent loading an optimizer indexed by shards onto unsharded parameters.
    with pytest.raises(ValueError, match="TP optimizer checkpoints"):
        serial.train(resume_from_checkpoint=str(tmp_path / "tp" / "checkpoint-2"))
    for trainer in (serial, tp, resumed):
        trainer.signal_source.cache.close()
