"""Exercise actual updates, frozen transitions, and optimizer/scheduler resumes."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import TrainerCallback, TrainerControl, TrainerState

from distillkit.gated_residual import GatedResidual
from distillkit.optimizers import (
    ArchitectureMetricsCallback,
    MixedMuonAdamW,
    UnfreezeBackboneCallback,
    build_mixed_optimizer,
    freeze_backbone_for_stage1,
    mixed_parameter_groups,
    validate_optimizer_backend,
)


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(12, 4)
        self.hidden = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.W_side_proj = nn.Linear(4, 4, bias=False)
        self.gated_residual = GatedResidual(4, num_branches=2)
        self.distillation_projections = nn.ModuleList([nn.Linear(4, 6, bias=False)])
        self.lm_head = nn.Linear(4, 12, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, ids):
        x = self.norm(self.hidden(self.embed_tokens(ids)))
        x = self.gated_residual(x, self.W_side_proj(x))
        return self.lm_head(x).square().mean() + self.distillation_projections[0](x).square().mean()


def test_routing_covers_each_parameter_once_and_excludes_embeddings_gates_auxiliaries():
    model = TinyStudent()
    groups = mixed_parameter_groups(model)
    params = [p for group in groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params}) == len(list(model.parameters()))
    muon_names = [name for g in groups if g["optimizer_kind"] == "muon" for name in g["param_names"]]
    assert muon_names == ["hidden.weight"]
    assert all(p.ndim >= 2 for g in groups if g["decay"] for p in g["params"])
    assert all(p.ndim < 2 for g in groups if not g["decay"] for p in g["params"])


def test_mixed_updates_match_official_optimizers_and_closure_runs_once():
    torch.manual_seed(8)
    matrix = nn.Parameter(torch.randn(4, 3))
    vector = nn.Parameter(torch.randn(3))
    matrix_ref = nn.Parameter(matrix.detach().clone())
    vector_ref = nn.Parameter(vector.detach().clone())
    mixed = MixedMuonAdamW(
        [{"params": [matrix], "optimizer_kind": "muon"},
         {"params": [vector], "optimizer_kind": "adamw"}],
        lr=0.003, muon_lr=0.01, weight_decay=0.02,
    )
    muon = torch.optim.Muon([matrix_ref], lr=0.01, weight_decay=0.02, adjust_lr_fn="match_rms_adamw")
    adam = torch.optim.AdamW([vector_ref], lr=0.003, weight_decay=0.02, foreach=False)
    calls = []
    for _ in range(3):
        matrix.grad = torch.randn_like(matrix)
        vector.grad = torch.randn_like(vector)
        matrix_ref.grad = matrix.grad.clone()
        vector_ref.grad = vector.grad.clone()
        def closure():
            calls.append(1)
            return torch.tensor(7.0, requires_grad=True)
        assert mixed.step(closure).item() == 7
        muon.step()
        adam.step()
        assert torch.equal(matrix, matrix_ref)
        assert torch.equal(vector, vector_ref)
    assert len(calls) == 3
    assert "momentum_buffer" in mixed.state[matrix]
    assert "exp_avg" in mixed.state[vector]


def test_standard_optimizer_scheduler_state_roundtrip_reproduces_next_update(tmp_path):
    torch.manual_seed(9)
    model = TinyStudent()
    optimizer = build_mixed_optimizer(model, lr=0.003, muon_lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    ids = torch.tensor([[1, 2, 3]])
    for _ in range(2):
        optimizer.zero_grad()
        model(ids).backward()
        optimizer.step()
        scheduler.step()
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = TinyStudent()
    restored.load_state_dict(saved["model"])
    restored_optimizer = build_mixed_optimizer(restored)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.5)
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_scheduler.load_state_dict(saved["scheduler"])
    for student, opt, sched in [(model, optimizer, scheduler), (restored, restored_optimizer, restored_scheduler)]:
        opt.zero_grad()
        student(ids).backward()
        opt.step()
        sched.step()
    assert scheduler.get_last_lr() == restored_scheduler.get_last_lr()
    for original, resumed in zip(model.parameters(), restored.parameters()):
        assert torch.equal(original, resumed)


def test_frozen_backbone_is_in_optimizer_and_updates_only_after_scheduled_unfreeze():
    torch.manual_seed(7)
    model = TinyStudent()
    model.norm.bias.requires_grad_(False)  # A deliberate permanent freeze.
    frozen = freeze_backbone_for_stage1(model)
    assert "hidden.weight" in frozen and "norm.bias" not in frozen
    assert model.distillation_projections[0].weight.requires_grad
    assert model.W_side_proj.weight.requires_grad
    optimizer = build_mixed_optimizer(model, lr=0.003)
    before = model.hidden.weight.detach().clone()
    args = SimpleNamespace(deepspeed=None, fsdp=[], world_size=1)
    state, control = TrainerState(), TrainerControl()
    callback = UnfreezeBackboneCallback(1, frozen)
    callback.on_train_begin(args, state, control, model=model, optimizer=optimizer)
    model(torch.tensor([[2, 3]])).backward()
    optimizer.step()
    assert torch.equal(before, model.hidden.weight)
    assert model.hidden.weight not in optimizer.state
    state.global_step = 1
    callback.on_step_begin(args, state, control, model=model, optimizer=optimizer)
    optimizer.zero_grad()
    model(torch.tensor([[2, 3]])).backward()
    optimizer.step()
    assert not torch.equal(before, model.hidden.weight)
    assert model.hidden.weight in optimizer.state
    assert not model.norm.bias.requires_grad


def test_resume_unfreezes_before_first_forward_and_detects_dropped_backbone():
    model = TinyStudent()
    frozen = freeze_backbone_for_stage1(model)
    callback = UnfreezeBackboneCallback(3, frozen)
    args, state = SimpleNamespace(world_size=1), TrainerState(global_step=8)
    optimizer = build_mixed_optimizer(model, include_frozen=False)
    with pytest.raises(ValueError, match="omitted frozen"):
        callback.on_train_begin(args, state, TrainerControl(), model=model, optimizer=optimizer)
    optimizer = build_mixed_optimizer(model)
    callback.on_train_begin(args, state, TrainerControl(), model=model, optimizer=optimizer)
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("backend", [{"deepspeed": "zero2.json"}, {"fsdp": ["full_shard"]}])
def test_hybrid_rejects_partitioned_backends(backend):
    with pytest.raises(ValueError, match="partitioning/flattening"):
        validate_optimizer_backend(**backend)
    validate_optimizer_backend(strategy="adamw", **backend)


@pytest.mark.parametrize("backend", [{"deepspeed": "zero2.json"}, {"fsdp": ["full_shard"]}, {"world_size": 2}])
def test_dynamic_unfreezing_rejects_distributed_engines(backend):
    with pytest.raises(ValueError, match="single unpartitioned"):
        validate_optimizer_backend(strategy="adamw", dynamic_unfreeze=True, **backend)


def test_metrics_callback_records_gate_activations_and_projection_norms():
    model = TinyStudent()
    model(torch.tensor([[1, 2]]))
    callback = ArchitectureMetricsCallback(every_n_steps=2)
    state = TrainerState(global_step=2, log_history=[{"step": 2, "loss": 1.0}])
    control = TrainerControl()
    callback.on_step_end(None, state, control)
    assert control.should_log
    logs = {"loss": 1.0}
    callback.on_log(None, state, control, logs=logs, model=model)
    assert "architecture/W_side_proj/weight_norm" in logs
    assert "architecture/gated_residual/gate_1_mean" in logs
    assert "architecture/gated_residual/W_x_norm" in state.log_history[-1]


def test_duplicate_parameters_and_changed_checkpoint_routing_rejected():
    parameter = nn.Parameter(torch.randn(3, 3))
    with pytest.raises(ValueError, match="more than once"):
        MixedMuonAdamW([{"params": [parameter, parameter], "optimizer_kind": "muon"}])
    optimizer = build_mixed_optimizer(TinyStudent())
    saved = copy.deepcopy(optimizer.state_dict())
    saved["param_groups"][0]["optimizer_kind"] = "invalid"
    with pytest.raises(ValueError, match="routing differs"):
        optimizer.load_state_dict(saved)
    with pytest.raises(RuntimeError, match="future-unfrozen"):
        optimizer.add_param_group({"params": [parameter], "optimizer_kind": "muon"})


def test_hybrid_trainer_delivers_metrics_to_integrations_before_callback_copy(tmp_path):
    # Reporting integrations are registered ahead of user callbacks and consume
    # the log dict as they see it; a fake integration that copies immediately
    # must still receive gate/projection metrics alongside ordinary loss metrics.
    from datasets import Dataset
    from tokenizers import Tokenizer as FastTokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from trl import SFTConfig

    from distillkit.configuration import DistillationRunConfig
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.signals import OnlineSignalSource
    from distillkit.trainer import HybridDistillationTrainer
    from test_sidecar_model import raw_batch, tiny_config

    # In-memory stand-in so SFTTrainer has a processing class without files/network.
    words = [f"t{i}" for i in range(64)]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=FastTokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="t0")),
        pad_token="t0", eos_token="t1", unk_token="t0",
    )

    torch.manual_seed(3)
    model = Qwen35SidecarForCausalLM(tiny_config())
    # Prime gate activations so architecture_metrics has values to report;
    # gate stats are only stashed in train mode.
    with torch.no_grad():
        model(torch.tensor([[5, 8, 9, 3]]), ngram_raw=raw_batch(batch=1, length=4))
    model.eval()
    teacher_dir = tmp_path / "teacher"
    model.save_pretrained(teacher_dir)

    run_config = DistillationRunConfig.model_validate({
        "model": str(teacher_dir),
        "dataset": {"train_dataset": {"repo_id": "dummy/dummy"}},
        "teacher": {"kind": "hf", "path": str(teacher_dir)},
        "sequence_length": 8,
        "output_path": str(tmp_path / "out"),
        "use_flash_attention": False,
    })

    class FakeReporter(TrainerCallback):
        def __init__(self):
            self.received = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                self.received.append(dict(logs))  # copy immediately, like real integrations

    reporter = FakeReporter()
    metrics_callback = ArchitectureMetricsCallback(every_n_steps=2)
    trainer = HybridDistillationTrainer(
        model=model,
        config=run_config,
        signal_source=OnlineSignalSource(model, vocab_size=64),
        true_vocab_size=64,
        args=SFTConfig(
            output_dir=str(tmp_path / "out"), max_steps=1, report_to=[],
            per_device_train_batch_size=1, max_length=8,
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
        train_dataset=Dataset.from_dict(
            {"input_ids": [[1, 2, 3, 4]], "attention_mask": [[1, 1, 1, 1]]}
        ),
        processing_class=tokenizer,
        # Same relative ordering as Trainer: integrations before user callbacks.
        callbacks=[reporter, metrics_callback],
    )

    trainer.state.global_step = 100
    trainer.log({"loss": 0.5})
    assert reporter.received, "fake integration never received a log"
    payload = reporter.received[-1]
    assert payload["loss"] == 0.5
    assert any(k.startswith("architecture/") and k.endswith("/weight_norm") for k in payload)
    assert any("gate_1_mean" in k for k in payload)
    history = trainer.state.log_history[-1]
    assert set(payload) <= set(history), "local history must receive the same metrics"

    # Cadence: a step that is not a multiple of every_n_steps adds nothing new.
    trainer.state.global_step = 101
    trainer.log({"loss": 0.4})
    late = reporter.received[-1]
    assert late["loss"] == 0.4
    assert not any(k.startswith("architecture/") for k in late)


def test_dataparallel_gathered_loss_is_reduced_to_a_scalar():
    """nn.DataParallel returns one loss per replica; everything downstream wants a scalar.

    With two visible GPUs and no distributed launcher, HF Trainer wraps the model in
    nn.DataParallel, and `student_outputs.loss` comes back shaped [n_gpu]. The
    cross_entropy loss function returns that object directly, the trainer logs
    `loss.item()` on it, and the weighted sum broadcasts it -- so an unreduced loss
    fails with "a Tensor with 2 elements cannot be converted to Scalar", but only at
    batch > 1, since DataParallel cannot split a single example.
    """
    import torch
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from distillkit.lossfuncs.cross_entropy import CrossEntropyLoss

    gathered = CausalLMOutputWithPast(
        loss=torch.tensor([1.5, 2.5]), logits=torch.zeros(2, 3, 8)
    )
    # Mirrors DistillationTrainer.compute_loss's reduction.
    if gathered.loss is not None and gathered.loss.dim() > 0:
        gathered.loss = gathered.loss.mean()

    value = CrossEntropyLoss()(gathered, signal=None)
    assert value.dim() == 0, f"expected a scalar, got shape {tuple(value.shape)}"
    assert value.item() == pytest.approx(2.0)


@pytest.mark.parametrize(
    "student_dtype,proj_dtype,teacher_dtype",
    [
        (torch.bfloat16, torch.float32, torch.bfloat16),  # projections built in fp32
        (torch.float32, torch.bfloat16, torch.float32),   # the reverse, seen in practice
        (torch.bfloat16, torch.bfloat16, torch.float32),  # teacher upcast from fp8
    ],
)
def test_hidden_state_loss_tolerates_mixed_dtypes(student_dtype, proj_dtype, teacher_dtype):
    """The projection matmul must not depend on autocast being active at the call site.

    HiddenStateMapping builds its projections in fp32 after the student is loaded in
    bf16. Without explicit alignment this raises "expected mat1 and mat2 to have the
    same dtype" at step 0, naming neither the tensor nor the side that is wrong.
    """
    import torch.nn as nn
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from distillkit.lossfuncs.hidden_state import compute_hs_loss

    B, T, Hs, Ht = 2, 6, 16, 32

    class Mapping:
        layer_mapping = [(1, 0)]
        projections = nn.ModuleList([nn.Linear(Hs, Ht, bias=False).to(proj_dtype)])

    outputs = CausalLMOutputWithPast(
        logits=torch.zeros(B, T, 8),
        hidden_states=tuple(torch.randn(B, T, Hs, dtype=student_dtype) for _ in range(2)),
    )

    class Signal:
        hidden_states = (torch.randn(B, T, Ht, dtype=teacher_dtype),)

    for kind in ("mse", "cosine"):
        value = compute_hs_loss(kind, outputs, Signal(), mask=None, hidden_state_mapping=Mapping())
        assert value.dim() == 0 and torch.isfinite(value), f"{kind} -> {value}"
