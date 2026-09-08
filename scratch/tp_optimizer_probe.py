"""Per-phase VRAM peaks of a tensor-parallel training step, with the real optimizer.

scratch/tp_real_probe.py times forward+backward and never builds an optimizer, so its
peaks understate training: AdamW8bit adds 2 bytes per parameter of state that is
present during every step after the first. This runs two full steps and reports the
peak of each phase separately, because the first step has no optimizer state yet and
understates every later one by about 4 GiB.

    python scratch/tp_optimizer_probe.py 4096 [batch]

The batch argument answers "can per_device_train_batch_size go up?". Batches pad to
their longest member (``data.py``), so the case to measure is a length-grouped batch of
full-length sequences -- the most expensive group the sampler can build, not the mean.

Numbers recorded in PROGRESS.md ("Where the tied embedding lives, measured three ways")
came from this script.
"""

import sys

import yaml
import torch

from distillkit.anchor_tap import AnchorTap
from distillkit.chunked_head import HeadContext
from distillkit.configuration import DistillationRunConfig
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.hidden_state import compute_hs_loss
from distillkit.lossfuncs.kl import KLDLoss
from distillkit.main import load_student_model
from distillkit.signals import SparseSignal
from distillkit.tp_model import shard_model, sync_replicated_gradients

SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1
cfg = DistillationRunConfig.model_validate(
    yaml.safe_load(open("examples/qwen35_sidecar_stage2_tp.yml"))
)
for index in range(2):
    torch.cuda.set_per_process_memory_fraction(cfg.max_vram_fraction, index)
cap = cfg.max_vram_fraction * torch.cuda.get_device_properties(0).total_memory / 1024**3

model = load_student_model(cfg, 248077, 248320)
model = shard_model(model, ["cuda:0", "cuda:1"])
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
model.train()
print("cap %.2f GiB per card | batch %d x seq %d" % (cap, BATCH, SEQ))
print("weights            %6.2f / %6.2f GiB" % tuple(torch.cuda.memory_allocated(i) / 1024**3 for i in range(2)))

import bitsandbytes as bnb  # noqa: E402

optimizer = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=1e-5)

hsm = HiddenStateMapping(model, 5120, cfg.layer_mapping)
anchors = [a for a, _ in cfg.layer_mapping] + [model.config.num_hidden_layers]
sidecar = model.model.layers[cfg.sidecar.layer_index].sidecar
ids = torch.randint(0, 248000, (BATCH, SEQ), device="cuda:0")
mask = torch.ones(BATCH, SEQ, 1, dtype=torch.bool, device="cuda:0")
raw = torch.zeros(BATCH, SEQ, sidecar.num_heads, sidecar.bytes_per_head, dtype=torch.uint8)
signal = SparseSignal(
    sparse_ids=torch.randint(0, 248320, (BATCH, SEQ, 64), device="cuda:0"),
    sparse_values=torch.log_softmax(torch.randn(BATCH, SEQ, 64, device="cuda:0"), -1),
    log_values=True, generation_temperature=1.0,
    hidden_states=(torch.randn(BATCH, SEQ, 5120, device="cuda:0", dtype=torch.bfloat16),
                   torch.randn(BATCH, SEQ, 5120, device="cuda:0", dtype=torch.bfloat16)),
    vocab_size=248320,
)


def peak(label):
    print("%-18s %6.2f / %6.2f GiB" % (label, *[torch.cuda.max_memory_allocated(i) / 1024**3 for i in range(2)]))
    for i in range(2):
        torch.cuda.reset_peak_memory_stats(i)


def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with AnchorTap(model, anchors) as tap:
            out = model(input_ids=ids, ngram_raw=raw, sidecar_enabled=True,
                        return_dict=True, logits_to_keep=1)
        out.hidden_states = tap.states()
        kl = KLDLoss(temperature=1.0, sparse_chunk_length=256)(
            out, signal, mask=mask, hidden_state_mapping=hsm,
            head_context=HeadContext(out.hidden_states[model.config.num_hidden_layers],
                                     model.lm_head, vocab_size=248320, chunk_length=256))
        # The trainer reduces the loss terms onto one card; do the same here.
        loss = 0.7 * kl.to("cuda:0") + 0.3 * compute_hs_loss("cosine", out, signal, mask, hsm)
    peak("  forward+loss")
    loss.backward()
    sync_replicated_gradients(model)
    peak("  backward")


def tensor_bytes(tensors):
    totals = [0, 0]
    for t in tensors:
        totals[t.device.index] += t.numel() * t.element_size()
    return tuple(b / 1024**3 for b in totals)


print("first step (no optimizer state yet):")
step()
print("+ gradients        %6.2f / %6.2f GiB" % tensor_bytes(p.grad for p in model.parameters() if p.grad is not None))
optimizer.step()
print("+ AdamW8bit state  %6.2f / %6.2f GiB" % tensor_bytes(
    v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v) and v.dim()))
peak("  optimizer step")
optimizer.zero_grad(set_to_none=True)

print("steady state (optimizer state present):")
step()
optimizer.step()
peak("  optimizer step")
print("reserved           %6.2f / %6.2f GiB" % tuple(torch.cuda.memory_reserved(i) / 1024**3 for i in range(2)))
