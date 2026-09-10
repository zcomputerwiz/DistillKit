"""CPU-only reproduction of the old sketch's forced-PLE loading error."""

import json
from pathlib import Path

import torch
from transformers import AutoConfig

from distillkit.independent_eval import write_json
from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM


path = Path("../runs/gr-stage1-1m")
config = AutoConfig.from_pretrained(path, local_files_only=True)
saved_variant = config.sidecar_variant
# heldout_ce.py's YAML passes this override through load_student_model.
config.sidecar_variant = "ple"
model, info = Qwen35SidecarForCausalLM.from_pretrained(
    path, config=config, local_files_only=True, dtype=torch.bfloat16,
    output_loading_info=True,
)
ple = model.model.layers[1].sidecar.ple
report = {
    "saved_variant": saved_variant,
    "legacy_forced_variant": config.sidecar_variant,
    "missing_keys": sorted(info["missing_keys"]),
    "unexpected_keys": sorted(info["unexpected_keys"]),
    "fresh_ple_value_proj_norm": ple.value_proj.weight.float().norm().item(),
    "fresh_ple_conv_norm": ple.conv1d.weight.float().norm().item(),
}
write_json("scratch/independent-eval/legacy-loading-audit.json", report)
print(json.dumps(report))
