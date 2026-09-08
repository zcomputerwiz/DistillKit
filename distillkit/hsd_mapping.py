import torch
import torch.nn as nn
import transformers

from distillkit.sharding import hidden_state_device


class HiddenStateMapping:
    layer_mapping: list[tuple[int, int]]
    hidden_state_mapping: torch.nn.ModuleList | None

    def __init__(
        self,
        student: transformers.PreTrainedModel,
        teacher_hidden_size: int,
        layer_mapping: list[tuple[int, int]],
        init_strategy: str = "xavier",
        force_projection: bool = False,
    ):
        student_hidden_size = student.config.hidden_size
        need_projection = force_projection or (
            teacher_hidden_size != student_hidden_size
        )

        self.layer_mapping = layer_mapping
        if need_projection:
            self.projections = nn.ModuleList(
                [
                    nn.Linear(student_hidden_size, teacher_hidden_size, bias=False)
                    for _ in layer_mapping
                ]
            )

            # init projections
            for proj in self.projections:
                if init_strategy == "xavier":
                    nn.init.xavier_uniform_(proj.weight)
                elif init_strategy == "kaiming":
                    nn.init.kaiming_uniform_(proj.weight, nonlinearity="linear")
                elif init_strategy == "zero":
                    nn.init.zeros_(proj.weight)
                elif init_strategy == "identity":
                    # Initialize as truncated identity matrix
                    nn.init.zeros_(proj.weight)
                    min_dim = min(student_hidden_size, teacher_hidden_size)
                    with torch.no_grad():
                        proj.weight[:min_dim, :min_dim] = torch.eye(min_dim)
                else:
                    raise ValueError(f"Unknown projection_init: {init_strategy}")

            # slap 'em on the student so they're trained and saved
            embedding = next(student.get_input_embeddings().parameters())  # a shard, when vocab-parallel
            self.projections.to(dtype=embedding.dtype)
            # Each projection consumes one student anchor, and on a split model those
            # anchors are not all on the embeddings' card. A projection cannot be moved
            # later without orphaning its optimizer state, so place each one on its
            # anchor's card now. The trainer taps anchors at their own modules, so this
            # is where the tensor is produced *and* observed; the loss still aligns
            # devices defensively for callers that pass gathered states instead.
            for projection, (student_layer_idx, _) in zip(self.projections, layer_mapping):
                projection.to(device=hidden_state_device(student, student_layer_idx))
            student.add_module("distillation_projections", self.projections)
        else:
            self.projections = None
