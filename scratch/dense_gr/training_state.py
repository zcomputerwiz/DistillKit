"""Deterministic batches and same-layout, step-boundary training resumption."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def position_weight(record):
    """A record's per-position loss weights, or None when every position weighs 1. An
    older record's boolean `supervised` mask is the 0/1 case. The loss and the budget
    both read weights through here, so they always charge the same thing."""
    if "weight" in record:
        return record["weight"]
    if "supervised" in record:
        return record["supervised"].float()
    return None


def coverage_order(teacher, groups, generator, count):
    """Assisted-by: Codex. One epoch permutation with source coverage up front.

    Select an existing group per retained source, fill and shuffle the prefix.
    No group is duplicated, truncated, or modified. Balances coverage, not tokens.
    """
    if not 1 <= count <= len(groups):
        raise ValueError("coverage prefix must fit within one epoch")
    owner = getattr(teacher.cache, "_owner", None)
    fallback = str(getattr(teacher.cache, "path", "cache"))
    candidates = {}
    for index, (docs, width) in enumerate(groups):
        sources = {str(owner[d][0]) if owner else fallback for d in docs}
        for source in sources:
            candidates.setdefault(source, []).append(index)
    if count < len(candidates):
        raise ValueError("coverage prefix needs at least one batch per source")
    anchors = set()
    for source in sorted(candidates):
        if not anchors.intersection(candidates[source]):
            anchors.add(int(generator.choice(candidates[source])))
    order = generator.permutation(len(groups)).tolist()
    prefix = sorted(anchors)
    prefix += [i for i in order if i not in anchors][:count - len(prefix)]
    generator.shuffle(prefix)
    chosen = set(prefix)
    return prefix + [i for i in order if i not in chosen]


class OrderedBatches:
    """Assisted-by: Codex. A finite, frozen sequence; never reshuffle or repeat.

    Validate each group against the current canonical plan. The ordinary plan's
    fingerprint binds manifests, masks and objective weights for resume safety.
    """
    def __init__(self, teacher, groups, manifest):
        canonical = {(tuple(docs), width) for docs, width in groups}
        selected = [(list(row['documents']), int(row['width'])) for row in manifest['groups']]
        if not selected or any((tuple(d), w) not in canonical for d, w in selected):
            raise ValueError('frozen batches must be nonempty canonical groups')
        self.teacher, self.groups, self.cursor = teacher, selected, 0
        base = PlannedBatches(teacher, selected, 0)
        self.fingerprint = base.fingerprint
        expected = manifest.get('cache_sha256')
        manifests = [c.manifest for c in getattr(teacher.cache, 'caches', [teacher.cache])]
        actual = hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()
        if expected != actual:
            raise ValueError('frozen batch cache manifests changed')

    def next_targets(self):
        if self.cursor == len(self.groups):
            return float('inf')
        docs, width = self.groups[self.cursor]
        return self.teacher.group_weight(docs, width)

    def take(self, remaining):
        if self.cursor == len(self.groups) or self.next_targets() > remaining:
            return None
        docs, width = self.groups[self.cursor]
        self.cursor += 1
        return self.teacher.read_batch(docs, width)

    def state_dict(self):
        return dict(fingerprint=self.fingerprint, cursor=self.cursor)

    def exposure(self):
        stats, seen = {}, set()
        for docs, width in self.groups:
            for doc in docs:
                source = Path(self.teacher.cache._owner[doc][0]).name
                row = stats.setdefault(source, [0, 0, 0.])
                row[0] += 1
                row[1] += doc not in seen
                seen.add(doc)
                row[2] += self.teacher.group_weight([doc], width)
        return stats

    def load_state_dict(self, state):
        if state['fingerprint'] != self.fingerprint or not 0 <= state['cursor'] <= len(self.groups):
            raise ValueError('resume frozen batch plan differs')
        self.cursor = state['cursor']


class PlannedBatches:
    def __init__(self, teacher, groups, seed, balanced_prefix_batches=0):
        if not groups:
            raise ValueError("no training documents survive the sample plan")
        self.teacher, self.groups = teacher, groups
        self.generator = np.random.default_rng(seed)
        self.order, self.cursor = [], 0
        if balanced_prefix_batches:
            self.order = coverage_order(teacher, groups, self.generator, balanced_prefix_batches)
        cache = getattr(teacher, "cache", None)
        captures = getattr(cache, "caches", [cache])
        identity = dict(groups=groups, manifests=[getattr(c, "manifest", None) for c in captures])
        if balanced_prefix_batches:
            identity["balanced_prefix_batches"] = balanced_prefix_batches
        # The loss weights are part of the plan too: a resume against changed answer spans,
        # weight, padding or masks would charge and optimize a different objective.
        weights = getattr(teacher, "weight_identity", None)
        if weights is not None:
            identity["weights"] = weights()
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def state_dict(self):
        return dict(fingerprint=self.fingerprint, order=self.order, cursor=self.cursor,
                    rng=self.generator.bit_generator.state)

    def load_state_dict(self, state):
        if state["fingerprint"] != self.fingerprint:
            raise ValueError("resume sample plan differs from checkpoint")
        self.order, self.cursor = list(state["order"]), state["cursor"]
        self.generator.bit_generator.state = state["rng"]

    def next_targets(self):
        if self.cursor == len(self.order):
            self.order = self.generator.permutation(len(self.groups)).tolist()
            self.cursor = 0
        group, width = self.groups[self.order[self.cursor]]
        # What the step will actually charge: the group's total loss weight, the same sum
        # the loss is normalized by (padding, assistant-only masks and answer weights).
        weight = getattr(self.teacher, "group_weight", None)
        return weight(group, width) if weight is not None else len(group) * (width - 1)

    def take(self, remaining):
        # Do not truncate a canonical prefix to spend a budget remainder. Stop
        # before the next whole microbatch; the unused targets are reported.
        if self.next_targets() > remaining:
            return None
        group, width = self.groups[self.order[self.cursor]]
        self.cursor += 1
        return self.teacher.read_batch(group, width)


def prefix_exposure(teacher, groups, seed, budget, balanced_prefix_batches=0):
    """What `budget` targets will see of each capture: the seed's whole-batch prefix, replayed
    as PlannedBatches.take() spends it. {capture name: [visits, distinct documents, targets]}."""
    replay = PlannedBatches(teacher, groups, seed, balanced_prefix_batches)
    owner = getattr(getattr(teacher, "cache", None), "_owner", None)
    seen, stats, spent = set(), {}, 0.0
    while True:
        targets = replay.next_targets()
        if targets > budget - spent:
            return stats
        group, width = groups[replay.order[replay.cursor]]
        replay.cursor += 1
        spent += targets
        for doc_id in group:
            row = stats.setdefault(Path(owner[doc_id][0]).name if owner else "all", [0, 0, 0.0])
            row[0] += 1
            row[1] += doc_id not in seen
            seen.add(doc_id)
            row[2] += teacher.group_weight([doc_id], width)


class WindowBatches:
    def __init__(self, stream, rows, width, seed, device="cuda"):
        if len(stream) < width or rows < 1 or width < 2:
            raise ValueError("invalid fixed-window sample plan")
        self.stream, self.rows, self.width, self.device = stream, rows, width, device
        self.generator = np.random.default_rng(seed)

    def state_dict(self):
        return dict(shape=[len(self.stream), self.rows, self.width],
                    rng=self.generator.bit_generator.state)

    def load_state_dict(self, state):
        if state["shape"] != [len(self.stream), self.rows, self.width]:
            raise ValueError("resume window plan differs from checkpoint")
        self.generator.bit_generator.state = state["rng"]

    def next_targets(self):
        return self.rows * (self.width - 1)

    def take(self, remaining):
        if self.next_targets() > remaining:
            return None
        starts = self.generator.integers(0, len(self.stream) - self.width + 1, size=self.rows)
        ids = np.stack([self.stream[s:s + self.width] for s in starts]).astype(np.int64)
        return {"input_ids": torch.from_numpy(ids).to(self.device)}


def take_step(batches, accumulate, remaining):
    if accumulate < 1:
        raise ValueError("accumulation count must be positive")
    records = []
    for _ in range(accumulate):
        record = batches.take(remaining)
        if record is None:
            break
        records.append(record)
        ids = record["input_ids"]
        weight = position_weight(record)
        remaining -= float(weight[:, :-1].sum()) if weight is not None else ids.numel() - ids.shape[0]
    return records


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(v) for v in value)
    return value


def parameter_layout(model):
    return [(n, list(p.shape), str(p.dtype), str(p.device), p.requires_grad)
            for n, p in model.named_parameters()]


def optimizer_layout(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def execution_config(config):
    if config is None:
        return None
    # save_pretrained fills these export hints without changing the live model.
    # Actual parameter dtypes and names are checked separately by parameter_layout.
    return {k: v for k, v in config.items() if k not in {"architectures", "dtype", "torch_dtype"}}


def model_config(model):
    config = getattr(model, "config", None)
    return execution_config(json.loads(json.dumps(config.to_dict(), default=str))) if config is not None else None


def save_training_state(path, model, optimizer, batches, progress, run_args):
    """An exclusive directory + final marker prevents overwrites/partial resumes."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    np_state = np.random.get_state()
    payload = dict(version=1, model=_cpu(model.state_dict()),
                   optimizer=_cpu(optimizer.state_dict()), layout=parameter_layout(model),
                   optimizer_layout=optimizer_layout(model, optimizer), config=model_config(model),
                   batches=batches.state_dict(), progress=progress, run_args=run_args,
                   torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                   numpy_rng=(np_state[0], np_state[1].tolist(), *np_state[2:]),
                   cuda_rng=[s.cpu() for s in torch.cuda.get_rng_state_all()]
                   if torch.cuda.is_available() else [])
    torch.save(payload, path / "state.pt")
    (path / "complete.json").write_text(json.dumps({"version": 1}), encoding="utf-8")


def read_training_state(path):
    path = Path(path)
    if not (path / "complete.json").is_file():
        raise ValueError("checkpoint is incomplete (missing completion marker)")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if state["version"] != 1:
        raise ValueError("unsupported training checkpoint version")
    return state


def restore_training_state(state, model, optimizer, batches):
    if state["layout"] != parameter_layout(model):
        raise ValueError("resume requires identical parameter names, dtype and device layout")
    if state["optimizer_layout"] != optimizer_layout(model, optimizer):
        raise ValueError("resume optimizer parameter order differs from checkpoint")
    if execution_config(state["config"]) != model_config(model):
        raise ValueError("resume model configuration differs from checkpoint")
    batches.load_state_dict(state["batches"])
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    name, keys, pos, gauss, cached = state["numpy_rng"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, gauss, cached))
    if state["cuda_rng"]:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA device count differs from checkpoint")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["progress"]
