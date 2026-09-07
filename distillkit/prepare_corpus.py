"""Render a chat SFT dataset to the JSONL that ``sample-transformers`` captures from.

Written for r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation, whose ``sft*`` and
``openai_messages`` configs carry OpenAI-style ``messages`` (role/content, with any
``<think>`` block already inside the assistant ``content`` -- the separate
``reasoning_content`` field is empty in this release, so nothing needs merging).

Two choices here are deliberate and worth stating, because both are easy to get
subtly wrong and neither is recoverable after a multi-hour capture:

* **The teacher's own chat template is applied, not the dataset's pre-rendered text.**
  The dataset also ships ``prompt_completion_text`` (generic ``<|system|>`` markers)
  and ``glm47_native`` (pre-tokenized with GLM's tokenizer). Both would put the 27B
  teacher off-distribution: we are recording *its* next-token distribution, so the
  text must be framed the way it expects.
* **Documents are emitted whole, one per row, unpadded.** ``capture_teacher`` requires
  unpadded, unpacked documents and truncates to ``sequence_length`` itself.

The dataset's own ``split`` column is preserved so capture keeps its eval holdout
aligned with the upstream split rather than inventing one. Note the upstream
validation/test splits are documented as benchmark-derived and contaminated -- fine as
a held-out loss signal, not usable as a capability claim.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

LOG = logging.getLogger(__name__)

__all__ = ["render_messages", "iter_dataset_documents"]


def render_messages(messages, tokenizer) -> str:
    """Apply the teacher's chat template to role/content pairs."""
    cleaned = []
    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        if not role:
            continue
        # If a future release splits reasoning out of content, fold it back in rather
        # than silently dropping the trace the run is meant to learn from.
        if reasoning and reasoning not in content:
            content = f"<think>\n{reasoning}\n</think>\n{content}"
        if not content.strip():
            continue
        cleaned.append({"role": role, "content": content})
    if not cleaned:
        return ""
    return tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=False)


def iter_dataset_documents(
    repo: str,
    config: str,
    tokenizer,
    *,
    split: str | None = None,
    limit: int | None = None,
    min_tokens: int = 16,
):
    """Yield ``{doc_id, text, split}`` records from a Hugging Face chat dataset.

    Reads the config's parquet shards directly rather than going through
    ``load_dataset``: this repo publishes many configs as sibling directories under
    ``data/``, and the loader tries to unify their (different) schemas into one, which
    fails. Streaming the shards keeps memory flat and skips that entirely.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    prefix = f"datasets/{repo}/data/{config}"
    shards = sorted(f for f in fs.ls(prefix, detail=False) if f.endswith(".parquet"))
    if not shards:
        raise ValueError(f"No parquet shards under {prefix}")
    # Shard filenames sort "test-*" before "train-*", so an unfiltered --limit run
    # silently draws its whole sample from the held-out split. Read the requested
    # split's shards first so a truncated run is still representative.
    if split is not None:
        preferred = [f for f in shards if f"/{split}-" in f.replace("\\", "/")]
        shards = preferred + [f for f in shards if f not in preferred]

    emitted = 0
    for shard in shards:
        with fs.open(shard, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for group in range(parquet.num_row_groups):
                for row in parquet.read_row_group(group).to_pylist():
                    row_split = row.get("split") or "train"
                    if split is not None and row_split != split:
                        continue
                    text = render_messages(row.get("messages") or [], tokenizer)
                    if not text or len(tokenizer(text)["input_ids"]) < min_tokens:
                        continue
                    yield {
                        "doc_id": str(row.get("id") or row.get("parent_id") or emitted),
                        "text": text,
                        # capture_teacher honours an explicit split and only falls back
                        # to its every-Nth heuristic when one is absent.
                        "split": "eval" if row_split in ("validation", "test") else "train",
                        "source": row.get("source"),
                        "domain": row.get("domain"),
                    }
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return


@click.command("prepare-corpus")
@click.option("--repo", default="r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation", show_default=True)
@click.option("--config", default="sft_balanced", show_default=True)
@click.option("--tokenizer", "tokenizer_path", required=True, help="Teacher checkpoint or tokenizer dir.")
@click.option("--output", type=click.Path(dir_okay=False), required=True)
@click.option("--split", default=None, help="Keep only this upstream split (train/validation/test).")
@click.option("--limit", type=int, default=None, help="Stop after this many documents.")
@click.option("--max-tokens", type=int, default=None, help="Stop once this many tokens are written.")
def main(repo, config, tokenizer_path, output, split, limit, max_tokens):
    """Render a chat dataset to capture-ready JSONL."""
    logging.basicConfig(level=logging.INFO)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = docs = 0
    with out.open("w", encoding="utf-8") as handle:
        for record in iter_dataset_documents(repo, config, tok, split=split, limit=limit):
            n = len(tok(record["text"])["input_ids"])
            if max_tokens is not None and total_tokens + n > max_tokens:
                break
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_tokens += n
            docs += 1
            if docs % 500 == 0:
                LOG.info("rendered %d documents, %d tokens", docs, total_tokens)

    click.echo(f"wrote {docs} documents / {total_tokens:,} tokens to {out}")
    click.echo(f"cache estimate at 2 anchors: {total_tokens * 10.4 / 1e6:.2f} GB")


if __name__ == "__main__":
    main()
