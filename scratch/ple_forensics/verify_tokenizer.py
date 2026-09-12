"""Does the hardcoded n-gram hash configuration actually describe Flash-Next?

`NGramHashConfig` carries the whole addressing scheme as *defaults* -- vocab_size,
eos_token_id, ngram_size, the per-head vocabulary base, the seed, the PLE layer index.
`from_pretrained_config` exists to build it from a real Flash-Next config, and nothing
in the project calls it: `SidecarDataCollator` and `analyze_donor_reader` both construct
a bare `NGramHasher()`. So every row address ever computed here rests on those defaults
being right.

They also rest on one further assumption nobody has checked: that the *student's*
tokenizer agrees with Flash-Next's. The rows are addressed by the student's token IDs,
because that is what the collator hashes. If the two tokenizers assign different IDs to
the same text, every address is a correct hash of the wrong number, silently, and every
result in this project that depends on row identity is meaningless -- the novelty
stratification, the frequency bins, the shuffled control's matched hit rate.

This is a verification, not an analysis. It compares the defaults against the real
config and the two tokenizers against each other, and names the specific token IDs the
layout/content split depends on.

    python scratch/ple_forensics/verify_tokenizer.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FLASH_NEXT = Path("C:/Users/Owner/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next")
STUDENT = Path("D:/DeepThought/Projects/HybridModel/student-hf")
CACHE = Path("D:/DeepThought/Projects/HybridModel/teacher-cache-1m/manifest.json")

#: The IDs the layout/content split and the hasher's EOS handling are written against.
NAMED_IDS = {198: "\\n", 248044: "eos/pad", 248045: "<|im_start|>",
             248046: "<|im_end|>", 248068: "<think>", 248069: "</think>"}


def snapshot(root):
    matches = sorted(root.glob("snapshots/*/config.json"))
    if not matches:
        raise FileNotFoundError("no config.json under %s" % root)
    return matches[0].parent


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_hash_config(text_config):
    """Every field `NGramHashConfig` hardcodes, against the donor's own config.

    A field absent from `config.json` is not unset: it takes `Qwen4ExpTextConfig`'s own
    default, which is what the reference module would use. `seed` is exactly that case --
    Flash-Next omits it and the class supplies 1234 -- so reading the raw JSON alone
    would report a mismatch against a value that actually agrees.
    """
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    from distillkit.ngram_hash import NGramHashConfig

    reference = Qwen4ExpTextConfig()
    defaults = NGramHashConfig()

    def declared(field):
        return text_config.get(field, getattr(reference, field, None))

    eos = declared("eos_token_id")
    eos = eos[0] if isinstance(eos, (list, tuple)) else eos
    expected = {
        "vocab_size": declared("vocab_size"),
        "ngram_size": declared("ngram_size"),
        "heads_per_ngram": declared("heads_per_ngram"),
        "ngram_vocab_size_base": declared("ngram_vocab_size_base"),
        "make_ngram_vocab_size_divisible_by": declared("make_ngram_vocab_size_divisible_by"),
        "seed": declared("seed"),
        "ple_embed_dim": declared("ple_embed_dim"),
        "eos_token_id": eos,
    }
    print("hash configuration: hardcoded default against Flash-Next's own config")
    problems = []
    for field, donor in expected.items():
        ours = getattr(defaults, field)
        agree = ours == donor
        problems += [] if agree else [field]
        print("  %-36s %-12s %-12s %s"
              % (field, ours, donor, "ok" if agree else "MISMATCH"))
    # ple_layer_index is the position within ple_layer_ids, not a decoder index; getting
    # it wrong shifts every head offset and every multiplier.
    layers = text_config.get("ple_layer_ids")
    print("  %-36s %-12s %-12s %s"
          % ("ple_layer_index (position in list)", defaults.ple_layer_index, layers,
             "ok" if layers and defaults.ple_layer_index < len(layers) else "CHECK"))
    return problems


def check_tokenizers(donor_path, student_path):
    """Same vocabulary, and the same IDs for the tokens this project names."""
    donor = load_json(donor_path)
    student = load_json(student_path)
    problems = []

    def vocab_of(blob):
        return blob["model"]["vocab"]

    donor_vocab, student_vocab = vocab_of(donor), vocab_of(student)
    if isinstance(donor_vocab, list):          # some tokenizer.json use pair lists
        donor_vocab = {entry[0]: index for index, entry in enumerate(donor_vocab)}
    if isinstance(student_vocab, list):
        student_vocab = {entry[0]: index for index, entry in enumerate(student_vocab)}

    print("\ntokenizer vocabularies")
    print("  donor entries   %d" % len(donor_vocab))
    print("  student entries %d" % len(student_vocab))
    if len(donor_vocab) != len(student_vocab):
        problems.append("vocabulary size")

    def digest(vocab):
        packed = "\n".join("%s\t%d" % (token, index)
                           for token, index in sorted(vocab.items(), key=lambda kv: kv[1]))
        return hashlib.sha256(packed.encode("utf-8")).hexdigest()

    donor_digest, student_digest = digest(donor_vocab), digest(student_vocab)
    print("  donor sha256    %s" % donor_digest)
    print("  student sha256  %s" % student_digest)
    identical = donor_digest == student_digest
    print("  %s" % ("identical vocabularies" if identical
                    else "VOCABULARIES DIFFER -- see the per-ID check below"))
    if not identical:
        problems.append("vocabulary content")

    # Added tokens live outside model.vocab in tokenizer.json, so resolve names both ways.
    def id_to_token(blob, vocab):
        table = {index: token for token, index in vocab.items()}
        for entry in blob.get("added_tokens", []):
            table[entry["id"]] = entry["content"]
        return table

    donor_ids = id_to_token(donor, donor_vocab)
    student_ids = id_to_token(student, student_vocab)
    print("\n  the IDs this project names")
    print("  %-8s %-14s %-24s %-24s" % ("id", "meaning", "donor", "student"))
    for token_id, meaning in NAMED_IDS.items():
        left = donor_ids.get(token_id, "<absent>")
        right = student_ids.get(token_id, "<absent>")
        agree = left == right
        problems += [] if agree else ["id %d" % token_id]
        # Byte-level BPE spells newline as U+010A and space as U+0120, neither of which
        # a cp1252 console can encode; the escaped form is also the unambiguous one.
        print("  %-8d %-14s %-24s %-24s %s"
              % (token_id, meaning, ascii(left), ascii(right),
                 "ok" if agree else "MISMATCH"))
    return problems


def main():
    donor_dir = snapshot(FLASH_NEXT)
    config = load_json(donor_dir / "config.json")
    text_config = config.get("text_config", config)
    print("Flash-Next snapshot: %s" % donor_dir)
    print("architecture: %s\n" % config.get("architectures"))

    problems = check_hash_config(text_config)
    problems += check_tokenizers(donor_dir / "tokenizer.json", STUDENT / "tokenizer.json")

    if CACHE.exists():
        manifest = load_json(CACHE)
        print("\nteacher cache")
        print("  vocab_size        %s" % manifest.get("vocab_size"))
        print("  tokenizer_hash    %s" % manifest.get("tokenizer_hash"))
        if manifest.get("vocab_size") != text_config.get("vocab_size"):
            problems.append("cache vocab_size")

    print("\n%s" % ("VERIFIED: every addressing assumption holds" if not problems
                    else "PROBLEMS: %s" % ", ".join(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
