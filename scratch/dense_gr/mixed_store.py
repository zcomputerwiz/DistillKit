"""Interleave two token stores so the calibration sees both distributions.

Calibrating on Python left the conversion 8.6 MMLU points below the source; calibrating on
chat at the same 32,768 tokens left it 2.9 points below, an interval spanning zero. But
the two corpora fixed different things: chat moved MMLU and left NLL at 1.4823 to four
figures, while eight times the chat calibration moved NLL to 1.4546 and did not help MMLU.

Neither is the distribution the model actually serves, which contains both. This writes a
store that alternates blocks from each, so a run of windows drawn from anywhere in the
stream sees the mixture rather than a run of one and then a run of the other.

The block is a window rather than a token: the conversion draws contiguous 1024-token
windows, and interleaving at finer grain would put a boundary inside most of them, which
calibrates the model on a transition that never occurs at inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CHAT = Path("scratch/dense_gr/chat-tokens")
CODE = Path("scratch/code_training/tokens-v2")


def interleave(sources, weights, block, total, destination):
    """Write `total` tokens, drawing `weights`-proportioned runs of `block` from each."""
    streams = [np.memmap(path, dtype=np.uint32, mode="r") for path in sources]
    offsets = [0] * len(streams)
    share = np.array(weights, dtype=np.float64)
    share = share / share.sum()
    # How many blocks of each per cycle, at the coarsest whole-block ratio that holds.
    per_cycle = np.maximum(1, np.round(share * len(share) * 4)).astype(int)
    written = 0
    with open(destination, "wb") as handle:
        while written < total:
            for index, stream in enumerate(streams):
                for _ in range(per_cycle[index]):
                    if written >= total:
                        break
                    start = offsets[index]
                    if start + block > len(stream):
                        start = offsets[index] = 0
                    handle.write(np.asarray(stream[start:start + block],
                                            dtype=np.uint32).tobytes())
                    offsets[index] = start + block
                    written += block
    for stream in streams:
        del stream
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat", type=Path, default=CHAT)
    parser.add_argument("--code", type=Path, default=CODE)
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/dense_gr/mixed-tokens"))
    parser.add_argument("--chat-share", type=float, default=0.5)
    parser.add_argument("--block", type=int, default=1024,
                        help="tokens per run from one source; the conversion's window")
    parser.add_argument("--train-tokens", type=int, default=4_000_000)
    parser.add_argument("--heldout-tokens", type=int, default=250_000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    weights = [args.chat_share, 1.0 - args.chat_share]
    train = interleave([args.chat / "train.bin", args.code / "train.bin"],
                       weights, args.block, args.train_tokens,
                       args.output / "train.bin")
    # The conversion scores its own held-out loss on `calibration`, which is the opposite
    # of what the name suggests. Mixed the same way, so that number means the mixture too.
    held = interleave([args.chat / "calibration.bin", args.code / "calibration.bin"],
                      weights, args.block, args.heldout_tokens,
                      args.output / "calibration.bin")
    print("wrote %s: train %d tokens, calibration %d tokens, chat share %.2f"
          % (args.output, train, held, args.chat_share))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
