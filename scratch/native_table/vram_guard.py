"""Stop a measurement the instant it stops fitting on the card.

Windows does not raise ``OutOfMemoryError`` when a run outgrows VRAM. WDDM satisfies the
allocation out of shared system memory instead, so the step still completes -- measured
here at 100.5 s against 1.38 s for work that fits -- while host RAM climbs until something
else on the machine suffers. A script that only catches ``OutOfMemoryError`` will happily
spend twenty minutes timing a configuration nobody can use, and the first person to notice
is whoever is using the desktop.

Two layers, because neither is sufficient alone:

``cap`` caps the caching allocator with ``set_per_process_memory_fraction``, which turns an
oversized allocation into an ordinary ``OutOfMemoryError`` before the driver is ever asked
to page anything. That is the real fix.

``spill_check`` is called after every step and catches what the cap cannot: memory taken
outside the caching allocator (cuBLAS and cuDNN workspaces), and host RAM growing, which is
what a spill looks like from outside the process.
"""

from __future__ import annotations

import psutil
import torch

__all__ = ["Spilled", "cap", "spill_check", "baseline", "HOST_RAM_LIMIT"]

# A spilling step on this machine grew host RAM by tens of gibibytes. Anything past this is
# not measurement noise.
HOST_RAM_LIMIT = 4 * 2**30


class Spilled(RuntimeError):
    """The configuration left the card: either over budget, or paging to host RAM."""


def cap(device, budget: float) -> tuple[int, int]:
    """Hard-limit this process to ``budget`` of the card. Returns (index, capacity)."""
    index = torch.device(device).index or 0
    _, capacity = torch.cuda.mem_get_info(index)
    torch.cuda.set_per_process_memory_fraction(budget, index)
    return index, capacity


def baseline() -> int:
    return psutil.Process().memory_info().rss


def spill_check(budget: float, capacity: int, baseline_rss: int, when: str) -> None:
    """Raise ``Spilled`` if VRAM is over budget or host RAM has started climbing."""
    reserved = torch.cuda.memory_reserved()
    if reserved > budget * capacity:
        raise Spilled("%s: reserved %.1f GiB of a %.1f GiB card"
                      % (when, reserved / 2**30, capacity / 2**30))
    growth = psutil.Process().memory_info().rss - baseline_rss
    if growth > HOST_RAM_LIMIT:
        raise Spilled("%s: host RAM grew %.1f GiB, which is what a spill into shared "
                      "system memory looks like" % (when, growth / 2**30))


def _self_check() -> None:
    """Both limbs of the guard must actually raise: python scratch/native_table/vram_guard.py"""
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")                      # so something is reserved
        try:
            spill_check(0.0, torch.cuda.mem_get_info(0)[1], baseline(), "fake")
        except Spilled as error:
            assert "reserved" in str(error), error
        else:
            raise AssertionError("the VRAM limb did not fire")

    try:
        spill_check(0.85, 1 << 60, baseline() - 5 * 2**30, "fake")
    except Spilled as error:
        assert "host RAM grew" in str(error), error
    else:
        raise AssertionError("the host RAM limb did not fire")
    print("vram_guard: both limbs fire")


if __name__ == "__main__":
    _self_check()
