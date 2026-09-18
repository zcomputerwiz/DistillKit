"""Make ``importlib.metadata.version("triton")`` answer on Windows.

Cut Cross-Entropy gates a kernel on ``is_triton_3_2()``, which reads the installed
version through package metadata. The Windows build of Triton is distributed as
``triton-windows``, so that lookup raises ``PackageNotFoundError`` and CCE fails at its
first call rather than falling back. Importing this module resolves the alias.

Import it before ``cut_cross_entropy``.
"""

from __future__ import annotations

import importlib.metadata as metadata

_original = metadata.version


def _version(name):
    try:
        return _original(name)
    except metadata.PackageNotFoundError:
        if name == "triton":
            return _original("triton-windows")
        raise


if metadata.version is _original:
    metadata.version = _version
