"""One place where bytes become a number with a unit on it.

A live tester on the Pi captured four figures for one volume in a single pass:

===============  ==========================================================
Sidebar          ``microSD · / | 8% used · 220.7 GB free``
Assistant        ``The memory card has 221.1 GB free of 251.7 GB.``
Home card        ``20.7 GB used`` / ``251.7 GB total``
API              ``free = 231,341,775,872``
===============  ==========================================================

Two separate defects are tangled in that table, and only one of them is
arithmetic.

**The arithmetic one.** Several conversions in this tree divided by 1024³ and
wrote ``GB``, or divided by 1024² and wrote ``MB``. That is a 7.4% and a 4.9%
understatement respectively - too small to look wrong and too large to be
right, which is how each of them survived review. The fix is not a new formula;
it is that the divisor and the suffix are now chosen together, here, and cannot
be paired up by hand at each call site.

**The one that is not arithmetic.** ``free`` and ``total - used`` are different
quantities on Linux and both are correct. ``statvfs`` reports ``f_bavail``, what
an unprivileged process may still write, and ``f_bfree``, which additionally
counts the filesystem's reserved blocks. On the tester's card those differ by
about 10 GB, and the product was showing one of them beside the other with
nothing saying which was which. This module cannot fix that - it is a question
about which field to read, answered in :mod:`vaelor.linux_storage` - but it is
recorded here because the two defects produce the same symptom and were
initially read as one.

No rounding happens until the last moment, and the unit is never inferred from
the magnitude: a figure that silently changes suffix between samples is a
figure two surfaces will disagree about.
"""

from __future__ import annotations

from typing import Any, Optional


#: Decimal units. What disk vendors, ``df``, and every user-facing capacity
#: figure in this product mean by KB/MB/GB/TB.
KB = 1000
MB = 1000 ** 2
GB = 1000 ** 3
TB = 1000 ** 4

#: Binary units. What memory, VRAM, and anything reported by the kernel or by
#: ``amd-smi`` is measured in. These carry the ``i`` in their names for the
#: same reason they carry the different divisor.
KIB = 1024
MIB = 1024 ** 2
GIB = 1024 ** 3
TIB = 1024 ** 4


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def gigabytes(value: Any) -> Optional[float]:
    """Decimal GB, or ``None`` for anything that is not a byte count."""
    number = _number(value)
    return None if number is None else number / GB


def gibibytes(value: Any) -> Optional[float]:
    """Binary GiB, or ``None``."""
    number = _number(value)
    return None if number is None else number / GIB


def mebibytes(value: Any) -> Optional[float]:
    """Binary MiB, or ``None``."""
    number = _number(value)
    return None if number is None else number / MIB


def describe_gb(value: Any, decimals: int = 1) -> str:
    """``"231.3 GB"``. Decimal divisor, decimal suffix, together."""
    number = gigabytes(value)
    return "" if number is None else "{:.{}f} GB".format(number, decimals)


def describe_gib(value: Any, decimals: int = 1) -> str:
    """``"215.5 GiB"``."""
    number = gibibytes(value)
    return "" if number is None else "{:.{}f} GiB".format(number, decimals)


def describe_mib(value: Any, decimals: int = 0) -> str:
    """``"5903 MiB"``. The unit ``amd-smi`` and the kernel actually report."""
    number = mebibytes(value)
    return "" if number is None else "{:.{}f} MiB".format(number, decimals)
