"""Optional compatibility bridge to SunFounder Raspberry Pi status helpers.

Host power deliberately does **not** live here any more. This module used to
answer "can this machine reboot?" with an import check for a Raspberry Pi
package, so every x86 host got a function written to raise — a refusal that
described a missing SunFounder helper rather than any real limitation of the
machine. That decision now belongs to the platform driver
(``vaelor/platforms/*.power_actions()``), which reaches systemd-logind on a
generic host and SunFounder's sequenced helpers on a Pironman.
"""

from __future__ import annotations


try:
    from sf_rpi_status import get_disks, get_ips
    AVAILABLE = True
except ImportError:  # Generic Linux and amd64 installations
    AVAILABLE = False

    def get_disks():
        return []

    def get_ips():
        return {}
