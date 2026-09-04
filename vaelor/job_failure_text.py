"""Translate operating-system failure text into something a user can act on.

Vaelor promises that no terminal is required, and then handed the two most
common Docker and systemd failures to the user verbatim, including the
``journalctl`` invocation the distribution suggests:

    Cannot connect to the Docker daemon at unix:///var/run/docker.sock.
    Is the docker daemon running?

    Job for gnome-remote-desktop.service failed because the control process
    exited with error code. See 'systemctl status ...' and
    'journalctl -xeu ...' for details.

The frontend already rewrites executor exceptions for display
(``frontend/src/lib/userFacingErrors.ts``), and this module deliberately
mirrors its vocabulary - cause, recovery, and a "technical" original - but it
runs where the text originates, so the stored job message, the operation
projection, the activity feed, and any future surface all agree.

Nothing is discarded. The original text is returned as ``technical`` for the
caller to keep in the job result and in the log.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# Ordered most specific first. Every recovery names a Vaelor screen; none of
# them names a command, a unit file, a socket path, or a log tool.
_RULES: Tuple[Tuple[re.Pattern[str], str, str, str], ...] = (
    (
        re.compile(
            r"permission denied while trying to connect to the docker|"
            r"(?:got\s+)?permission denied.{0,80}docker\.sock|"
            r"dial unix\s+\S*docker\.sock.{0,40}permission denied",
            re.IGNORECASE | re.DOTALL,
        ),
        "Vaelor is not allowed to use the container engine on this appliance.",
        "Docker is installed, but the Vaelor service account has not been "
        "granted access to it.",
        "Open System > Hardware & services and run the Docker readiness check, "
        "then retry this operation.",
    ),
    (
        re.compile(
            r"cannot connect to the docker daemon|"
            r"is the docker daemon running|"
            r"docker daemon is not running|"
            r"error during connect.{0,80}docker|"
            r"dial unix\s+\S*docker\.sock",
            re.IGNORECASE | re.DOTALL,
        ),
        "Vaelor could not reach the container engine on this appliance.",
        "Docker is installed, but its background service is not running.",
        "Open System > Hardware & services, start Docker, and retry this "
        "operation once it reports ready.",
    ),
    (
        re.compile(
            r"\bjob for \S+\.(?:service|socket|timer) failed|"
            r"\bfailed to start\b|"
            r"\bunit \S+\.(?:service|socket|timer) (?:has )?(?:failed|entered failed state)|"
            r"\bjournalctl\b|"
            r"\bsystemctl status\b|"
            r"\bactive: failed\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "A background service this operation needs did not start.",
        "The service stopped immediately after Vaelor started it, so the "
        "operation could not continue.",
        "Open System > Hardware & services to check service health, then retry. "
        "If it keeps stopping, restore a checkpoint or reinstall the affected "
        "feature from the same screen.",
    ),
)


def user_facing_failure(message: str) -> Optional[Dict[str, str]]:
    """Return plain-language replacement text, or None to leave ``message`` alone.

    Only messages that carry operating-system mechanics are rewritten. Failure
    text Vaelor already writes for people ("Choose an available port") must
    reach the user unchanged.
    """
    raw = " ".join(str(message or "").split())
    if not raw:
        return None
    for pattern, summary, cause, recovery in _RULES:
        if pattern.search(raw):
            return {
                "summary": summary,
                "cause": cause,
                "recovery": recovery,
                "technical": raw[:2000],
            }
    return None
