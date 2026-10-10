"""How the household authority touches the host: systemd-creds and file ownership (VD-212).

Split from :mod:`vaelor.tls_authority` to keep it under the module ceiling;
:mod:`vaelor.tls_authority` re-exports every name here.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Callable, Optional

CREDENTIAL_NAME = "vaelor-household-root"
SERVICE_GROUP = "vaelor"
_COMMAND_TIMEOUT_SECONDS = 60

Runner = Callable[..., "subprocess.CompletedProcess"]


class AuthorityError(RuntimeError):
    """A failure the owner must see. Its text never carries key material."""


def _scrubbed(stderr: Any) -> str:
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr or "")
    text = " ".join(text.split())[:300]
    return "[output withheld]" if "PRIVATE KEY" in text else text


class SystemdCreds:
    """Seal and unseal the root key with ``systemd-creds --with-key=host``.

    The key travels over pipes only: never argv, never a temporary file. A
    failure raises :class:`AuthorityError` built from the tool's stderr alone,
    never a ``CalledProcessError`` (whose text would carry stdout - the key).
    """

    def __init__(self, run: Runner = subprocess.run):
        self._run = run

    def _call(self, verb: str, payload: bytes) -> bytes:
        argv = ["systemd-creds", verb, "--name=" + CREDENTIAL_NAME]
        if verb == "encrypt":
            argv.append("--with-key=host")
        argv += ["-", "-"]
        try:
            result = self._run(argv, input=payload, capture_output=True,
                               timeout=_COMMAND_TIMEOUT_SECONDS, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            raise AuthorityError("systemd-creds {} could not run ({}).".format(
                verb, type(error).__name__)) from None
        if result.returncode != 0 or not result.stdout:
            raise AuthorityError("systemd-creds {} failed (exit {}): {}".format(
                verb, result.returncode, _scrubbed(result.stderr)))
        return result.stdout

    def seal(self, plaintext: bytes) -> bytes:
        return self._call("encrypt", plaintext)

    def unseal(self, sealed: bytes) -> bytes:
        return self._call("decrypt", sealed)


def _service_gid() -> Optional[int]:
    try:
        import grp
    except ImportError:
        return None
    try:
        return grp.getgrnam(SERVICE_GROUP).gr_gid
    except KeyError:
        raise AuthorityError("The '{}' group does not exist; the console could not "
                             "read its certificate.".format(SERVICE_GROUP)) from None


def default_chown(path: str, service_group: bool, service_user: bool = False) -> None:
    """root:root, root:vaelor for what the console must read, or vaelor:vaelor
    for the folder the control plane itself writes in."""
    if os.name != "posix":
        return
    uid = 0
    if service_user:
        import pwd

        try:
            uid = pwd.getpwnam(SERVICE_GROUP).pw_uid
        except KeyError:
            raise AuthorityError("The '{}' account does not exist.".format(
                SERVICE_GROUP)) from None
    os.chown(path, uid, _service_gid() if service_group else 0)
