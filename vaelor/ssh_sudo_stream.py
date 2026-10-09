"""How a worker command is elevated, and how a file lands on a worker (VD-164).

Two rules live here, both for `vaelor.ssh_transport.SshTransport`:

**The sudo password never shares a stream with a payload it might not be
consumed from.** ``sudo -S`` reads a password line only when it needs one. On a
worker whose sudo needs none (NOPASSWD, or a credential it still has cached)
the line it skipped used to reach the command instead - ``tee`` wrote the
password into root-owned files (ACC-166). So the worker's shell reads that line
itself, hands it alone to ``sudo -S -v``, and the command runs under ``sudo -n``
with only its own input (`elevated`).

**A file is written beside its target, checked, and only then renamed into
place** (`land_file`). A check that fails never deletes the good file already
there, and a write that fails never leaves a truncated one.
"""

from __future__ import annotations

import hashlib
import secrets
import shlex
import stat
from typing import Callable, List, Optional

#: stderr line: the command failed and sudo had not accepted the password.
REFUSED_MARKER = "vaelor-sudo-password-refused"
#: stderr line: sudo accepted the password but would not run the next command
#: without it again - the worker keeps no credential cache.
NO_CACHE_MARKER = "vaelor-sudo-keeps-no-cache"
MARKERS = (REFUSED_MARKER, NO_CACHE_MARKER)

#: What precedes the 16 hex characters that name the file a write lands in
#: before it is renamed into place: ``PATH.vaelor-new-<16 hex>``, unique per
#: write, so two writers never share a staging file.
STAGING_SUFFIX = ".vaelor-new-"
_STAGING_TOKEN_HEX = 16
_HEX = frozenset("0123456789abcdef")
#: No setuid, setgid or sticky bit, and no group or other write, ever survives
#: a mode copied from a target (a unit found 4755 or 0666 lands 0755 / 0644).
_MODE_CEILING = 0o755
#: A write whose target positively does not exist lands as ``tee`` would
#: create it; one whose target could not be read lands private, so a config
#: holding keys never becomes world-readable because a ``stat`` failed.
_NEW_FILE_MODE = "0644"
_UNKNOWN_MODE = "0600"
#: The only two endings of a failed ``stat`` that mean the target does not
#: exist, under ``LC_ALL=C``: GNU coreutils' (and busybox's), and uutils'
#: (Ubuntu 26.04), which adds the errno. Anything else is unknown.
_ABSENT_ENDINGS = (
    ": No such file or directory",
    ": No such file or directory (os error 2)",
)

REFUSED_MESSAGE = (
    "sudo on this machine did not run the command for this account. Either it "
    "did not accept the saved sign-in password, or its policy will not run "
    "commands over Vaelor's connection (for example one that requires a "
    "terminal). Check the password, that the account may use sudo, and that "
    "sudo does not require a terminal for it."
)
NO_CACHE_MESSAGE = (
    "This worker's sudo does not keep the password between commands "
    "(timestamp_timeout=0), so Vaelor cannot run root commands on it without "
    "mixing the password into the command's input. Allow a short credential "
    "cache for the Vaelor account, or give it passwordless sudo for Vaelor's "
    "commands."
)



def _error(message: str):
    from .ssh_transport import SshTransportError

    return SshTransportError(message)


def redacted(text: str, secret: Optional[str]) -> str:
    """``text`` with every copy of ``secret`` removed, for an error message.

    Anything a worker prints may echo what it was given - a file a polluted
    write left behind, a program that repeats its input - so no remote text
    reaches an exception with the sign-in password still in it.
    """
    if not secret:
        return text
    return text.replace(secret, "[password removed]")


def sudo_password(profile) -> Optional[str]:
    """The line to authenticate sudo with, or None when none is sent.

    A password with a line break in it cannot be sent as one line - its second
    half would be read as the start of the command's input - so it is refused
    before anything connects. The refusal does not quote it.
    """
    if not profile.get("sudo_uses_login_password", True):
        return None
    password = str(profile.get("password") or "")
    if any(character in password for character in "\r\n\0"):
        raise _error(
            "The saved password for this machine contains a line break, so it "
            "cannot be passed to sudo safely. Nothing was run."
        )
    return password


def elevated(command: str, password: Optional[str], c_locale: bool = False) -> str:
    """The command string that runs the quoted ``command`` as root.

    With a password, the worker's shell:

    1. reads exactly the first stdin line (``IFS= read -r``: no whitespace
       trimmed, no backslash processed, and a pipe is read one byte at a time,
       so not a byte of the payload goes with it). That line is ALWAYS
       consumed, whatever sudo will do;
    2. pipes that line alone with ``printf '%s\\n'`` - a builtin, so the
       password is on no argv, and ``%s`` so no byte of it is interpreted - to
       ``sudo -S -v``, which authenticates and caches the credential, or ignores
       the line when it needs no password;
    3. runs the command under ``sudo -n``, which never reads stdin, with only
       the payload left. If sudo still wants a password it refuses and runs
       nothing - fail closed.

    Only when the command failed does the shell say why sudo was involved, in
    Vaelor's own marker lines (sudo's own wording differs by version and
    language): `REFUSED_MARKER` when step 2 failed, `NO_CACHE_MARKER` when step
    2 passed but ``sudo -n -v`` is refused straight after it - a question
    about the credential, not about any command a policy may not allow. The
    probe runs only after a failure. Each marker is printed after a newline,
    so a command whose error text lacks one cannot swallow it. The shell ends
    with its own ``exit`` so it
    does not ``exec`` into step 3: sudo's credential cache is tied to the
    calling session and parent process, and every sudo here shares one parent.

    ``c_locale`` runs the command with ``LC_ALL=C`` (sudo passes ``LC_*``
    through by default), so a message the caller must recognise is in English.
    """
    sudo = "LC_ALL=C sudo" if c_locale else "sudo"
    if password is None:
        return f"{sudo} -n -- {command}"
    script = (
        "IFS= read -r vaelor_pw; "
        "printf '%s\\n' \"$vaelor_pw\" 2>/dev/null"  # absence-ok: only printf's broken-pipe complaint when sudo needs no password; the outcome is kept in vaelor_ok
        " | sudo -S -p '' -v >/dev/null 2>&1; "
        "vaelor_ok=$?; vaelor_pw=; "
        f"{sudo} -n -- {command}; vaelor_rc=$?; "
        "if [ \"$vaelor_rc\" -ne 0 ]; then "
        f"if [ \"$vaelor_ok\" -ne 0 ]; then printf '\\n%s\\n' {REFUSED_MARKER} >&2; "
        "elif ! sudo -n -v >/dev/null 2>&1; then "
        f"printf '\\n%s\\n' {NO_CACHE_MARKER} >&2; fi; fi; "
        "exit \"$vaelor_rc\""
    )
    return "sh -c " + shlex.quote(script)


def failure(error_text: str, password: Optional[str]):
    """The transport error for a failed command's stderr, markers resolved.

    A failure that wrote no text of its own - which includes a connection that
    closed under the command - carries the transport's one named sentence,
    `ssh_transport.COMMAND_ENDED_WITHOUT_REASON`, so a caller can still tell it
    from a machine that refused and said why (a worker restart).
    """
    from .ssh_transport import COMMAND_ENDED_WITHOUT_REASON

    lines = [line for line in error_text.splitlines() if line.strip()]
    detail = redacted(
        "\n".join(line for line in lines if line not in MARKERS), password,
    )[:400]
    said = f" The machine said: {detail}" if detail else ""
    if NO_CACHE_MARKER in lines:
        return _error(NO_CACHE_MESSAGE + said)
    if REFUSED_MARKER in lines:
        return _error(REFUSED_MESSAGE + said)
    return _error(detail or COMMAND_ENDED_WITHOUT_REASON)


def tee_target(arguments: List[str]) -> Optional[str]:
    """The one file a ``tee`` writes, or None for any other command.

    ``tee PATH`` is the only write shape callers use and the only one
    `land_file` can stage and check, so any other ``tee`` is refused.
    """
    if arguments[0] != "tee":
        return None
    if len(arguments) != 2 or not arguments[1] or arguments[1].startswith("-"):
        raise _error("Only a write of one file with tee is permitted.")
    return arguments[1]


def check_rename(arguments: List[str]) -> None:
    """Refuse every ``mv`` but the one `land_file` issues.

    ``mv -f PATH.vaelor-new-<16 lowercase hex> PATH``: the source is the
    target plus a staging name, so a rename can only put a checked file in
    place of its own target.
    """
    if arguments[0] != "mv":
        return
    target = arguments[3] if len(arguments) == 4 else ""
    source = arguments[2] if len(arguments) == 4 else ""
    token = source[len(target) + len(STAGING_SUFFIX):]
    if (
        len(arguments) != 4
        or arguments[1] != "-f"
        or not target
        or target.startswith("-")
        or target.endswith("/")
        or source[:len(target) + len(STAGING_SUFFIX)] != target + STAGING_SUFFIX
        or len(token) != _STAGING_TOKEN_HEX
        or not set(token) <= _HEX
    ):
        raise _error("Only moving a checked file into place is permitted.")


def _says_absent(message: str) -> bool:
    """Whether a failed ``stat`` said, exactly, that the path does not exist.

    Anchored to the end of the whole error, and only in the two known forms.
    """
    return message.rstrip().endswith(_ABSENT_ENDINGS)


def _target_mode(execute: Callable[..., str], path: str) -> str:
    """The mode a write of ``path`` lands with, or a refusal.

    ``stat -c '%f %a'`` (under ``LC_ALL=C``) prints the raw mode in hex and the
    permissions in octal, and does not follow a link. The file type is decoded
    from the raw mode here (``stat.S_ISREG`` and friends), never from the
    translated name ``%F`` prints, so a worker's language cannot refuse every
    write.

    - A regular file keeps its permissions, masked by `_MODE_CEILING`.
    - A link is REFUSED, never replaced: in ``/etc/systemd/system`` it is how
      an administrator masks a unit (a link to ``/dev/null``); renaming over it
      would silently unmask the unit, and copying its 777 would make a root
      unit world-writable. Any other non-regular target is refused too.
    - A target stat positively reports missing lands 0644, what ``tee`` itself
      created. ANY other failure, or an answer that cannot be read, lands 0600:
      a gate or telemetry config holding keys is never widened by a stat that
      failed for some other reason.
    """
    try:
        answer = execute(["stat", "-c", "%f %a", path], c_locale=True).split()
    except Exception as error:
        return _NEW_FILE_MODE if _says_absent(str(error)) else _UNKNOWN_MODE
    try:
        raw = int(answer[0], 16)
        permissions = int(answer[1], 8)
    except (IndexError, ValueError):
        return _UNKNOWN_MODE
    if stat.S_ISLNK(raw):
        raise _error(
            f"{path} on this machine is a symbolic link, so the unit looks "
            "masked or redirected. Vaelor did not replace it; resolve it (for "
            "example unmask the unit) and try again."
        )
    if not stat.S_ISREG(raw):
        what = "a directory" if stat.S_ISDIR(raw) else "neither a file nor a link"
        raise _error(
            f"{path} on this machine is {what}, so Vaelor did not replace it."
        )
    return "0{:o}".format(permissions & _MODE_CEILING)


def land_file(
    execute: Callable[..., str], path: str, text: str,
) -> str:
    """Write ``text`` to ``path`` so that only a proven copy ever replaces it.

    ``execute(argv, stdin_text="", write=False, c_locale=False)`` runs one
    argv on the worker with the caller's privilege and raises the transport
    error on failure.

    1. Decide the mode from what the target is now (`_target_mode`): a link
       or other non-regular target is refused before anything is written, and
       a target that could not be read makes the write land private (0600).
    2. ``install -m 0600 /dev/null PATH.vaelor-new-<16 hex>``: this write's
       own staging file, unique, starts empty and private.
    3. ``tee`` the text into it.
    4. ``sha256sum`` it against the digest of what was sent.
    5. Only on a match: ``chmod`` it to the mode from step 1, then ``mv -f``
       it onto the target - a rename, so the target changes in one step.

    A mismatch removes this write's staging file and says the file did not
    match; a write or check that could not run removes it and says so. Either
    way the target is left exactly as it was. A staging file left by a crash
    is an orphan under its own name: never renamed, never glob-deleted. The new file is owned by the elevating
    user (root under sudo), whatever owned the old one. The digest is compared,
    never the text, so nothing secret reaches an error; ``""`` is returned
    rather than ``tee``'s echo of the content.
    """
    mode = _target_mode(execute, path)
    stage = path + STAGING_SUFFIX + secrets.token_hex(_STAGING_TOKEN_HEX // 2)
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    staged = []

    def discard(message: str):
        if not staged:
            return _error(message)
        try:
            execute(["rm", "-f", "--", stage])
        except Exception:
            message += f" A leftover {stage} may remain on the machine."
        return _error(message)

    def unchecked(cause=None) -> str:
        because = f" {cause}" if cause else ""
        return (
            f"The file {path} on this machine could not be written or checked, "
            f"so it was left as it was.{because}"
        )

    try:
        execute(["install", "-m", "0600", "/dev/null", stage])
        staged.append(stage)
        execute(["tee", stage], stdin_text=text, write=True)
        listing = execute(["sha256sum", "--", stage]).split()
    except Exception as cause:
        raise discard(unchecked(cause)) from None
    if not listing:
        raise discard(unchecked())
    if listing[0] != expected:
        raise discard(
            f"The file written for {path} on this machine did not match what "
            "Vaelor sent, so it was discarded and the file was left as it was."
        )
    try:
        execute(["chmod", mode, stage])
        execute(["mv", "-f", stage, path])
    except Exception as cause:
        raise discard(unchecked(cause)) from None
    return ""
