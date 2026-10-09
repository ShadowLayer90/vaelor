"""What the root bridge may run on the controller, shape by shape.

VD-125, narrowed by VD-143. The controller serves GPU models as an ordinary
cluster participant, so some of its commands leave `GpuPoolRuntime` through
`vaelor.bridge_transport` and land in `vaelor.hardware_bridge`'s ``run_argv``,
which spawns them as **root**. This module is the rule for what may cross that
line, and it is deliberately NOT the SSH transport's allowlist.

**Why the two boundaries cannot share one rule.** On a worker,
:data:`vaelor.ssh_transport.REMOTE_COMMAND_ALLOWLIST` bounds a channel whose
client is already a full sudoer - the login Vaelor holds could run anything it
likes with or without that list, so the list is hygiene, not a privilege
boundary. The controller's bridge socket is the opposite: its clients are the
control plane and the workload executor (`vaelor.bridge_peers`), neither of
which holds any root primitive of its own. Handing them the SSH list would hand
them ``python3 -c`` as root in one call. So the rule here is the INTERSECTION
of that list with what the GPU runtime actually emits on this path, and each
surviving token is pinned to the SHAPE the runtime uses.

**Nothing here writes a file or starts a unit (VD-143).** The first cut of this
policy admitted ``tee`` of a ``vaelor-vllm-*.service`` and never read the body
``tee`` was handed, so a caller could write an ``ExecStart`` of its choosing,
``daemon-reload`` it and ``enable --now`` it: root in three accepted calls. That
residual is closed at its cause rather than by reading unit text: the bridge
renders every unit it installs from the runtime's own template and typed
values (`vaelor.bridge_managed_units`), and ``tee``, ``install``, ``stat``,
``sha256sum``, ``systemctl enable`` and ``systemctl start`` are gone from this
table. The model store's reads and removals left with them, to verbs that
follow no link - a root ``cat`` or ``rm -rf`` inside a group-writable store
followed whatever link a group member planted there.

Pure text, so the whole surface is testable without a socket, a node or a root
process. Both boundaries read it: `BridgeTransport.run` refuses client-side so a
bad argv never crosses the socket, and ``run_argv`` refuses again on the far
side, which is the check that matters because everything on this side is
control-plane-writable (LESSONS #178).

The shapes, and where each comes from in `vaelor.gpu_pool_runtime` /
`vaelor.gpu_node_facts`:

============  =================================================================
``cat``       ``/etc/group`` (the container's numeric GIDs) and
              ``/proc/net/route`` (the cluster NIC) - two kernel/system facts.
``docker``    ``image inspect``/``pull`` of :data:`BRIDGE_IMAGE_REFS` - the one
              image the runtime serves from, not any ref. A ``stop``/``rm`` of a
              Vaelor-named container is permitted for a rollback that has to
              clear one by hand. ``run`` is NOT: every container the deploy
              starts is launched by a unit the bridge rendered, so a ``docker
              run`` arriving here would be a container nothing wrote.
``rm``        ``-f`` of a Vaelor unit file - the removal half of a stop. Never
              a store path, a gate config or anything recursive.
``systemctl`` ``daemon-reload``, or one verb that stops, resets or reads a
              ``vaelor-vllm-*`` / ``vaelor-model-pull-*`` unit. Never ``enable``
              or ``start``: a managed unit is started only by the install that
              rendered it.
============  =================================================================
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Sequence


#: Where every managed unit file lives. Spelled here rather than imported from
#: `gpu_pool_runtime` so this module stays free of the code it polices - a
#: policy that imported its subject could be widened by a change to that
#: subject.
UNIT_ROOT = "/etc/systemd/system"

#: The first-argv tokens the GPU runtime emits through the bridge transport,
#: and so the only ones the root bridge accepts. The SSH allowlist's other
#: tokens are absent because the controller path never emits them; ``python3``
#: most of all, and since VD-143 every token that writes a file.
BRIDGE_ARGV_TOKENS = frozenset({"cat", "docker", "journalctl", "rm", "systemctl"})

#: The image references ``docker pull``/``image inspect`` may name. A ref
#: PATTERN would have let the bridge fetch and run anything a registry serves,
#: which is the one docker verb reachable here that puts new code on the box, so
#: this is a membership test against the images the GPU tier serves from: the
#: pinned table in `vllm_images` (vLLM 0.22.1, the rollback, and AMD's vLLM
#: 0.27 / ROCm 10 build for gfx1151, the default).
#:
#: Spelled literally for the reason `UNIT_ROOT` is: importing
#: `vllm_images.IMAGES` would make the policy import the code it polices, so a
#: change to that module could widen this one. The two are tied instead by
#: `tests/test_image_pins.py`, which asserts this set is exactly the table's
#: images - a failing test rather than a silent widening.
BRIDGE_IMAGE_REFS = frozenset({
    "oci-registry.ryai.dev/ryai-vllm:latest"
    "@sha256:ff4c0d784b14bcde538f015b5551d251339fc83a223e9f2967f3d1a3b05c3973",
    "rocm/vllm"
    "@sha256:b8a082f346d069376d35784250e38b23a043efe979408ae3a33d7c6b62ee3276",
})

#: Every matching rule below is a PATTERN STRING compiled at its use site, never
#: a module-level `re.Pattern`. That is the tree's convention for a rule that no
#: table instruments (`gpu_pool_units.SERVING_UNIT_PATTERN` is the same shape),
#: and it keeps each rule visible to the vocabulary-reachability guard rather
#: than sitting in a module-level binding nothing measures (LESSONS #197).

#: The two managed unit/container families. Every ``systemctl`` verb, every unit
#: file removed, and every container named in a ``docker stop``/``rm`` belongs
#: to one of them, so nothing here can reach an unrelated unit.
_MANAGED = r"vaelor-(?:vllm|model-pull)-[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
_MANAGED_UNIT = _MANAGED + r"\.service"
_UNIT_PATH = re.escape(UNIT_ROOT) + "/" + _MANAGED_UNIT

#: The fixed files a ``cat`` may read: two kernel/system facts, neither under a
#: directory any account but root can change.
_READABLE = ("/etc/group", "/proc/net/route")

#: The verbs a managed unit accepts here, and the flags that may ride with them.
#: ``--property=`` carries the property list `pull_unit_status` asks for;
#: ``--now`` rides only with ``disable``, which then stops the unit too.
_SYSTEMCTL_VERBS = frozenset({
    "disable", "stop", "reset-failed", "show", "is-active",
})
_SYSTEMCTL_FLAG = r"--(?:now|property=[A-Za-z,]{1,200})"

#: The verbs that CHANGE what systemd will run, as opposed to reading.
#: `vaelor.hardware_bridge` serialises only these; see `mutates_units`.
_MUTATING_VERBS = frozenset({
    "daemon-reload", "disable", "stop", "reset-failed",
})

#: Characters that may never appear anywhere in an argv reaching the root side.
#: A NUL truncates a C string, and a newline is how a group file or a property
#: list would be forged from a single argument.
_FORBIDDEN_CHARACTERS = "\x00\n\r"


def _matches(pattern: str, value: str) -> bool:
    """Whether ``value`` is the WHOLE of ``pattern``.

    ``fullmatch`` throughout, never ``search``: a rule that matched a substring
    would accept ``/etc/systemd/system/vaelor-vllm-x.service.evil`` for the unit
    it names. `re` caches the compiled form, so compiling at the use site costs
    nothing and keeps the rule where it is read.
    """
    return re.fullmatch(pattern, value) is not None


def _split_flags(arguments: Sequence[str]) -> tuple:
    """``(flags, words)`` for a verb's arguments, in the order given.

    A flag is any token beginning with ``--``; everything else is a word. Both
    are returned so a shape can check the flags it allows and then require the
    exact number of words it expects, whichever order they arrived in.
    """
    flags = [item for item in arguments if item.startswith("--")]
    words = [item for item in arguments if not item.startswith("--")]
    return flags, words


#: The one journal read: the last lines of THIS RUN of ONE vLLM server unit,
#: as plain text. Nothing else - no gate, split worker, balancer or model-pull
#: unit (only the server runs vLLM, and only its refusal is wanted), no
#: earlier run (``-I``, the current invocation: a tail spanning runs could
#: quote a previous start's refusal as this one's), no follow, no output
#: format that could carry a field the reader did not ask for, no option that
#: changes the journal (``--vacuum-*``, ``--rotate``). ``-I`` needs systemd
#: 256 or later; an older ``journalctl`` rejects it, the read fails, and the
#: failure is said without vLLM's words (`gpu_pool_startup.server_died`) -
#: never retried as a tail across runs. The server's own log: no key or token
#: is put in its command line or environment and request logging is left
#: off, so it carries no key and no request (DECISIONS VD-154).
JOURNAL_LINES = "200"
_SERVER_UNIT = r"vaelor-vllm-[a-z0-9][a-z0-9-]{0,38}-server\.service"
JOURNAL_SERVER_ONLY = "Only a vLLM server unit's journal is read."


def journal_argv(unit: str) -> List[str]:
    """The one ``journalctl`` argv the caller sends and :func:`_check_journalctl` admits."""
    if not _matches(_SERVER_UNIT, str(unit)):
        raise ValueError(JOURNAL_SERVER_ONLY)
    return ["journalctl", "-u", str(unit), "-I", "-n", JOURNAL_LINES, "--no-pager", "-o", "cat"]


def _check_journalctl(arguments: List[str]) -> bool:
    return (
        len(arguments) == 8
        and arguments[0] == "-u" and _matches(_SERVER_UNIT, arguments[1])
        and arguments[2:] == ["-I", "-n", JOURNAL_LINES, "--no-pager", "-o", "cat"]
    )


def _check_cat(arguments: List[str]) -> bool:
    return len(arguments) == 1 and arguments[0] in _READABLE


def _check_rm(arguments: List[str]) -> bool:
    """``rm -f <one managed unit file>``, exactly - the removal half of a stop.

    A unit file lives in a directory only root can write, so the path cannot
    be redirected; and removing one takes a unit away rather than adding one.
    """
    return (
        len(arguments) == 2 and arguments[0] == "-f"
        and _matches(_UNIT_PATH, arguments[1])
    )


def _check_systemctl(arguments: List[str]) -> bool:
    if arguments == ["daemon-reload"]:
        return True
    if not arguments or arguments[0] not in _SYSTEMCTL_VERBS:
        return False
    flags, words = _split_flags(arguments[1:])
    if not all(_matches(_SYSTEMCTL_FLAG, flag) for flag in flags):
        return False
    if "--now" in flags and arguments[0] != "disable":
        return False
    return len(words) == 1 and _matches(_MANAGED_UNIT, words[0])


def _check_docker(arguments: List[str]) -> bool:
    """``image inspect``/``pull`` of the one image, or ``stop``/``rm`` of a
    container.

    ``run`` is absent on purpose: the deploy's containers are started by the
    systemd units the bridge renders itself, so a ``docker run`` arriving
    through the bridge would be a container with no unit supervising it - and
    ``run`` is the one docker verb that takes ``--privileged``, a bind mount of
    ``/`` and an arbitrary entrypoint.

    The ref is a MEMBERSHIP test against :data:`BRIDGE_IMAGE_REFS`, not a ref
    pattern. A pull is how new code arrives on the box, so "any well-formed
    reference" would let whoever holds the socket fetch an image of their
    choosing and leave it named on the node for a unit to run.
    """
    if arguments[:2] == ["image", "inspect"]:
        return len(arguments) == 3 and arguments[2] in BRIDGE_IMAGE_REFS
    if arguments[:1] == ["pull"]:
        return len(arguments) == 2 and arguments[1] in BRIDGE_IMAGE_REFS
    if arguments[:1] in (["stop"], ["rm"]):
        flags, words = _split_flags(arguments[1:])
        if flags:
            return False
        return len(words) == 1 and _matches(_MANAGED, words[0])
    return False


_SHAPES: Dict[str, Callable[[List[str]], bool]] = {
    "cat": _check_cat,
    "docker": _check_docker,
    "journalctl": _check_journalctl,
    "rm": _check_rm,
    "systemctl": _check_systemctl,
}


def check_bridge_argv(argv: Sequence[str]) -> None:
    """Raise :class:`ValueError` unless ``argv`` is one of the reviewed shapes.

    Returns nothing on success, so a caller reads as ``check_bridge_argv(args)``
    followed by the thing it is guarding, and cannot forget to look at a boolean.

    The first token is matched EXACTLY and bare: ``/usr/bin/docker`` is not
    ``docker``, because a policy keyed on a path invites a second path to the
    same binary - and because the runtime spells every token bare, an absolute
    one never came from it.
    """
    args = [str(item) for item in (argv or [])]
    if not args:
        raise ValueError("A controller command needs an argument list.")
    for token in args:
        if any(character in token for character in _FORBIDDEN_CHARACTERS):
            raise ValueError(
                "A controller command may not contain a newline or a null byte."
            )
        if ".." in token:
            raise ValueError(
                "A controller command may not walk out of a directory with '..'."
            )
    if args[0] not in BRIDGE_ARGV_TOKENS:
        raise ValueError(
            "{!r} is not a reviewed cluster command.".format(args[0])
        )
    if not _SHAPES[args[0]](args[1:]):
        raise ValueError(
            "Those {} arguments are outside the shapes this controller runs as "
            "root.".format(args[0])
        )


#: What either side of the socket says to a controller command carrying stdin:
#: no reviewed shape reads it, so text arriving there could only be a body for
#: a command to write - the thing VD-143 took off this path.
STDIN_REFUSAL = "A controller command may not carry input text."


def mutates_units(argv: Sequence[str]) -> bool:
    """Whether ``argv`` changes what systemd will run on this controller.

    `vaelor.hardware_bridge` holds its runtime lock across these and nothing
    else. A unit removal and a ``daemon-reload`` must not interleave with
    another deploy's install (which takes the same lock, root-side); a ``docker
    pull``, which can hold the process for an hour, must not block the NPU, GPU
    and proxy verbs that share that lock.

    **Deleting a unit file counts.** ``stop_unit`` and ``stop_pull_unit`` both
    ``rm -f`` a unit and then ``daemon-reload``, and a removal interleaved with
    another deploy's write and reload is the same race from the other end.

    Read on an argv that has already passed :func:`check_bridge_argv`, so the
    shapes it recognises are the only ones these tokens can be in.
    """
    args = [str(item) for item in (argv or [])]
    if not args:
        return False
    if args[0] == "systemctl":
        return len(args) > 1 and args[1] in _MUTATING_VERBS
    return args[0] == "rm" and len(args) > 2 and args[2].startswith(UNIT_ROOT)
