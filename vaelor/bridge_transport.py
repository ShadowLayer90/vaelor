"""The controller's own transport: the same command surface, over the root bridge.

VD-125. A GPU cluster made of the head controller and one worker is the lab's
only two-node shape and the obvious product shape, and the controller's GPU is
half of it. The controller cannot be enrolled as its own SSH worker
(`cluster_manager.enroll` refuses that, correctly — it is the Swarm manager, not
a member), so it needs a transport of its own.

This is that transport, and for READS and ``docker`` it is deliberately **the
same object shape** `SshTransport` presents to the layers above: a
``run(arguments, sudo=…, stdin_text=…, timeout=…) -> str`` and a ``profile``
mapping carrying ``host``. `gpu_node_facts` never learns which it holds, and
every ``except SshTransportError`` works unchanged, because failures here raise
that same class (its docstring records that the name is historical and it is
now the one transport error type).

**Except for anything that WRITES (VD-143).** A unit, the pull program and the
model store are never written through ``run`` here: no ``tee``, no ``install``,
no ``rm -rf`` crosses this socket. `GpuPoolRuntime` asks
:func:`root_renders_units` and, for this transport, sends the unit's KIND and
typed VALUES through :meth:`BridgeTransport.install_managed_unit`; the root
side renders the unit from the runtime's own template and writes it with
no-follow file operations. The two store operations the model library needs -
reading a pull's progress record and removing a repo's weights - are verbs of
their own for the same reason: a ``cat`` or an ``rm -rf`` spawned as root
follows whatever link an account in the store's group planted.

**Where the privilege comes from, and why there is no ``sudo``.** On a worker,
`SshTransport` logs in as an unprivileged sudoer and elevates per command. Here
there is nobody to log in as: the process holding this object IS the
unprivileged control plane, and the privilege lives behind
`vaelor.hardware_bridge` — the root service the appliance already runs for the
NPU and GPU model servers. So ``sudo`` is accepted and ignored: the bridge IS
the privilege boundary, exactly as it is for ``gpu_start``, and a flag asking
for elevation that has already happened is meaningless rather than wrong.

**Two checks, and the one that matters is the far one.** The argv policy is
enforced here so a bad argv is refused before it crosses the socket — a cheap,
local, honest failure — and again inside the bridge's ``run_argv``, which is the
boundary that actually spawns the process and the only one an attacker on this
side cannot skip (LESSONS #178). Both read
:func:`vaelor.bridge_argv_policy.check_bridge_argv`, its one home — which is
**narrower** than the SSH allowlist on purpose: this socket's clients are the
control plane and the workload executor (`vaelor.bridge_peers`), neither a
sudoer, so the rule is the intersection of that allowlist with the shapes the
GPU runtime actually emits here. No reviewed shape reads stdin, so a non-empty
``stdin_text`` is refused on both sides as well: text that could only be a
body for some command to write is never carried across (VD-143).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .bridge_argv_policy import STDIN_REFUSAL, check_bridge_argv
from .hardware_bridge import (
    RUN_ARGV_MAX_TIMEOUT_SECONDS,
    RUN_ARGV_TIMEOUT_SECONDS,
    HardwareBridgeClient,
    HardwareBridgeError,
)
from .ssh_transport import SshTransportError

#: How much longer than the command's own deadline this side waits on the
#: socket. The ROOT side is what bounds the command (it kills the child at the
#: deadline and answers with the reason); this side must outlast that, or a slow
#: command comes back as "the bridge is unavailable" instead of as its own
#: failure - and a container image pull would be reported as a dead bridge.
_REPLY_GRACE_SECONDS = 30


def root_renders_units(transport: Any) -> bool:
    """Whether ``transport`` is the controller's root bridge, which renders units itself.

    ``is True`` rather than truthiness, so a test double that answers every
    attribute (a ``Mock``) is never mistaken for the bridge.
    """
    return getattr(transport, "renders_units", False) is True


class BridgeTransport:
    """Run reviewed cluster commands on THIS machine through the root bridge."""

    #: The root side renders every unit this transport installs (VD-143).
    renders_units = True

    def __init__(self, host: str, client: Optional[Any] = None):
        # `profile` exists because the layers above read `transport.profile
        # ["host"]` to say which machine a command went to. The controller has
        # no SSH login, so the only true field is the address the cluster
        # reaches it on - the one its own Swarm advertise address carries - and
        # the rest of an SSH profile is deliberately absent rather than faked.
        self.profile: Dict[str, Any] = {"host": str(host), "transport": "bridge"}
        self._client = client or HardwareBridgeClient()

    def run(
        self,
        arguments: Iterable[str],
        *,
        sudo: bool = False,
        stdin_text: str = "",
        timeout: Optional[int] = None,
    ) -> str:
        """Stdout of one allowlisted argv run as root on the controller.

        ``sudo`` is accepted and ignored - see the module docstring: the bridge
        is the privilege boundary, so the command is already root and there is
        no elevation step to perform. The parameter stays in the signature
        because `GpuPoolRuntime` passes it on every call and must not have to
        know which transport it holds.

        ``timeout`` bounds the COMMAND, on the root side, exactly as it does
        over SSH - and this side's socket wait is stretched to outlast it. The
        client's default wait is three seconds, which is right for the telemetry
        verbs the bridge was built for and far too short for these: an image
        pull holds the socket for as long as the pull takes, and a socket that
        gave up first would report a multi-GB fetch as an unavailable bridge.

        Failures - a refused argv, an unreachable bridge, a non-zero exit, an
        answer too big for the socket - all raise :class:`SshTransportError`,
        which is what every caller already catches. The last of those is a
        REFUSAL rather than a short string: every output this transport carries
        is parsed (a group file, a routing table, a progress JSON, a property
        list), and a cut one parses cleanly into the wrong facts.
        """
        args = [str(item) for item in arguments]
        try:
            check_bridge_argv(args)
            if stdin_text:
                raise ValueError(STDIN_REFUSAL)
        except ValueError as error:
            raise SshTransportError(str(error)) from error
        original = getattr(self._client, "timeout", None)
        try:
            self._client.timeout = self._reply_wait(timeout)
            reply = self._client.run_argv(
                args, stdin_text=stdin_text, timeout=timeout
            )
        except HardwareBridgeError as error:
            raise SshTransportError(str(error)) from error
        finally:
            if original is not None:
                self._client.timeout = original
        if reply.get("truncated"):
            raise SshTransportError(
                "The {} output exceeded the bridge limit, so this controller "
                "cannot answer from it.".format(args[0])
            )
        return str(reply.get("stdout", "") or "")

    def install_managed_unit(self, kind: str, values: Dict[str, Any]) -> None:
        """Have the root bridge render, write and start one managed unit.

        The kind and the typed values cross the socket; the unit text never
        does. The bridge refuses a kind or a value name the runtime's template
        does not take, re-checks every value, writes the file it rendered and
        only then reloads and starts it (VD-143).
        """
        self._bridge_call(self._client.managed_unit_install, str(kind), dict(values))

    def prepare_ray_plane(
        self, name: str, token: str, link: str, address: str, peers: List[str],
        shared: bool = False,
    ) -> None:
        """Have the root bridge write a split's slice, token and fence here (ACC-163/187)."""
        self._bridge_call(self._client.ray_plane_prepare, str(name), str(token),
                          str(link), str(address), [str(peer) for peer in peers],
                          bool(shared))

    def ray_container_cgroup(self, name: str, role: str) -> str:
        """Where one of a split's Ray containers runs here, ``""`` while it is not yet."""
        reply = self._bridge_call(self._client.ray_container_cgroup, str(name), str(role))
        return str((reply or {}).get("cgroup", "") or "")

    def clear_ray_plane(self, name: str) -> None:
        """Have the root bridge remove a split's token and firewall here."""
        self._bridge_call(self._client.ray_plane_clear, str(name))

    def read_pull_progress(self, repo: str) -> str:
        """The text of a pull's progress record, read as root without following a link."""
        reply = self._bridge_call(self._client.pull_progress_read, str(repo))
        return str(reply.get("text", "") or "")

    def remove_model_cache(self, repo: str) -> str:
        """Remove a repo's cached weights as root, without following a link."""
        reply = self._bridge_call(self._client.model_cache_remove, str(repo))
        return str(reply.get("path", "") or "")

    @staticmethod
    def _bridge_call(verb, *arguments) -> Dict[str, Any]:
        """One structured bridge verb, its failure raised as the transport error."""
        try:
            reply = verb(*arguments)
        except HardwareBridgeError as error:
            raise SshTransportError(str(error)) from error
        return reply if isinstance(reply, dict) else {}

    @staticmethod
    def _reply_wait(timeout: Optional[int]) -> float:
        """How long to wait on the socket for a command allowed ``timeout``.

        Mirrors the root side's own clamp so the two agree on the deadline, then
        adds the grace that keeps the command's failure - not the socket's - the
        one the caller sees.
        """
        try:
            seconds = int(
                timeout if timeout is not None else RUN_ARGV_TIMEOUT_SECONDS
            )
        except (TypeError, ValueError):
            seconds = RUN_ARGV_TIMEOUT_SECONDS
        seconds = min(RUN_ARGV_MAX_TIMEOUT_SECONDS, max(1, seconds))
        return float(seconds + _REPLY_GRACE_SECONDS)


def default_bridge_transport(node: Dict[str, Any]) -> BridgeTransport:
    """The controller's transport: its own commands, over the root bridge.

    The production factory `GpuPoolOperations` and `GpuModelLibrary` default
    to, public here (VD-127 cleanup item 28) so the whole controller path is
    injectable in one place - a test hands the operations a factory and can
    then prove the bridge was used for the controller and SSH for every other
    node, which a transport that reached for a socket on construction could
    not show.
    """
    return BridgeTransport(str(node.get("host", "")), client=HardwareBridgeClient())
