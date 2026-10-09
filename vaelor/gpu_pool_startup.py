"""Wait for a started vLLM server's API while the server is ALIVE - and no longer.

The sibling of `gpu_pool_pull` for the step after the weights are on disk,
housed out of `gpu_pool_operations` for the line ceiling `CLAUDE.md` sets. The
runtime still owns every command (the liveness read is
`GpuPoolRuntime.unit_state`, the same ``systemctl show`` `stop_unit` makes);
this owns the waiting, the cap rule, the progress sentence and the three
failure sentences.

**Why the wait reads the unit and not only a clock (VD-127, the
bigger-than-one-box proof).** The first attempt to serve an FP8 32B across the
two Strix Halo nodes loaded its weights in a minute and then spent the next
twelve compiling Triton block-scaled kernels - an iGPU with no FP8 hardware
does that work in software, once per fresh container. The deploy's wait was a
15-minute timer, and at minute fifteen exactly it stopped a server that was
alive and working, drained the Ray worker (killed 137), and wrote a failed
record with no reason. The fact the deploy needed - "is the server still
there?" - was one property read away on a transport it already held, which is
the second time in VD-127 a timer stood in for a fact this process holds
(LESSONS pattern 14, the reconcile's 6 h backstop was the first). So:

* **A server that is not alive fails the wait NOW**, with the unit, the node
  and what systemd said of it (``ActiveState``/``SubState``, and the
  ``Result``/``ExecMainStatus`` of the exit) - and, when its journal holds
  one, vLLM's own reason (:func:`startup_refusal`): "the estimated maximum
  model length is 175984" says what to change where "exit code 1" says
  nothing. The journal is read through ONE reviewed shape
  (`bridge_argv_policy.JOURNAL_LINES`: the last lines of one managed unit,
  as text), and a journal that cannot be read leaves the plain sentence.
* **A server that is alive is waited for**, up to a HARD CAP that is a
  documented fact about first-start compile on this class of GPU
  (:data:`STARTUP_CAP_SECONDS`), with a progress line every minute that says
  so. The cap sentence says the server was still alive and that the cap, not
  the server, ended the wait, so the operator reads a bound and not a crash.
* **A liveness read that FAILS is not a server that died.** A bridge or SSH
  hiccup is bounded the way the pull pollers bound it
  (`gpu_pool_pull.PULL_TRANSPORT_FAILURES`); only a node that stops answering
  altogether is the failure, and the sentence then names the node.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Mapping, Optional

from .gpu_pool_pull import PULL_TRANSPORT_FAILURES
from .gpu_pool_units import SERVER_ROLE, unit_role
from .ssh_transport import SshTransportError

#: What a health probe answers when NOTHING at the address answered - no
#: connection, a timeout - as against ``False`` for a server that answered and
#: was not healthy and ``True`` for one that was. The one probe seam serves
#: the startup waits (which read truth alone) and the mode watch (which records
#: the difference on a replicated row as ``reachable``, VD-129), so the
#: distinction is made where the socket is, `probe_health`, and nowhere else.
UNREACHABLE = None


def probe_health(url: str) -> Optional[bool]:
    """``True`` healthy, ``False`` answered unhealthy, :data:`UNREACHABLE` if nothing did.

    The production probe behind every readiness wait and the mode watch's
    replica reading. An HTTP error is an answer (a server still loading says
    503); a connection failure or a timeout is not.
    """
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False
    except (OSError, urllib.error.URLError):
        return UNREACHABLE

#: How long an ALIVE vLLM server may take to answer its API before the deploy
#: gives up on it. 3600 s is a fact about the pair, not a guess: the FP8 32B's
#: first start compiled for over twelve minutes before the old 900 s deadline
#: killed it, and a cold inductor cache on a two-rank pipeline can double that.
#: ``startup_timeout_seconds`` on the deploy payload stays an operator override,
#: clamped to the floor and ceiling below - a cap under a minute would fail
#: every real start, and one over two hours is a hung server nobody is told
#: about.
STARTUP_CAP_SECONDS = 3600
STARTUP_CAP_FLOOR = 60
STARTUP_CAP_CEILING = 7200

#: The gap between API probes, and how often the job's progress line is
#: rewritten while the server is alive and still loading or compiling.
API_POLL_SECONDS = 3.0
STARTUP_REPORT_SECONDS = 60

#: The percent every startup checkpoint reports: the deploy's own ladder has
#: the units started at 70 and health at 100, and inventing a climbing number
#: from elapsed time would be a second, disagreeing progress figure.
STARTUP_CHECKPOINT_PERCENT = 82

#: The progress line while the server is alive and its API has not answered.
STARTUP_PROGRESS = (
    "vLLM server alive on {node}, waiting for its API "
    "({minutes} min; first start compiles kernels)"
)

#: Failure: the unit left its running state while the deploy waited. ``state``
#: is :func:`describe_unit_state`'s reading of what systemd reported.
SERVER_DIED = (
    "The vLLM server {unit} on {node} stopped while the deploy was waiting for "
    "its API ({state})."
)

#: What follows :data:`SERVER_DIED` when the unit's journal says why.
SERVER_SAID = " vLLM said: {reason}"

#: The lines of a vLLM start-up that carry its reason: an exception's own
#: message, and the fit sentences vLLM writes before refusing. The LAST such
#: line wins - vLLM chains "see root cause above" after the cause.
_REASON_MARKERS = (
    "ValueError: ", "RuntimeError: ", "OutOfMemoryError: ", "torch.OutOfMemoryError: ",
    "AssertionError: ",
)
_ROOT_CAUSE_ECHO = "See root cause above"
_REASON_LIMIT = 600


def startup_refusal(journal: Any) -> str:
    """vLLM's own reason for not starting, from its journal tail; ``""`` if none.

    The message of the last exception line that is not merely pointing back
    at an earlier one - so vLLM's "Engine core initialization failed. See root
    cause above." gives way to the cause itself, typically its "estimated
    maximum model length is N" refusal. Bounded, and never the traceback.
    """
    best = ""
    for line in str(journal or "").splitlines():
        for marker in _REASON_MARKERS:
            index = line.find(marker)
            if index < 0:
                continue
            message = line[index + len(marker):].strip()
            if message and _ROOT_CAUSE_ECHO not in message:
                best = message
            break
    return best[:_REASON_LIMIT]


def server_died(runtime: Any, transport: Any, unit: str, node: str, state: Mapping[str, str]) -> str:
    """:data:`SERVER_DIED`, with vLLM's reason from the unit's journal when it has one.

    One sentence for the distributed lead's wait and each replica's alike.
    Only a server unit's journal is read - a dead gate is named without one.
    A journal that cannot be read - a node that stopped answering, an older
    bridge without the shape, a ``journalctl`` too old for ``-I`` - leaves
    the plain sentence, never a tail spanning earlier runs.
    """
    sentence = SERVER_DIED.format(unit=unit, node=node, state=describe_unit_state(state))
    reason = ""
    try:
        if unit_role(unit) == SERVER_ROLE:
            reason = startup_refusal(runtime.unit_journal(transport, unit))
    except (SshTransportError, ValueError, RuntimeError):
        reason = ""
    return sentence + (SERVER_SAID.format(reason=reason) if reason else "")


#: Failure: the cap, not the server, ended the wait. Said in those words so a
#: bound is never read as a crash.
STARTUP_CAP_REACHED = (
    "The vLLM server on {node} was still alive after {minutes} minutes but its "
    "API had not answered; the {cap}-minute startup cap ended the wait, not the "
    "server. A first start compiles kernels; raise startup_timeout_seconds to "
    "wait longer."
)

#: Failure: the node holding the server stopped answering the liveness read.
NODE_UNREACHABLE = (
    "{node} stopped answering while its vLLM server was starting: {error}"
)

#: The ``ActiveState`` values under which the server's main process exists. An
#: ``activating`` unit is alive too - ``ExecStartPre`` is running or the main
#: process is about to fork - EXCEPT in ``auto-restart``, which is systemd
#: holding a unit whose process has DIED for ``RestartSec`` before running it
#: again: a server that crashed and will be relaunched is a crash the operator
#: must see, not a restart loop the deploy waits an hour on.
_LIVE_ACTIVE_STATES = frozenset({"active", "reloading"})
_ACTIVATING = "activating"
_AUTO_RESTART = "auto-restart"


def startup_cap_seconds(requested: Any) -> int:
    """The wait's hard cap: the operator's override, clamped, else the default."""
    try:
        seconds = int(requested)
    except (TypeError, ValueError):
        seconds = STARTUP_CAP_SECONDS
    return max(STARTUP_CAP_FLOOR, min(seconds, STARTUP_CAP_CEILING))


def unit_alive(properties: Mapping[str, str]) -> bool:
    """Whether a serving unit's systemd properties say its process exists.

    The one reading of "alive" the startup wait acts on. ``failed``,
    ``inactive`` and ``deactivating`` are not; ``activating`` is unless systemd
    is in ``auto-restart`` over a dead process (see :data:`_LIVE_ACTIVE_STATES`).
    """
    active = str(properties.get("ActiveState", "") or "").strip()
    sub = str(properties.get("SubState", "") or "").strip()
    if active in _LIVE_ACTIVE_STATES:
        return True
    return active == _ACTIVATING and sub != _AUTO_RESTART


def describe_unit_state(properties: Mapping[str, str]) -> str:
    """``active/running``, with the exit evidence when systemd recorded any.

    ``Result`` and ``ExecMainStatus`` ride along whenever they say something -
    a ``Result`` other than ``success`` or a non-zero exit - so a failure
    sentence carries the one fact the allowlists let this module read about
    WHY the process is gone (an OOM kill's 137, a Python traceback's 1).
    """
    active = str(properties.get("ActiveState", "") or "?").strip()
    sub = str(properties.get("SubState", "") or "?").strip()
    result = str(properties.get("Result", "") or "").strip()
    status = str(properties.get("ExecMainStatus", "") or "").strip()
    evidence = []
    if result and result != "success":
        evidence.append(f"Result={result}")
    if status and status != "0":
        evidence.append(f"ExecMainStatus={status}")
    text = f"{active}/{sub}"
    return f"{text} ({', '.join(evidence)})" if evidence else text


class ServerStartupWaiter:
    """Block until a started server's API answers, while the server is alive.

    ``runtime`` is the :class:`vaelor.gpu_pool_runtime.GpuPoolRuntime` (or a
    test's mock of it); ``probe`` answers whether the API URL is up; ``sleep``
    and ``monotonic`` are the poll wait and the clock, injected in the shape
    `ModelPullWaiter` uses so a forty-minute compile is driven in a unit test
    in milliseconds.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        probe: Callable[[str], bool],
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.runtime = runtime
        self._probe = probe
        self._sleep = sleep
        self._monotonic = monotonic

    def wait(
        self,
        transport: Any,
        node: Dict[str, Any],
        unit: str,
        endpoint: str,
        payload: Mapping[str, Any],
        report: Callable[[int, str], None],
        probe_url: Optional[str] = None,
    ) -> None:
        """Wait for ``endpoint`` to answer while ``unit`` on ``node`` is alive.

        ``probe_url`` replaces the model-list probe where a keyed gate fronts
        the server: the gate answers ``/health`` without the key, and nothing
        else.

        The order inside one poll is the argument: the API is probed first (a
        server that answers is healthy whatever else is true); then the unit
        is read, and a unit that is not alive fails the wait at once; only
        then is the cap consulted, so a cap sentence is only ever written
        about a server that was just seen alive.
        """
        cap = startup_cap_seconds(payload.get("startup_timeout_seconds"))
        url = probe_url or self.runtime.health_url(endpoint)
        name = str(node.get("name", node.get("id", "")))
        started = self._monotonic()
        reported_minute = -1
        unreachable = 0
        while True:
            if self._probe(url):
                return
            try:
                state = self.runtime.unit_state(transport, unit)
            except SshTransportError as error:
                unreachable += 1
                if unreachable >= PULL_TRANSPORT_FAILURES:
                    raise RuntimeError(
                        NODE_UNREACHABLE.format(node=name, error=error)
                    ) from error
            else:
                unreachable = 0
                if not unit_alive(state):
                    raise RuntimeError(server_died(self.runtime, transport, unit, name, state))
            elapsed = self._monotonic() - started
            if elapsed >= cap:
                raise RuntimeError(STARTUP_CAP_REACHED.format(
                    node=name, minutes=int(elapsed // 60), cap=cap // 60,
                ))
            minute = int(elapsed // STARTUP_REPORT_SECONDS)
            if minute > reported_minute:
                reported_minute = minute
                report(
                    STARTUP_CHECKPOINT_PERCENT,
                    STARTUP_PROGRESS.format(node=name, minutes=minute),
                )
            self._sleep(API_POLL_SECONDS)
