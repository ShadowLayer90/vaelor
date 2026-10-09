"""The LLM Server's wake door: a keyed client wakes an idle-unloaded cluster model.

ACC-058. An idle cluster model is scaled to zero (G3b) and counts as available
(owner, 2026-09-28): AI Chat and the ``/inference/v1`` gateway wake it on their
next request, but the LLM Server's port (11434) was closed while the model was
unloaded, so an OpenAI client of the LLM Server could never bring it back.

**The shape.** While the deployment is unloaded AFTER SITTING IDLE
(`gpu_serving_target.unload_cause`), and ONLY while this responder is up
(:func:`wake_door_ready`), the Mode B reconcile keeps the LLM Server's keyed
nginx door up and points its upstream at this responder's UNIX SOCKET instead
of the stopped model (`gpu_cluster_lan_doors.ClusterLanDoorsMixin._converge_wake_door`).
nginx refuses a request without a current key; one that passes reaches the
responder, which asks the wake seam AI Chat uses
(``ControlPlaneRuntime._cluster_wake``, asked as `gpu_idle_watch.LLM_SERVER_WAKE`,
VD-210) and answers **503 with
``Retry-After``** and an OpenAI-shaped error, so a client retries while the
model loads. Once the load finishes, the next healthy pass points the door back
at the model.

**Why a unix socket, and who may own it (review S2).** nginx forwards the
client's ``Authorization`` header, so whatever sits behind the door sees the
plaintext key. A loopback TCP port can be taken by any local process the
moment the responder is not holding it; this socket lives in
:data:`WAKE_SOCKET_DIR`, a folder the control plane's account creates and
owns, which no other account can write into or rename (``/run/vaelor`` is
root-owned and sticky). Another member of the ``vaelor`` group could make that
folder FIRST - then the control plane refuses to adopt it, the root bridge
refuses to mount it (:func:`wake_socket_refusal`), and the door simply stays
down. The executor points the door here only while the socket answers as this
responder (:func:`wake_door_ready`).

**Manual unloads are not woken.** A manual unload records its cause in the mode
file: the reconcile keeps the door DOWN, and the seam - asked for the LLM
Server's own target - refuses to wake it even if a request reached here.

**Keys are enforced twice, and never decrypted here.** nginx checks the bearer;
the responder checks it again by FINGERPRINT against the broker's
fingerprint-only listing (no decrypt, no plaintext-read audit), in constant time
over every row. The key is never logged or echoed.

Runs in ONE process: the control plane, and only on a box with the root
hardware bridge (:func:`start_control_plane_wake_door`); it stops with the
process. The executor imports :func:`wake_door_ready`; the root bridge,
:func:`wake_socket_refusal` and the socket's paths.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import socket
import socketserver
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from .runtime_paths import run_path

LOGGER = logging.getLogger(__name__)

#: The responder's folder and socket. The folder is the control plane's; the
#: socket is world-connectable (0666) because nginx's worker runs as its own
#: user inside the gate container - connecting sends a request, it never hands
#: anything to the connecting side but the 503.
WAKE_SOCKET_DIR = run_path("llm-wake")
WAKE_SOCKET_NAME = "wake.sock"
WAKE_SOCKET = WAKE_SOCKET_DIR + "/" + WAKE_SOCKET_NAME
#: Where the gate container sees the folder (mounted read-only).
WAKE_SOCKET_MOUNT = "/run/vaelor-wake"
#: The account that must own the folder and the socket: the control plane's.
WAKE_SOCKET_OWNER = "vaelor"

#: Bounds on one connection: how long a slow client may hold it, and how many
#: may be served at once (the rest are closed unanswered).
REQUEST_TIMEOUT_SECONDS = 10
MAX_CONCURRENT_REQUESTS = 8
#: The largest request body read (and discarded) before answering.
MAX_DISCARDED_BODY = 1024 * 1024

#: How long the control plane waits, at its own start, for the root bridge's
#: socket to appear before concluding this machine has no bridge, and how
#: often it looks. Ten minutes covers the slowest boot seen on the pair many
#: times over; a machine with no bridge at all (a Pi) stops looking after it.
BRIDGE_WAIT_SECONDS = 600
BRIDGE_POLL_SECONDS = 2.0

#: The marker ``/health`` answers with, so the executor can tell this responder
#: from anything else that might answer on the socket.
HEALTH_CODE = "model_unloaded"

WAKING_MESSAGE = (
    "The model behind the LLM Server was unloaded after sitting idle and is "
    "loading now; retry this request in a moment."
)
NOT_WAKING_MESSAGE = (
    "The model behind the LLM Server is not loaded, and a request cannot load "
    "it; load it from Cluster > Deployments in the Vaelor console."
)
REFUSED_MESSAGE = "A valid LLM Server API key is required."
HEALTH_MESSAGE = "The model behind the LLM Server is unloaded."


def _retry_after() -> str:
    """The gateway's own wake hint, so both doors tell a client the same wait."""
    from .inference_gateway import GATEWAY_WAKE_RETRY_AFTER

    return GATEWAY_WAKE_RETRY_AFTER


def error_body(message: str, kind: str, code: str) -> bytes:
    """An OpenAI-shaped error body: ``{"error": {message, type, code}}``.

    One spelling for every answer the LLM Server's port gives while its model
    is unloaded: this responder's, and the door nginx answers for itself
    (`llm_gate_config.render_unloaded_config`).
    """
    return json.dumps(
        {"error": {"message": message, "type": kind, "code": code}},
        separators=(",", ":"),
    ).encode("utf-8")


_error = error_body


def presented_key(authorization: Optional[str]) -> str:
    """The bearer token of an ``Authorization`` header, or ``""``."""
    scheme, _, token = str(authorization or "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def key_admitted(presented: str, fingerprints: Sequence[str]) -> bool:
    """Whether ``presented``'s fingerprint is one of ``fingerprints``, in constant time.

    The fingerprint is the broker's own (`served_endpoint_keys.served_key_fingerprint`),
    so nothing is decrypted to compare; every row is compared whatever the outcome.
    """
    if not presented:
        return False
    from .served_endpoint_keys import served_key_fingerprint

    mine = served_key_fingerprint(presented).encode("utf-8")
    admitted = False
    for fingerprint in fingerprints:
        if hmac.compare_digest(str(fingerprint).encode("utf-8"), mine):
            admitted = True
    return admitted


class WakeDecision:
    """What one request is answered: ``(status, body, headers)``."""

    def __init__(self, status: int, body: bytes, headers: Optional[dict] = None):
        self.status = status
        self.body = body
        self.headers = dict(headers or {})


def decide(
    path: str, authorization: Optional[str], *,
    wake: Callable[[], Any], fingerprints: Callable[[], Sequence[str]],
) -> WakeDecision:
    """The responder's answer to one request, waking only a keyed client."""
    if str(path or "").split("?", 1)[0] == "/health":
        return WakeDecision(503, _error(HEALTH_MESSAGE, "model_unloaded", HEALTH_CODE))
    try:
        current = list(fingerprints())
    except Exception as error:  # noqa: BLE001 - an unreadable key set wakes nothing
        LOGGER.warning(
            "The LLM Server wake door could not read the key set (%s), so the "
            "request was refused and nothing was woken.", type(error).__name__,
        )
        return WakeDecision(503, _error(NOT_WAKING_MESSAGE, "model_unavailable", "model_unavailable"))
    if not key_admitted(presented_key(authorization), current):
        return WakeDecision(401, _error(REFUSED_MESSAGE, "invalid_request_error", "invalid_api_key"))
    try:
        woken = str(wake() or "")
    except Exception as error:  # noqa: BLE001 - a failed wake is an honest 503
        LOGGER.warning("The LLM Server wake door could not wake the model: %s", error)
        woken = ""
    if not woken:
        return WakeDecision(503, _error(NOT_WAKING_MESSAGE, "model_unavailable", "model_unavailable"))
    return WakeDecision(
        503, _error(WAKING_MESSAGE, "model_waking", "model_waking"),
        {"Retry-After": _retry_after()},
    )


class _Handler(BaseHTTPRequestHandler):
    """Frames :func:`decide` for every method; the server carries the seams."""

    server_version = "VaelorWake"
    sys_version = ""
    timeout = REQUEST_TIMEOUT_SECONDS

    def _answer(self, *, body: bool = True) -> None:
        try:
            length = min(int(self.headers.get("Content-Length") or 0), MAX_DISCARDED_BODY)
        except ValueError:
            length = 0
        if length > 0:
            self.rfile.read(length)
        decision = decide(
            self.path, self.headers.get("Authorization"),
            wake=self.server.wake, fingerprints=self.server.fingerprints,  # type: ignore[attr-defined]
        )
        self.send_response(decision.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(decision.body)))
        for name, value in decision.headers.items():
            self.send_header(name, value)
        self.end_headers()
        if body:
            self.wfile.write(decision.body)

    def do_HEAD(self) -> None:  # noqa: N802 - http.server's name
        self._answer(body=False)

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _answer

    def address_string(self) -> str:
        return "wake-door"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silent: the gate's access log is the LLM Server's, not this door's."""


class _Bounded:
    """At most :data:`MAX_CONCURRENT_REQUESTS` handled at once; the rest closed."""

    def __init__(self) -> None:
        self._slots = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)

    def process_request(self, request, client_address) -> None:  # type: ignore[no-untyped-def]
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)  # type: ignore[attr-defined]
            return
        try:
            super().process_request(request, client_address)  # type: ignore[misc]
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:  # type: ignore[no-untyped-def]
        try:
            super().process_request_thread(request, client_address)  # type: ignore[misc]
        finally:
            self._slots.release()


if hasattr(socketserver, "ThreadingUnixStreamServer"):
    class _UnixServer(_Bounded, socketserver.ThreadingUnixStreamServer):
        daemon_threads = True

        def __init__(self, *args, **kwargs):
            _Bounded.__init__(self)
            socketserver.ThreadingUnixStreamServer.__init__(self, *args, **kwargs)
else:  # pragma: no cover - a host without AF_UNIX serves no wake door
    _UnixServer = None  # type: ignore[assignment,misc]


class _TcpServer(_Bounded, socketserver.ThreadingTCPServer):
    """Loopback TCP, for a test host without AF_UNIX only; production never uses it."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        _Bounded.__init__(self)
        socketserver.ThreadingTCPServer.__init__(self, *args, **kwargs)


Address = Union[str, Tuple[str, int]]


class LlmServerWakeResponder:
    """The wake door, started only where the LLM Server can front a cluster.

    ``address`` is :data:`WAKE_SOCKET` in production (a unix socket in the
    control plane's own folder); a ``(host, port)`` pair is accepted for a test
    host without AF_UNIX. ``stop`` shuts the server and removes the socket.
    """

    def __init__(
        self, *, wake: Callable[[], Any], fingerprints: Callable[[], Sequence[str]],
        address: Address = WAKE_SOCKET,
    ):
        self._wake = wake
        self._fingerprints = fingerprints
        self._address = address
        self._server: Any = None

    @property
    def address(self) -> Any:
        return self._server.server_address if self._server else None

    def start(self) -> bool:
        if self._server is not None:
            return True
        try:
            if isinstance(self._address, str):
                if _UnixServer is None:
                    return False
                _prepare_socket_dir(self._address)
                server = _UnixServer(self._address, _Handler)
                os.chmod(self._address, 0o666)
            else:
                server = _TcpServer(self._address, _Handler)
        except OSError as error:
            LOGGER.warning(
                "The LLM Server wake door could not listen at %s (%s); an LLM "
                "Server client cannot wake an idle cluster model until the "
                "control plane restarts.", self._address, error,
            )
            return False
        server.wake = self._wake
        server.fingerprints = self._fingerprints
        self._server = server
        threading.Thread(
            target=server.serve_forever, name="vaelor-llm-server-wake", daemon=True,
        ).start()
        return True

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if isinstance(self._address, str):
            try:
                os.unlink(self._address)
            except OSError:
                pass


def _prepare_socket_dir(path: str) -> None:
    """Make the socket's folder this account's own, and clear a stale socket.

    Refuses (``PermissionError``) a folder another account already made: the
    folder's owner is what the root bridge trusts, so a pre-planted one must
    never be adopted.
    """
    folder = os.path.dirname(path)
    try:
        os.mkdir(folder, 0o755)
    except FileExistsError:
        pass
    info = os.lstat(folder)
    if not stat.S_ISDIR(info.st_mode) or (
        hasattr(os, "getuid") and info.st_uid != os.getuid()
    ):
        raise PermissionError("the wake door's folder is not this account's own")
    os.chmod(folder, 0o755)
    try:
        if stat.S_ISSOCK(os.lstat(path).st_mode):
            os.unlink(path)
    except FileNotFoundError:
        pass


def wake_door_ready(path: str = WAKE_SOCKET, timeout: float = 2.0) -> bool:
    """Whether THIS responder answers on ``path`` - asked before the door points here.

    A socket that answers ``/health`` with the responder's own code; anything
    else (absent, refused, another answer) is not ready, and the door stays down.
    """
    family = getattr(socket, "AF_UNIX", None)
    if family is None:
        return False
    try:
        with socket.socket(family, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(path)
            conn.sendall(b"GET /health HTTP/1.0\r\nHost: wake\r\n\r\n")
            reply = b""
            while len(reply) < 8192:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                reply += chunk
    except OSError:
        return False
    return reply.startswith(b"HTTP/1.0 503") and HEALTH_CODE.encode() in reply


def wake_socket_refusal(
    folder: str = WAKE_SOCKET_DIR, lstat: Callable[[str], Any] = os.lstat,
    owner_uid: Optional[int] = None,
) -> Optional[str]:
    """Why the root bridge must not mount the wake folder into the gate, or ``None``.

    The folder must be a real folder owned by the control plane's account and
    writable by nobody else, and the socket in it a socket of that account:
    what the gate forwards keyed requests to is then the control plane's own
    responder and nothing another account planted. ``owner_uid`` defaults to
    :data:`WAKE_SOCKET_OWNER`'s.
    """
    if owner_uid is None:
        try:
            import pwd

            owner_uid = pwd.getpwnam(WAKE_SOCKET_OWNER).pw_uid
        except (ImportError, KeyError):
            return "The control plane's account does not exist on this machine."
    try:
        info = lstat(folder)
        sock = lstat(folder + "/" + WAKE_SOCKET_NAME)
    except OSError:
        return "The LLM Server wake door is not running on this machine."
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != owner_uid or (
        info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        return (
            "The wake door's folder {} is not the control plane's own, so the "
            "LLM Server is not pointed at it.".format(folder)
        )
    if not stat.S_ISSOCK(sock.st_mode) or sock.st_uid != owner_uid:
        return "The wake door's socket is not the control plane's own."
    return None


def start_wake_door(
    *, wake: Callable[[], Any], fingerprints: Callable[[], Sequence[str]],
    bridge_available: bool, address: Address = WAKE_SOCKET,
) -> Optional[LlmServerWakeResponder]:
    """Start the door where it can be used, else nothing (review nit).

    Only a box with the root hardware bridge can run the LLM Server's gate at
    all, so elsewhere (a development host, a box without the bridge) nothing
    is started and no socket is made. The door stops when the process exits.
    """
    if not bridge_available or (isinstance(address, str) and _UnixServer is None):
        return None
    responder = LlmServerWakeResponder(wake=wake, fingerprints=fingerprints, address=address)
    if not responder.start():
        return None
    import atexit

    atexit.register(responder.stop)
    return responder


def _spawn_daemon(target: Callable[[], None]) -> None:
    threading.Thread(target=target, name="vaelor-llm-server-wake-wait", daemon=True).start()


def start_control_plane_wake_door(
    runtime: Any, *,
    bridge_up: Optional[Callable[[], bool]] = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    spawn: Callable[[Callable[[], None]], None] = _spawn_daemon,
) -> Optional[LlmServerWakeResponder]:
    """The control plane's wiring, out of `control_plane_runtime` for its line budget.

    It wakes with the ``ai-chat`` purpose, not ``cluster-inference``: an idle
    unload leaves AI Chat on the cluster and a manual unload parks it, so this
    is what keeps a manually unloaded model from being woken from the LAN. The
    key set is the broker's fingerprint-only listing of the LLM Server's
    current keys (no decrypt), read per request.

    **The bridge is waited for, not sampled once (ACC-165).** Whether this
    machine has a root bridge is read off its socket. The control plane and
    the bridge are both ``Type=simple`` units, so "start after the bridge"
    orders the two processes and says nothing about when the socket exists:
    on a reboot the control plane could look before the bridge had bound it,
    start no responder, and never look again - an idle-unloaded model then
    could not be woken through the LLM Server until the control plane was
    restarted by hand. A socket already there starts the responder at once
    and returns it; otherwise a daemon thread looks again every
    :data:`BRIDGE_POLL_SECONDS` for :data:`BRIDGE_WAIT_SECONDS`, starts the
    responder when the socket appears and hands it to ``runtime``, and says
    plainly when it gives up. The seams are for the test: production passes
    none of them.
    """
    from .gpu_idle_watch import LLM_SERVER_WAKE
    from .hardware_bridge_client import SOCKET_PATH
    from .llm_server_state import active_key_fingerprints

    if bridge_up is None:
        bridge_up = lambda: os.path.exists(SOCKET_PATH)  # noqa: E731 - read at call time

    def start() -> Optional[LlmServerWakeResponder]:
        return start_wake_door(
            wake=lambda: runtime._cluster_wake(LLM_SERVER_WAKE),
            fingerprints=lambda: active_key_fingerprints(runtime.credential_broker),
            bridge_available=True,
        )

    if bridge_up():
        return start()

    def wait_for_the_bridge() -> None:
        deadline = monotonic() + BRIDGE_WAIT_SECONDS
        while monotonic() < deadline:
            sleep(BRIDGE_POLL_SECONDS)
            if bridge_up():
                runtime.llm_server_wake = start()
                return
        LOGGER.info(
            "No root hardware bridge appeared within %d s of the control plane "
            "starting, so the LLM Server's wake door was not started; this "
            "machine runs no LLM Server gate.", BRIDGE_WAIT_SECONDS,
        )

    spawn(wait_for_the_bridge)
    return None
