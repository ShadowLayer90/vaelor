"""The replica balancer: the nginx in front of a replicated deployment's replicas.

Housed out of `gpu_pool_replicas` (which sat within a few lines of the
ceiling) with everything it needs and nothing else: the pool derived from a
record, the root-side config renderer and process, and the executor-side
controller that converges it. `gpu_pool_replicas` re-exports every name here,
so its callers and its tests are unchanged.

**Two processes run this file** (LESSONS pattern 14): the ROOT HARDWARE BRIDGE
runs :class:`BalancerProcess` and :func:`render_balancer_config` behind its
``balancer_start``/``stop``/``status`` verbs; the WORKLOAD EXECUTOR runs
:class:`BalancerController`, which only ever asks the bridge.

**Every balancer relaunch cuts every stream in flight through it.** The launch
replaces the container (`LlmServerProxyProcess._launch`), so a relaunch is
never a reload: :meth:`BalancerController.converged` must say "converged" for a
balancer that is already right, or each reconcile pass would cut every AI Chat
answer being streamed at that moment.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .cluster_placement import CONTROLLER_PLACEMENT_NAME
from .gpu_pool_units import balancer_name, deployment_name, unstopped
from .gpu_serving_target import (
    CLUSTER_INFERENCE_PURPOSE, SERVER_LOOPBACK_HOST, serving_port,
)
from .llm_server_proxy import (
    LlmServerProxyProcess, replica_locations, validate_api_key,
)
from .runtime_paths import run_path
from .served_endpoint_keys import served_key_fingerprint
from .session_affinity import NGINX_SESSION_VARIABLE, SESSION_HEADER, SESSION_KEY_PATTERN

LOGGER = logging.getLogger(__name__)

#: The nginx upstream pool's name inside the balancer config.
BALANCER_UPSTREAM = "vaelor_replicas"

#: The nginx variable the pool hashes on (VD-157): a valid ``X-Session-Id``
#: when the request carries one, else nginx's own per-request id. Session
#: stickiness is what lets a conversation reuse one replica's prompt cache:
#: measured on the pair (2026-09-30), turns two to five of a conversation took
#: 3.7-6.4 s to the first word when ``least_conn`` alternated them between
#: replicas and 1.5-1.7 s hashed onto one.
BALANCER_AFFINITY_VARIABLE = "$vaelor_affinity"

#: What a request with no valid session key is hashed on: ``$request_id``, 16
#: random bytes nginx makes per request. So keyless traffic lands all over the
#: hash ring and both replicas serve it - never an empty string or a constant,
#: which would put every keyless request on ONE replica (the defect the shared
#: zone fixed, back by another door), and never the client's address, which
#: would pin a whole LAN app to one machine.
BALANCER_KEYLESS_VALUE = "$request_id"

#: The shared-memory zone the pool's state lives in, named after the pool.
#: Without it every nginx worker process (``worker_processes auto`` in the
#: image: 32 on the controller) kept its own connection counts and its own
#: cursor over tied servers, so at light load each worker's first pick among
#: equals was the first server and the second replica served nothing (live,
#: 2026-09-29: 0 requests to the worker in 14 h). With the session hash the
#: zone still carries the one thing every worker must agree on: which replica
#: is marked failed. 64k holds the state of far more servers than a pool of
#: one replica per machine will ever have.
BALANCER_ZONE_SIZE = "64k"

#: How long the balancer waits to CONNECT to a replica before trying the next
#: one. nginx's default is 60 s, so a worker that dropped off the network
#: without refusing (unplugged, suspended) held each request it was picked
#: for a full minute; 5 s is long for a LAN connect and short for a person.
BALANCER_CONNECT_TIMEOUT = "5s"

#: When the balancer gives up on a replica and tries the next: a connection
#: that failed, was dropped unanswered (a worker's gate closes with 444 when
#: its replica is down) or timed out, and a gateway status. Each marks that
#: replica failed for nginx's ``fail_timeout`` (10 s by default), in the
#: shared zone, so every worker stops picking it at once.
#:
#: ``non_idempotent`` (owner decision, 2026-09-29, VD-149 amendment): nginx
#: may send a POST - every chat completion - to the other replica after it
#: was already sent. Without it the live failover test lost 3 of 20 POSTs:
#: nginx still tries a failed replica once per ``fail_timeout`` with a live
#: request, the gate dropped it, and the client got nginx's 502. The resend
#: is bounded by :data:`BALANCER_RESEND_TRIES` and :data:`BALANCER_RESEND_WINDOW`:
#: once, and only when the first attempt failed before any reply and within
#: 6 s of starting. A replica may have begun a short generation, which is then
#: repeated on the other (duplicate work, never duplicate output to the
#: client). A failure after 6 s is answered with its error, not resent.
BALANCER_NEXT_UPSTREAM = "error timeout http_502 http_503 http_504 non_idempotent"

#: Attempts per request, the first included: ONE resend at most, never a
#: tour of the pool.
BALANCER_RESEND_TRIES = 2

#: How long after the FIRST attempt began a failed request may still be sent
#: to the other replica (``proxy_next_upstream_timeout``). The flags cannot
#: tell "never reached a replica" from "a replica was working on it": nginx
#: files a connect timeout and a read timeout under the same ``timeout``, and
#: a replica that dies mid-generation fails as ``error`` just as a refused
#: connection does. Time can. A refusal and a gate's 444 come back in
#: milliseconds and a connect gives up at :data:`BALANCER_CONNECT_TIMEOUT`
#: (5 s), so 6 s admits every "never reached" case. It also admits a
#: replica that accepted the request and died within 6 s - tokenizing,
#: prefilling, or a short unstreamed generation - whose work is then done
#: again on the other replica: duplicate GPU work, never a duplicate answer,
#: because the first attempt delivered nothing. A request that fails later
#: is answered with its error, not resent. The read timeout (3600 s,
#: `STREAMING_DIRECTIVES`) is far outside the window, so a long generation
#: is never resent, and nginx resends nothing once a reply has begun, so a
#: stream is never replayed.
BALANCER_RESEND_WINDOW = "6s"

#: The record key the mode watch writes while the root bridge runs the
#: balancer from another release than the executor's (S1): the one fact the
#: console reads, so the log is not the only place it is said.
BALANCER_OTHER_RELEASE = "balancer_other_release"

#: What the console says of it, beside the balancer's state.
BALANCER_OTHER_RELEASE_NOTE = (
    "The replica balancer is still running the routing of a different Vaelor "
    "release than the rest of this appliance, so conversations are not yet "
    "kept on one machine. Restart Vaelor's services, or the appliance, to "
    "finish the upgrade."
)

#: What the key is rendered as when the config is digested: the digest is of
#: the config's SHAPE, and must never be a function of the key.
_DIGEST_KEY_BLANK = "<cluster key>"


def replica_entries(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """``units.replicas`` of a stored record, as a list of dicts, else empty."""
    units = record.get("units") or {}
    replicas = units.get("replicas") if isinstance(units, Mapping) else None
    return [dict(entry) for entry in (replicas or []) if isinstance(entry, Mapping)]


def replica_upstreams(record: Mapping[str, Any]) -> List[str]:
    """The balancer's ``host:port`` pool, derived from the record's replicas.

    The controller's replica on loopback, and each worker's GATE on the
    worker's cluster address (the replica behind it is loopback-only) -
    exactly the ``address`` each ``units.replicas[]`` entry recorded when the
    deploy started it, so the balancer and the record cannot name different
    machines.
    """
    return [
        "{}:{}".format(entry.get("address", ""), int(entry.get("port", 0) or 0))
        for entry in replica_entries(record)
    ]


def _validate_upstream(value: Any) -> str:
    """``host:port`` for a replica: a loopback or private IPv4 in the serving band.

    Rendered into the balancer's ``upstream`` block root-side, so it admits
    only address-shaped text: the record it came from is control-plane
    writable, and a hostname here would be a DNS lookup nginx makes as root.
    """
    text = str(value or "").strip()
    host, separator, port = text.rpartition(":")
    if not separator:
        raise ValueError("A balancer upstream must be 'host:port'.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("A balancer upstream must name an IPv4 address.") from error
    if address.version != 4 or not (address.is_loopback or address.is_private):
        raise ValueError(
            "A balancer upstream must be a loopback or private IPv4 address."
        )
    return "{}:{}".format(address, serving_port(port))


def render_balancer_config(
    *, port: int, upstreams: Sequence[str], api_key: str
) -> str:
    """The nginx config that pools the replicas behind ``127.0.0.1:<port>``.

    * ``listen 127.0.0.1:<port>``: loopback only, so the LLM Server proxy is
      the only LAN door, exactly as for the distributed head;
    * one ``server`` per replica under ``hash $vaelor_affinity consistent``
      (VD-157): a request that names its conversation in a valid
      ``X-Session-Id`` always reaches the same replica while it is up, so that
      replica's prompt cache is reused; any other request is hashed on a value
      of its own and spread (:func:`_affinity_map`). ``consistent`` keeps every
      other session where it was when a replica leaves or returns. The pool's
      state is in a shared ``zone`` so every nginx worker reads ONE failure
      state rather than its own (:data:`BALANCER_ZONE_SIZE`); the worker count
      stays the image's, because this file is mounted inside the image's
      ``http`` block where ``worker_processes`` is not allowed;
    * the ``/health``/``/v1/``/catch-all path map is
      `llm_server_proxy.replica_locations` - THE SAME ONE a worker's gate
      renders (the final VD-129 review's S2 finding: before this, the
      balancer forwarded every path to the pool while a gate 404'd everything
      but ``/health`` and ``/v1/``, so a keyed client's request to a side
      door like ``/tokenize`` got a different answer depending on which
      replica the balancer picked);
    * ``proxy_connect_timeout`` and ``proxy_next_upstream`` at SERVER level,
      so ``/health`` and ``/v1/`` inherit the one rule
      (:data:`BALANCER_CONNECT_TIMEOUT`, :data:`BALANCER_NEXT_UPSTREAM`): a
      replica that cannot be reached is skipped and marked failed, and a
      request is sent to the other replica at most once
      (:data:`BALANCER_RESEND_TRIES`): only when the first attempt failed
      before any reply and within :data:`BALANCER_RESEND_WINDOW` of starting.
      A replica may have begun a short generation, which is then repeated on
      the other (duplicate work, never duplicate output to the client); a
      failure after the window is answered with its error, not resent. The
      resend goes to the OTHER replica: the hash walks on round its ring past
      the replica already tried for this request;
    * ``proxy_set_header Authorization "Bearer <key>"`` on ``/v1/``: the one
      key opens every worker's gate, and the keyless controller replica
      ignores it. The key is validated by the proxy's own rule so it cannot
      break out of the quoted string, and lives only in this root-owned
      ``0600`` file.
    """
    return _render(port=port, upstreams=upstreams, key=validate_api_key(api_key))


def balancer_config_digest(*, port: int, upstreams: Sequence[str]) -> str:
    """A short digest of the config :func:`render_balancer_config` renders, key blanked.

    The root side reports it for the RUNNING balancer and the controller
    computes it for the record, so a balancer running an older rendering of
    the same pool (a fix like the shared zone, shipped while it ran) is seen
    as drift and replaced once. Never a function of the key: the key is
    compared by its own fingerprint.
    """
    text = _render(port=port, upstreams=upstreams, key=_DIGEST_KEY_BLANK)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _affinity_map() -> str:
    """The ``map`` that turns a request into the value the pool hashes on.

    The header is CLIENT-SUPPLIED, so it is judged here before it chooses
    anything: the one non-default row is the anchored pattern
    `session_affinity.SESSION_KEY_PATTERN` - the very string Python's
    ``valid_session_key`` compiles, so nginx and Python cannot admit different
    keys - and a value it does not match (empty, too long, a character outside
    the class, two headers nginx joined with a comma) falls to the default and
    is treated as absent. The pattern is rendered in double quotes because its
    length bound has braces; it carries no backslash and no variable, so
    nginx reads it exactly as written. The value is never written to a log:
    this config declares no ``log_format`` and no ``access_log``.
    """
    return (
        "# Which replica a request goes to. One that carries a valid\n"
        "# {header} keeps to one replica, so a conversation reuses that\n"
        "# replica's prompt cache; any other request gets a value of its own\n"
        "# and is spread. The header's value is never written to a log.\n"
        "map {source} {affinity} {{\n"
        "    default {keyless};\n"
        "    \"~{pattern}\" {source};\n"
        "}}\n"
    ).format(
        header=SESSION_HEADER, source=NGINX_SESSION_VARIABLE,
        affinity=BALANCER_AFFINITY_VARIABLE, keyless=BALANCER_KEYLESS_VALUE,
        pattern=SESSION_KEY_PATTERN,
    )


def _render(*, port: int, upstreams: Sequence[str], key: str) -> str:
    """The one rendering, with ``key`` already validated (or the digest's blank)."""
    listen = serving_port(port)
    pool = [_validate_upstream(item) for item in upstreams]
    if not pool:
        raise ValueError("A balancer needs at least one replica upstream.")
    servers = "".join("    server {};\n".format(item) for item in pool)
    v1_prefix = (
        "        # The one cluster key opens every worker's gate.\n"
        "        proxy_set_header Authorization \"Bearer {key}\";\n"
    ).format(key=key)
    locations = replica_locations(
        upstream="http://{}".format(BALANCER_UPSTREAM), v1_prefix=v1_prefix,
    )
    return (
        "{affinity_map}"
        "\n"
        "upstream {upstream} {{\n"
        "    # One shared-memory copy of the pool's state for every nginx worker.\n"
        "    zone {upstream} {zone_size};\n"
        "    # The same session reaches the same replica while it is up.\n"
        "    hash {affinity} consistent;\n"
        "{servers}"
        "}}\n"
        "\n"
        "server {{\n"
        "    listen {loopback}:{port};\n"
        "    server_name _;\n"
        "    # A replica that cannot be reached is given up on in seconds.\n"
        "    proxy_connect_timeout {connect};\n"
        "    # Skip a replica that is down and mark it failed. A request is sent\n"
        "    # to the other replica at most once: only if the first attempt failed\n"
        "    # before any reply and within {window} of starting. A short generation it\n"
        "    # began is then repeated (duplicate work, never duplicate output);\n"
        "    # a failure after {window} is answered with its error, not resent.\n"
        "    proxy_next_upstream {next_upstream};\n"
        "    proxy_next_upstream_tries {tries};\n"
        "    proxy_next_upstream_timeout {window};\n"
        "\n"
        "{locations}"
        "}}\n"
    ).format(
        affinity_map=_affinity_map(), affinity=BALANCER_AFFINITY_VARIABLE,
        upstream=BALANCER_UPSTREAM, zone_size=BALANCER_ZONE_SIZE, servers=servers,
        loopback=SERVER_LOOPBACK_HOST, port=listen,
        connect=BALANCER_CONNECT_TIMEOUT, next_upstream=BALANCER_NEXT_UPSTREAM,
        tries=BALANCER_RESEND_TRIES, window=BALANCER_RESEND_WINDOW,
        locations=locations,
    )


def balancer_config_path(name: str) -> str:
    """The balancer's root-rendered config, under ``/run/vaelor`` like the proxy's."""
    return run_path("vllm-balancer-{}.conf".format(deployment_name(name)))


class BalancerProcess(LlmServerProxyProcess):
    """The balancer container, launched and stopped ROOT-side by the bridge.

    The proxy process with another config, another name and its own binding
    facts; the launch, the replacement, the read-only mount and the key
    coupling are inherited unchanged, so the two nginx containers the
    controller runs cannot drift on how they are run.

    Its status carries two facts about what the RUNNING container was given,
    never the key: the config's digest with the key blanked
    (:func:`balancer_config_digest`) and the key's fingerprint
    (`served_endpoint_keys.served_key_fingerprint`), so the controller can
    tell a stale rendering or a stale key from a right balancer.
    """

    label = "replica balancer"

    def __init__(self, name: str, *, run: Optional[Callable[..., Any]] = None,
                 docker: Optional[str] = None):
        self.name = deployment_name(name)
        super().__init__(
            run=run, docker=docker, container_name=balancer_name(self.name),
            config_host_file=balancer_config_path(self.name),
        )
        self.port = 0
        self.upstreams: List[str] = []
        self.config_digest = ""
        self.key_fingerprint = ""

    def start(self, *, port: int, upstreams: Sequence[str], api_key: str) -> dict:  # type: ignore[override]
        """Render and (re)launch. A running balancer is REPLACED: every stream
        in flight through it is cut, so the controller calls this only on drift."""
        config = render_balancer_config(
            port=port, upstreams=upstreams, api_key=api_key,
        )
        self._launch(config, listen_port=int(port), api_keys=[api_key])
        self.port = serving_port(port)
        self.upstreams = [_validate_upstream(item) for item in upstreams]
        self.config_digest = balancer_config_digest(port=port, upstreams=upstreams)
        self.key_fingerprint = served_key_fingerprint(str(api_key))
        return self.status()

    def stop(self, **kwargs: Any) -> dict:
        self.config_digest = ""
        self.key_fingerprint = ""
        return super().stop(**kwargs)

    def status(self) -> dict:
        return {
            "running": self.alive(), "name": self.name, "port": self.port,
            "upstreams": list(self.upstreams), "container": self._name,
            "config_digest": self.config_digest,
            "key_fingerprint": self.key_fingerprint,
        }


class BalancerController:
    """Converge the balancer to a replicated record, over the bridge.

    The executor-side seam, the twin of `LlmServerProxyController`: it holds a
    bridge-client-like object with ``balancer_start``/``balancer_stop``/
    ``balancer_status`` (production: the root `HardwareBridgeClient`; a test: a
    recording fake) and turns a record into the one right call. `converged`
    reads the live status - running, on the record's port, pooling the
    record's replicas, with the config this code renders for them and the
    cluster credential's key - so the reconcile applies only what is not
    already right, the way the proxy's `converged` stopped the LAN gate
    flapping once a pass (VD-127).

    ``broker`` (optional) is how the key is judged: the credential broker's
    fingerprint-only ``list()`` each pass, and a ``resolve`` only when the
    credential's fingerprint moved since this controller last read it (a
    rotation), so a steady pass decrypts nothing. Without a broker the key is
    not judged.
    """

    def __init__(self, client: Any, *, broker: Any = None):
        self._client = client
        self._broker = broker
        # Whether "this bridge reports no digest" has been logged, so a bridge
        # from before the digest is said once, not every 30 s pass.
        self._said_old_bridge = False
        # credential id -> (the broker's fingerprint of the credential, the
        # fingerprint of the key it held then). Process memory: a restarted
        # executor resolves once per credential and remembers again.
        self._key_memo: Dict[str, Tuple[str, str]] = {}
        # deployment name -> (the digest this code wants, the digest the
        # bridge rendered when asked for it). Present only while the bridge is
        # on another release and so cannot render the wanted config; see
        # `_note_rendering`. Process memory, like the key memo.
        self._other_release: Dict[str, Tuple[str, str]] = {}

    def start(self, record: Mapping[str, Any], api_key: str) -> dict:
        units = record.get("units") or {}
        status = self._client.balancer_start(
            str(record.get("name", "")), int(units.get("port", 0) or 0),
            replica_upstreams(record), api_key,
        )
        self._note_rendering(record, status)
        return status

    def release_skewed(self, name: str) -> bool:
        """Whether ``name``'s balancer was last started by a bridge on another release.

        This process's own fact (`_note_rendering`), read by the mode watch
        so it can be written on the row for the console.
        """
        return str(name) in self._other_release

    def _note_rendering(self, record: Mapping[str, Any], status: Any) -> None:
        """Remember when the bridge CANNOT render the config this code wants.

        The config is rendered root-side, by the bridge's own copy of this
        module, and judged here. While the bridge and the executor are on
        different releases - one restarted on an upgrade, the other not yet -
        the bridge renders its release's config, whose digest is not the one
        wanted and never will be. Judging the digest then relaunched the
        balancer on every 30 s pass, and each relaunch cuts every answer being
        streamed through it (2026-09-30 review, S1).

        A start that comes back with another digest is that proof. It is
        remembered against the pair (wanted, rendered), said once in plain
        words, and `converged` stops judging the digest for as long as the
        bridge keeps reporting that same rendering. A start that comes back
        as wanted clears it. A status with no digest at all is the older
        bridge `converged` already handles.
        """
        name = str(record.get("name", ""))
        if not isinstance(status, Mapping) or "config_digest" not in status:
            return
        units = record.get("units") or {}
        try:
            wanted = balancer_config_digest(
                port=int(units.get("port", 0) or 0), upstreams=replica_upstreams(record),
            )
        except (TypeError, ValueError):
            return
        rendered = str(status.get("config_digest") or "")
        if rendered == wanted:
            self._other_release.pop(name, None)
            return
        if self._other_release.get(name) != (wanted, rendered):
            LOGGER.warning(
                "The root bridge started the replica balancer for %s with a "
                "configuration from a different release than this executor's, "
                "so it is left running as the bridge made it and its "
                "configuration is not checked again until the bridge reports "
                "another. Restart vaelor-hardware-bridge and "
                "vaelor-workload-executor together on the current release.",
                name,
            )
        self._other_release[name] = (wanted, rendered)

    def stop(self, name: str) -> dict:
        return self._client.balancer_stop(str(name))

    def status(self, name: str) -> Mapping[str, Any]:
        return self._client.balancer_status(str(name))

    def converged(self, record: Mapping[str, Any]) -> bool:
        """Whether the LIVE balancer already serves ``record`` exactly.

        Running, on the record's port, pooling the record's replicas, with the
        config digest this code renders for them and the key the cluster
        credential holds. Fail-safe like the proxy's on the balancer's own
        facts: an unreadable status is "not converged", so the caller applies
        rather than trusts. An unreadable BROKER is the opposite: the key is
        then not judged, because a relaunch cuts every stream in flight and a
        broker that stays down must not cost AI Chat an answer every pass.

        A status with NO ``config_digest`` comes from a root bridge older than
        this check (the executor was updated, the bridge not yet restarted).
        That bridge renders its OWN config, so relaunching through it can
        never produce this code's rendering or report a key fingerprint:
        judging the digest or the key would relaunch - and cut every stream -
        every pass, forever. It is said once and only running, port and pool
        are judged until the bridge is current.

        A status whose digest is NOT the wanted one is drift - once. The
        relaunch that follows is compared with what was wanted (`start`), and
        a bridge that rendered something else again is on another release
        than this executor: its digest is then left unjudged until it changes,
        so a half-finished upgrade costs one relaunch, not one every pass.

        The key is judged for the record's ``credential_id`` - the id the
        deploy recorded on the row AND handed to the switch's ``repoint`` in
        the same step (`GpuPoolOperations.deploy`), so it is the one the
        reconcile's ``state.cluster_credential_id`` resolves when it starts
        the balancer; a rotation re-keys that credential in place.
        """
        units = record.get("units") or {}
        try:
            status = self.status(str(record.get("name", "")))
        except Exception:  # noqa: BLE001 - an unreadable balancer is an unconverged one
            return False
        try:
            port = int(status.get("port") or 0)
        except (TypeError, ValueError):
            return False
        pool = replica_upstreams(record)
        if not (
            bool(status.get("running"))
            and port == int(units.get("port", 0) or 0)
            and sorted(str(item) for item in (status.get("upstreams") or []))
            == sorted(pool)
        ):
            return False
        if "config_digest" not in status:
            if not self._said_old_bridge:
                self._said_old_bridge = True
                LOGGER.warning(
                    "The root bridge reports no balancer config digest; its "
                    "config and key are not checked until it is restarted on "
                    "the current release."
                )
            return True
        try:
            wanted_digest = balancer_config_digest(port=port, upstreams=pool)
        except ValueError:
            return False
        rendered = str(status.get("config_digest") or "")
        # A bridge on another release was already asked for the wanted config
        # and rendered this one instead (`_note_rendering`): asking again
        # would only cut the streams in flight. Its key is still judged.
        skewed = self._other_release.get(str(record.get("name", "")))
        if rendered != wanted_digest and skewed != (wanted_digest, rendered):
            return False
        wanted_key = self._wanted_key_fingerprint(record)
        return wanted_key is None or str(status.get("key_fingerprint") or "") == wanted_key

    def _wanted_key_fingerprint(self, record: Mapping[str, Any]) -> Optional[str]:
        """The fingerprint of the key the balancer should stamp, or ``None`` if unknowable."""
        credential_id = str(record.get("credential_id", "") or "")
        if self._broker is None or not credential_id:
            return None
        try:
            listed = {
                str(item.get("id", "")): str(item.get("fingerprint", "") or "")
                for item in (self._broker.list() or [])
            }
        except Exception as error:  # noqa: BLE001 - see `converged`
            LOGGER.warning("The balancer's key could not be checked: %s", error)
            return None
        credential_fingerprint = listed.get(credential_id, "")
        if not credential_fingerprint:
            return None
        memo = self._key_memo.get(credential_id)
        if memo is not None and memo[0] == credential_fingerprint:
            return memo[1]
        try:
            lease = self._broker.resolve(credential_id, CLUSTER_INFERENCE_PURPOSE)
        except Exception as error:  # noqa: BLE001 - see `converged`
            LOGGER.warning("The balancer's key could not be read: %s", error)
            return None
        key = str((lease or {}).get("api_key", "") or "")
        if not key:
            return None
        wanted = served_key_fingerprint(key)
        self._key_memo[credential_id] = (credential_fingerprint, wanted)
        return wanted


def default_balancer() -> BalancerController:
    """The production controller: over the root bridge, like the proxy's.

    The one wiring, used by `GpuPoolOperations` and by the mode switch's
    executor wiring alike; imported inside the call so importing this module
    opens no bridge socket, and both clients connect per request, so
    constructing it touches nothing. The broker is how `converged` judges
    the key the running balancer stamps.
    """
    from .credential_broker_client import CredentialBrokerClient
    from .hardware_bridge import HardwareBridgeClient

    return BalancerController(HardwareBridgeClient(), broker=CredentialBrokerClient())


def stop_balancer(balancer: BalancerController, name: str) -> List[Dict[str, str]]:
    """Stop ``name``'s balancer, never raising: what would not stop, as `unstopped`.

    The balancer is a container on THIS controller, not a unit on a node, so
    its failure entry carries no node id: `gpu_pool_units.controller_unstopped`
    keys the "leave AI Chat unassigned" rule on the controller's node id
    because that rule is about GPU residency, and a lingering nginx holds no
    GPU. The entry still names the container and the controller, so the row
    says what is running and where.
    """
    try:
        balancer.stop(name)
    except Exception as error:  # noqa: BLE001 - never mask the cause
        LOGGER.warning("Could not stop the replica balancer for %s: %s", name, error)
        return [unstopped("", CONTROLLER_PLACEMENT_NAME, balancer_name(name), error)]
    return []
