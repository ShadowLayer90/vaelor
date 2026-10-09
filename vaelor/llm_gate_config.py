"""The nginx configs of Vaelor's keyed doors, rendered and nothing else.

Housed out of `llm_server_proxy` (which sat three lines under the 1,000-line
ceiling `CLAUDE.md` sets) as the one cohesive piece it carried that launches
nothing: the text of a config. `llm_server_proxy` re-exports every name here,
so its callers and its tests are unchanged, and keeps what runs a container -
the argv, the root-owned file, the process and the controller.

Three doors are rendered here, from one Bearer check and one path map:

* the LLM Server proxy (:func:`render_proxy_config`), the LAN door in front of
  the serving model - or, while a cluster model is unloaded, in front of the
  control plane's wake responder, or of nothing at all, answering for itself
  (:func:`render_unloaded_config`, VD-159);
* a worker's replica gate (:func:`render_gate_config`);
* the path map the replica balancer shares with that gate
  (:func:`replica_locations`).

Pure functions of their arguments: no file, no socket, no docker.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

from .flm_service import _validate_port
from .llm_gate_usage import (
    GATE_KEY_VARIABLE,
    GATE_LOG_CONTAINER_DIR,
    GATE_LOG_FILENAME,
    GATE_LOG_FORMAT,
    GATE_LOG_FORMAT_NAME,
    GATE_REFUSED_FILENAME,
    GATE_REFUSED_FORMAT,
    GATE_REFUSED_FORMAT_NAME,
    GATE_REFUSED_VARIABLE,
)
from .llm_server_wake import (
    HEALTH_CODE, HEALTH_MESSAGE, WAKE_SOCKET_MOUNT, WAKE_SOCKET_NAME, error_body,
)
from .served_endpoint_keys import served_key_fingerprint

#: The loopback host the proxy forwards to (where the model container publishes).
MODEL_LOOPBACK_HOST = "127.0.0.1"

#: An upper bound on the key length, defence in depth behind the state store.
_MAX_API_KEY_LENGTH = 256

#: What a keyed client is told by the unloaded door (VD-159): the truth in
#: every unloaded state, whoever unloaded the model and whether or not
#: anything else could wake it.
UNLOADED_MESSAGE = (
    "The model behind the LLM Server is not loaded, and a request to this port "
    "cannot load it. Load it from Cluster > Deployments in the Vaelor console; "
    "this port answers normally once the model is serving."
)
UNLOADED_CODE = "model_unloaded"

#: What the same door says while the cluster model is being brought up
#: (ACC-193): the switch to cluster serving keeps port 11434 open with this
#: answer instead of closing it from the moment llama.cpp stops until the
#: cluster's gate starts. Nothing for the client to do but ask again.
LOADING_MESSAGE = (
    "The model behind the LLM Server is loading. Ask again shortly; this port "
    "answers normally once the model is serving."
)
LOADING_CODE = "model_loading"
LOADING_HEALTH_MESSAGE = "The model behind the LLM Server is loading."

#: The ``Retry-After`` the unloaded door sends with its 503, in seconds. The
#: door is re-judged on every pass of the mode reconcile, so this is the
#: soonest its answer can change; it is also the wait the gateway's own wake
#: answer names, so the appliance's doors tell a client one number.
UNLOADED_RETRY_AFTER_SECONDS = 30

#: Set by the unloaded door only AFTER the key check has passed, so the
#: ``Retry-After`` header is on the 503 and never on a 401: nginx leaves out
#: a header whose value is empty.
UNLOADED_RETRY_VARIABLE = "$vaelor_retry_after"


def validate_api_key(value: Any) -> str:
    """Return a proxy api key safe to place in an nginx config string, or raise.

    THE PROXY IS ONLY EVER BUILT WITH A KEY, so an empty value is refused here (the
    coupling lives at :func:`_require_keys`). The key is compared verbatim inside a
    double-quoted nginx ``if`` directive (``if ($http_authorization != "Bearer
    <key>")``), so any character that could break out of that string - whitespace,
    a control byte, a quote, a backslash, or nginx's ``$`` variable sigil - is
    refused. Vaelor's generated keys are ``vsk_`` + URL-safe base64 and pass; the
    check is defence in depth.

    Public because the replica balancer (`gpu_pool_replicas`, VD-129) renders
    the same key into a ``proxy_set_header``: one rule, so the balancer that
    sends the key and the two gates that check it cannot admit different keys.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError("The LLM Server proxy requires an api key.")
    if len(text) > _MAX_API_KEY_LENGTH:
        raise ValueError("The LLM Server proxy api key is too long.")
    if not re.fullmatch(r"[A-Za-z0-9._~+/=:\-]+", text):
        raise ValueError(
            "The LLM Server proxy api key carries a character that is unsafe in the "
            "nginx config (whitespace, a control byte, a quote, a backslash or '$') "
            "and is refused."
        )
    return text


def _require_keys(api_keys: Sequence[str]) -> List[str]:
    """THE SECURITY INVARIANT: an auth proxy is only ever started WITH a key set.

    The single home for the proxy/key coupling (LESSONS #178: the rule lives where
    the launch is built), generalised from one key to a SET (F3b): a proxy exists
    ONLY to gate the LAN behind a key, so building one from an EMPTY set - no keys
    at all - would defeat its entire purpose (the no-keyless-door invariant) and is
    refused here before any argv or config is produced, whichever path asks. A bare
    string is accepted as a one-key set. Called by :func:`bearer_check` and
    :func:`proxy_container_command`.
    """
    keys = [api_keys] if isinstance(api_keys, str) else list(api_keys)
    if not any(str(key).strip() for key in keys):
        raise ValueError(
            "SECURITY / LLM Server: refusing to build an auth proxy without an api "
            "key. The proxy is the ONLY LAN gate in front of the loopback model, so "
            "it must always carry at least one key that authenticates callers. Enable "
            "the LLM Server (which generates a key) rather than a keyless proxy."
        )
    return keys


#: How a proxied OpenAI location streams. nginx's defaults buffer a response
#: and time a read out at 60 s, either of which cuts an SSE token stream, so
#: every proxied location the product's three nginx configs carry - the LLM
#: Server proxy, a worker's gate (:func:`render_gate_config`) and the replica
#: balancer (`gpu_pool_replicas.render_balancer_config`) - is rendered with
#: these same lines rather than three spellings of them.
STREAMING_DIRECTIVES = (
    "        # OpenAI responses stream as SSE: no buffering, generous timeouts.\n"
    "        proxy_buffering off;\n"
    "        proxy_read_timeout 3600s;\n"
    "        proxy_send_timeout 3600s;\n"
)


def bearer_check(api_keys: Sequence[str], *, record_key: bool = False) -> str:
    """The nginx lines that reject anything but an exact ``Bearer <key>`` with 401.

    THE ONE key check, now over a SET (F3b, design B3): a request passes only when
    its ``Authorization`` header is exactly ``Bearer <k>`` for one of the endpoint's
    keys - matched BYTE-EXACT and CASE-SENSITIVE. nginx ``=`` on a string variable
    is a literal, case-sensitive comparison, unlike a ``map`` over plain-string
    keys, which lower-cases and would collapse the mixed-case ``vsk_`` entropy (a
    silent security regression). No key needs regex-escaping because ``=`` is a
    literal compare, not a ``~`` regex. An absent header and any wrong key fall to
    the default-deny 401. Each key is validated (:func:`validate_api_key`) so it
    cannot break out of the quoted string, and the coupling (:func:`_require_keys`)
    means a check is only ever rendered with a non-empty set. A SINGLE-key set
    renders the exact one-key gate the worker gate and LLM Server proxy shipped
    before F3b, byte-for-byte, so those two callers are behaviour-preserving.
    Indented for the ``location`` it lands in.

    ``record_key`` (the LLM Server proxy's usage log only) also sets
    :data:`~vaelor.llm_gate_usage.GATE_KEY_VARIABLE` to the matched key's vault
    fingerprint, inside the SAME byte-exact branch that admitted it - so the
    log names the key the gate actually matched, never a guess from the header,
    and nothing is set for a refused request. Off, the output is unchanged.
    """
    validated = []
    for key in _require_keys(api_keys):
        clean = validate_api_key(key)
        if clean not in validated:
            validated.append(clean)

    def mark(key: str, indent: str) -> str:
        if not record_key:
            return ""
        return '{indent}set {key_var} "{fp}";\n{indent}set {refused_var} "";\n'.format(
            indent=indent, key_var=GATE_KEY_VARIABLE, fp=served_key_fingerprint(key),
            refused_var=GATE_REFUSED_VARIABLE,
        )

    if len(validated) == 1:
        return (
            "        # The whole gate: anything but an exact Bearer <key> is rejected.\n"
            "        if ($http_authorization != \"Bearer {key}\") {{\n"
            "            return 401;\n"
            "        }}\n"
        ).format(key=validated[0]) + mark(validated[0], "        ")
    checks = "".join(
        (
            "        if ($http_authorization = \"Bearer {key}\") {{\n"
            "            set $vaelor_key_ok 1;\n"
            "{mark}"
            "        }}\n"
        ).format(key=key, mark=mark(key, "            "))
        for key in validated
    )
    return (
        "        # The whole gate: the Authorization header must be an exact\n"
        "        # Bearer <key> for one of this endpoint's keys, matched byte-exact\n"
        "        # and case-sensitively; anything else falls to the default-deny 401.\n"
        "        set $vaelor_key_ok 0;\n"
        "{checks}"
        "        if ($vaelor_key_ok = 0) {{\n"
        "            return 401;\n"
        "        }}\n"
    ).format(checks=checks)


def replica_locations(*, upstream: str, v1_prefix: str, v1_comment: str = "") -> str:
    """The ``/health``, ``/v1/`` and catch-all locations proxied to ``upstream``.

    THE ONE path map, shared byte-for-byte (apart from ``v1_prefix`` and
    ``v1_comment``) by a worker's gate (:func:`render_gate_config`) and the
    replica balancer (`gpu_pool_replicas.render_balancer_config`) - the final
    VD-129 review's S2 finding, fixed. Before this the balancer forwarded
    EVERY path to the pool while a worker's gate answered only ``/health``
    and ``/v1/`` and 404 to the rest: a keyed client asking ``/tokenize``,
    ``/invocations`` or ``/metrics`` through the balancer got 200 when
    ``least_conn`` picked the controller's keyless loopback replica and 404
    when it picked a worker's gate - one endpoint, two answers (LESSONS
    pattern 6). Both callers now proxy the SAME two paths and 404 everything
    else, from this one definition.

    ``v1_prefix`` is the one thing that must differ, rendered right inside
    ``location /v1/`` before ``proxy_pass``: a worker's gate checks the
    INCOMING request's key (:func:`bearer_check`), because it is the
    LAN-facing door; the balancer instead STAMPS the outgoing request with
    the key, because it already sits behind the keyed LAN proxy, loopback
    only, and checks no key of its own. ``v1_comment`` is an optional line
    (or lines) rendered just above ``location /v1/``.
    """
    return (
        "    # Unauthenticated health: proxied so it reflects the replica, and lets\n"
        "    # the controller's startup wait and mode watch see it without the key.\n"
        "    location = /health {{\n"
        "        proxy_pass {upstream}/health;\n"
        "        proxy_buffering off;\n"
        "    }}\n"
        "\n"
        "{v1_comment}"
        "    location /v1/ {{\n"
        "{v1_prefix}"
        "        proxy_pass {upstream};\n"
        "        proxy_http_version 1.1;\n"
        "        proxy_set_header Host $host;\n"
        "        proxy_set_header Connection \"\";\n"
        "{streaming}"
        "    }}\n"
        "\n"
        "    # Everything else - /invocations, /tokenize and /metrics, which vLLM's\n"
        "    # own key was live-probed to leave open - is not reachable from the LAN.\n"
        "    location / {{\n"
        "        return 404;\n"
        "    }}\n"
    ).format(
        upstream=upstream, v1_comment=v1_comment, v1_prefix=v1_prefix,
        streaming=STREAMING_DIRECTIVES,
    )


def _usage_log_lines(usage_log: bool) -> Tuple[str, str, str, str]:
    """``(head, server_start, health_log, request_log)`` for the usage log, or blanks.

    The LLM Server door's usage log, written once for the two configs that
    door can run (:func:`render_proxy_config`, :func:`render_unloaded_config`):
    the two ``log_format`` lines above the server block, the two variables
    every request starts with, ``/health`` kept out of the log, and the
    admitted and refused files of the keyed location.
    """
    if not usage_log:
        return "", "", "", ""
    head = (
        "# One line per ADMITTED request: the key's fingerprint, the status and\n"
        "# the seconds taken - never the Authorization header, a path or an\n"
        "# address. Refused requests go to their own, separately capped file.\n"
        "log_format {name} '{fields}';\n"
        "log_format {refused_name} '{refused_fields}';\n"
        "\n"
    ).format(
        name=GATE_LOG_FORMAT_NAME, fields=GATE_LOG_FORMAT,
        refused_name=GATE_REFUSED_FORMAT_NAME, refused_fields=GATE_REFUSED_FORMAT,
    )
    server_start = '    set {} "";\n    set {} "1";\n'.format(
        GATE_KEY_VARIABLE, GATE_REFUSED_VARIABLE,
    )
    request_log = (
        "        access_log {path} {name} if={variable};\n"
        "        access_log {refused_path} {refused_name} if={refused_variable};\n"
    ).format(
        path=GATE_LOG_CONTAINER_DIR + "/" + GATE_LOG_FILENAME,
        name=GATE_LOG_FORMAT_NAME, variable=GATE_KEY_VARIABLE,
        refused_path=GATE_LOG_CONTAINER_DIR + "/" + GATE_REFUSED_FILENAME,
        refused_name=GATE_REFUSED_FORMAT_NAME, refused_variable=GATE_REFUSED_VARIABLE,
    )
    return head, server_start, "        access_log off;\n", request_log


def _nginx_text(body: bytes) -> str:
    """A JSON body as the text of an nginx ``return``, inside single quotes.

    nginx expands ``$name`` in that text and ends it at a quote, so a body
    that carries ``$``, a single quote, a backslash or a line break is refused
    rather than rendered into something else.
    """
    text = body.decode("utf-8")
    if any(character in text for character in ("'", "$", "\\", "\n", "\r")):
        raise ValueError("The unloaded door's answer cannot be written into nginx.")
    return text


def render_unloaded_config(
    *, listen_host: str, listen_port: int, api_keys: Sequence[str],
    usage_log: bool = False, loading: bool = False,
) -> str:
    """The LLM Server door while the cluster model is unloaded (VD-159).

    The port stays open and keyed, and nginx itself answers: nothing is
    proxied, so the door depends on no model, no socket and no other process
    being up - which is what left the port with no listener after a reboot.

    * ``/health`` needs no key and answers 503 with a plain JSON body that
      says the model is unloaded;
    * any other request without a current key is refused 401, exactly as the
      serving door refuses it (:func:`bearer_check`, the one rendering);
    * a request WITH a current key is answered 503, a plain JSON error in the
      shape OpenAI clients read (:data:`UNLOADED_MESSAGE`) and
      ``Retry-After`` (:data:`UNLOADED_RETRY_AFTER_SECONDS`). The header's
      value is set only after the key check has passed, so a 401 never
      carries it;
    * both answers are JSON whatever the path looks like: an empty ``types``
      block leaves ``default_type`` as the only type, where nginx would
      otherwise call ``/v1/x.html`` text/html from its extension.

    Where a request CAN load the model - an idle unload, with the control
    plane's wake responder up - the switch runs the wake door instead
    (``render_proxy_config(wake_door=True)``).

    ``loading`` is the same door while the cluster model is being brought up
    (ACC-193): the same keyed 401 and 503 with ``Retry-After``, saying the
    model is loading (:data:`LOADING_MESSAGE`, :data:`LOADING_CODE`).
    """
    message, code, health = (
        (LOADING_MESSAGE, LOADING_CODE, LOADING_HEALTH_MESSAGE) if loading
        else (UNLOADED_MESSAGE, UNLOADED_CODE, HEALTH_MESSAGE)
    )
    gate = bearer_check(api_keys, record_key=usage_log)
    listen = _validate_bind_host(listen_host)
    listen_p = _validate_port(listen_port)
    head, server_start, health_log, request_log = _usage_log_lines(usage_log)
    return (
        "{head}"
        "server {{\n"
        "    listen {listen}:{listen_port};\n"
        "    server_name _;\n"
        "{server_start}"
        "    set {retry_variable} \"\";\n"
        "\n"
        "    # The cluster model behind this door is unloaded. Nothing is proxied:\n"
        "    # the door itself answers, and says so without a key on /health.\n"
        "    location = /health {{\n"
        "        types {{ }}\n"
        "        default_type application/json;\n"
        "{health_log}"
        "        return 503 '{health_body}';\n"
        "    }}\n"
        "\n"
        "    location / {{\n"
        "{gate}"
        "{request_log}"
        "        # A current key: the model is not loaded, and when to ask again.\n"
        "        set {retry_variable} \"{retry_after}\";\n"
        "        types {{ }}\n"
        "        default_type application/json;\n"
        "        add_header Retry-After {retry_variable} always;\n"
        "        return 503 '{body}';\n"
        "    }}\n"
        "}}\n"
    ).format(
        head=head, listen=listen, listen_port=listen_p, server_start=server_start,
        retry_variable=UNLOADED_RETRY_VARIABLE, retry_after=UNLOADED_RETRY_AFTER_SECONDS,
        health_log=health_log, gate=gate, request_log=request_log,
        health_body=_nginx_text(error_body(health, code, HEALTH_CODE)),
        body=_nginx_text(error_body(message, code, code)),
    )


def render_proxy_config(
    *, listen_host: str, listen_port: int, model_port: Optional[int],
    api_keys: Sequence[str], usage_log: bool = False, wake_door: bool = False,
    unloaded_notice: bool = False, loading: bool = False,
) -> str:
    """The nginx server block that gates the LAN behind ``Bearer <key>``.

    * ``listen <listen_host>:<listen_port>`` binds the host LAN interface (the
      container runs ``--network host``);
    * ``/health`` is unauthenticated and proxied, so a prober can confirm the tier
      without the key and see the model's own health;
    * every other request must present exactly ``Authorization: Bearer <key>`` or
      gets 401 (:func:`bearer_check`); a valid one is proxied to the model on
      loopback;
    * :data:`STREAMING_DIRECTIVES` keep OpenAI SSE token streaming flowing
      rather than buffered or cut off mid-stream.

    The plaintext key lives here, in a root-owned ``0600`` file mounted
    read-only - never on an argv.

    ``usage_log`` adds the usage log (see the module docstring): a
    ``log_format`` of the fingerprint, status and duration only, the admitted
    requests of ``location /`` written to the mounted log directory, and
    ``/health`` - polled by every reconcile pass - not logged at all.

    What the door fronts is one of three things, each named by its own
    marker and never by a sentinel port: the model on ``model_port``; the
    control plane's wake responder (``wake_door``); or nothing at all
    (``unloaded_notice``, :func:`render_unloaded_config`). The last two front
    no model port, and asking for both is refused.
    """
    if unloaded_notice:
        if wake_door or model_port is not None:
            raise ValueError("The LLM Server's unloaded door fronts nothing.")
        return render_unloaded_config(
            listen_host=listen_host, listen_port=listen_port, api_keys=api_keys,
            usage_log=usage_log, loading=loading,
        )
    if loading:
        raise ValueError("Only the door that fronts nothing can say the model is loading.")
    gate = bearer_check(api_keys, record_key=usage_log)
    listen = _validate_bind_host(listen_host)
    listen_p = _validate_port(listen_port)
    if wake_door:
        if model_port is not None:
            raise ValueError("The LLM Server wake door fronts no model port.")
        # The socket as the container sees it: nginx's ``unix:`` upstream, the
        # path ending at ``:`` before a URI (``proxy_pass`` form).
        socket = "unix:{}/{}".format(WAKE_SOCKET_MOUNT, WAKE_SOCKET_NAME)
        upstream = "http://" + socket
        health = "http://{}:/health".format(socket)
    else:
        upstream = "http://{}:{}".format(MODEL_LOOPBACK_HOST, _validate_port(model_port))
        health = upstream + "/health"
    head, log_start, health_log, request_log = _usage_log_lines(usage_log)
    return (
        "{head}"
        "server {{\n"
        "    listen {listen}:{listen_port};\n"
        "    server_name _;\n"
        "{log_start}"
        "\n"
        "    # Unauthenticated health: proxied so it reflects the model, and lets a\n"
        "    # prober confirm the gate is up without holding the key.\n"
        "    location = /health {{\n"
        "        proxy_pass {health};\n"
        "        proxy_buffering off;\n"
        "{health_log}"
        "    }}\n"
        "\n"
        "    location / {{\n"
        "{gate}"
        "{request_log}"
        "        proxy_pass {upstream};\n"
        "        proxy_http_version 1.1;\n"
        "        proxy_set_header Host $host;\n"
        "        proxy_set_header Connection \"\";\n"
        "{streaming}"
        "    }}\n"
        "}}\n"
    ).format(
        head=head, listen=listen, listen_port=listen_p, log_start=log_start,
        upstream=upstream, health=health, health_log=health_log, gate=gate,
        request_log=request_log, streaming=STREAMING_DIRECTIVES,
    )


def render_gate_config(*, listen_host: str, port: int, api_keys: Sequence[str]) -> str:
    """The nginx server block a WORKER's replica gate runs (VD-129, the amendment).

    The worker's vLLM binds ``127.0.0.1:<port>`` and this listens on the
    worker's cluster address on the SAME port - two binds on two addresses,
    which coexist - so the balancer's ``<address>:<port>`` upstream reaches
    the gate and nothing else on that machine is on the LAN:

    * ``/health`` is unauthenticated and proxied, exactly as the LLM Server
      proxy's: the controller's startup wait and mode watch probe it without
      the key, and it reflects the replica's own readiness;
    * ``/v1/`` requires exactly ``Authorization: Bearer <cluster key>``
      (:func:`bearer_check`, the one rendering) and is proxied to loopback
      with :data:`STREAMING_DIRECTIVES`;
    * everything else is ``404``: ``POST /invocations``, ``/tokenize`` and
      ``/metrics`` answered a keyed vLLM 0.22.1 with no key at all, live, and
      are the side doors this block exists to close.

    Rendered by `GpuPoolRuntime.start_gate` and shipped to the worker as a
    root ``0600`` file mounted read-only; the key is in it and nowhere else.
    The listen host is host-shaped here; that it is the worker's own private
    cluster address, never loopback and never every interface, is the
    runtime's check, where the argv is built.

    The ``/health``/``/v1/``/catch-all path map is :func:`replica_locations`,
    the same one the replica balancer renders (VD-129 review S2): a worker's
    gate and the balancer in front of it can never answer a path
    differently again, because there is only the one definition of which
    paths exist.
    """
    listen = _validate_bind_host(listen_host)
    listen_p = _validate_port(port)
    upstream = "http://{}:{}".format(MODEL_LOOPBACK_HOST, listen_p)
    # Replica down: drop the connection unanswered (444) rather than answer
    # 502/503/504. The balancer never replays a sent POST, so a status here
    # reached the client and did NOT mark this replica failed; a dropped
    # connection is an `error`, which nginx always counts (2026-09-29).
    # `error_page` catches nginx's own 502/504 (refused, timed out) and
    # `proxy_intercept_errors` vLLM's; `/health` keeps its honest 502.
    replica_down = (
        "        # The replica behind this gate is down: close the connection\n"
        "        # unanswered, so the balancer marks this replica failed - even\n"
        "        # for a POST - instead of handing a 502 to the client.\n"
        "        proxy_intercept_errors on;\n"
        "        error_page 502 503 504 = @vaelor_replica_down;\n"
    )
    locations = replica_locations(
        upstream=upstream, v1_prefix=bearer_check(api_keys) + replica_down,
        v1_comment=(
            "    # The OpenAI API and only that: the balancer's requests arrive here\n"
            "    # carrying the cluster key.\n"
        ),
    )
    return (
        "server {{\n"
        "    listen {listen}:{port};\n"
        "    server_name _;\n"
        "\n"
        "{locations}"
        "\n"
        "    # 444 is nginx's close-without-a-response.\n"
        "    location @vaelor_replica_down {{\n"
        "        return 444;\n"
        "    }}\n"
        "}}\n"
    ).format(listen=listen, port=listen_p, locations=locations)


def _validate_bind_host(value: Any) -> str:
    """Return a syntactically valid listen host, or raise.

    A correctness/defence gate, not an anti-injection one: the host is rendered
    into the ``listen`` directive, so it admits only host-shaped text.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError("A listen host is required for the LLM Server proxy.")
    if not re.fullmatch(r"[A-Za-z0-9_.:\-]{1,253}", text):
        raise ValueError(
            "The LLM Server proxy listen host '{}' is not a valid host.".format(text)
        )
    return text
