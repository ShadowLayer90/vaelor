"""Telegraf placement facts and rendered config/unit for the worker agent (E2b).

The transport-driven install lives in `worker_telemetry_runtime`; this module
owns the *names and text* that install writes onto a worker — where the Telegraf
binary, the emitter bundle and the config land, the pinned Telegraf release, and
the rendered ``telegraf.conf`` and ``vaelor-telegraf.service``. Kept apart from
the runtime for the same reason `gpu_pool_units` is kept apart from
`gpu_pool_runtime`: one module is the node command surface, the other is the
text it emits, and the text is independently unit-testable.

**Telegraf is the agent**, not a bespoke Python pusher: it is itself one lean
static binary, and it will also scrape a local vLLM/llama.cpp ``/metrics`` for
serving metrics (E4), so it goes in now. It runs the emitter as an
``inputs.exec`` input and posts the single ``influx`` line to the controller's
keyed ingest route.

**The one tag rule.** The ingest route 400s any tag whose key is not ``node``
(`telemetry_ingest._parse_line`). Telegraf adds a ``host`` tag by default;
``omit_hostname = true`` suppresses it, and this config defines **no**
``[global_tags]`` and no tag-adding processor, so the only tag on the wire is the
emitter's own ``node=<id>``. A test asserts exactly this, because a stray tag
makes every post fail.

**The key lives only in the config.** It is rendered into the ``0600`` file and
appears on no argv and in no unit — the same discipline `gpu_pool_runtime`'s gate
config keeps.
"""

from __future__ import annotations

from typing import Dict, Optional

#: Where the shipped stack lands on the worker. A Vaelor-owned tree under
#: ``/usr/local/lib`` so nothing collides with a distro package.
TELEGRAF_LIB_DIR = "/usr/local/lib/vaelor"
TELEGRAF_BINARY_PATH = TELEGRAF_LIB_DIR + "/telegraf"
EMITTER_PATH = TELEGRAF_LIB_DIR + "/vaelor-telemetry-emitter.pyz"

#: Config and cert root, root-owned. The config is ``0600`` (it carries the key).
CONFIG_DIR = "/etc/vaelor"
CONFIG_PATH = CONFIG_DIR + "/telegraf.conf"
CONTROLLER_CA_PATH = CONFIG_DIR + "/controller-ca.pem"

#: The systemd unit name. One agent per worker.
UNIT_NAME = "vaelor-telegraf.service"
UNIT_PATH = "/etc/systemd/system/" + UNIT_NAME

#: The GPU vendor sampler (VD-147): its bundle beside the emitter's, and its
#: own unit. It runs as a ``DynamicUser``, so there is no account to manage.
SAMPLER_PATH = TELEGRAF_LIB_DIR + "/vaelor-gpu-sampler.pyz"
SAMPLER_UNIT_NAME = "vaelor-gpu-sampler.service"
SAMPLER_UNIT_PATH = "/etc/systemd/system/" + SAMPLER_UNIT_NAME
#: Where the sampler's bundle is staged while it is verified: inside the
#: root-owned library directory, never under ``/tmp``.
SAMPLER_STAGING_PATH = TELEGRAF_LIB_DIR + "/.vaelor-gpu-sampler.pyz.new"
#: systemd's name for the sampler's runtime directory (``/run/vaelor-gpu``).
SAMPLER_RUNTIME_DIRECTORY = "vaelor-gpu"

#: The sandbox the sampler runs in, as ``(property, value)`` in unit order.
#: **This exact set was probed on both boxes** (S0, 2026-09-29, a transient
#: ``systemd-run`` unit): under it ``amd-smi metric --json`` exits 0 with no
#: access banner and every reading root sees. ``/dev/kfd`` is not allowed
#: because ``amd-smi`` never opens it (strace: only ``/dev/dri/renderD*``), and
#: the probe passed without it. A test pins this list, so the unit cannot drift
#: from what was measured.
SAMPLER_SANDBOX = (
    ("DynamicUser", "yes"),
    ("SupplementaryGroups", "render video"),
    ("RuntimeDirectory", SAMPLER_RUNTIME_DIRECTORY),
    ("RuntimeDirectoryMode", "0755"),
    ("PrivateNetwork", "yes"),
    ("RestrictAddressFamilies", "AF_UNIX"),
    ("CapabilityBoundingSet", ""),
    ("AmbientCapabilities", ""),
    ("ProtectKernelTunables", "yes"),
    ("NoNewPrivileges", "yes"),
    ("DevicePolicy", "closed"),
    ("DeviceAllow", "char-drm rw"),
)

#: The interpreter every Linux host ships and the only thing the lean worker is
#: assumed to pre-have. The emitter zipapp runs under it.
WORKER_PYTHON = "/usr/bin/python3"

#: The API path the controller exposes for keyed ingest (E2a).
INGEST_ROUTE = "/api/v2/telemetry/ingest"

#: The one line-protocol format directive, shared by the exec input (which reads
#: the emitter's ``influx`` output) and the http output (which posts it), so the
#: two cannot disagree about the wire format.
_INFLUX_FORMAT = '  data_format = "influx"'

#: The pinned Telegraf release. One version for the whole fleet so every worker
#: runs the reviewed binary; the per-arch tarball and its checksum are below.
TELEGRAF_VERSION = "1.32.3"

#: Per-arch Telegraf static-binary artifacts, pinned by sha256. The hashes are
#: of the official release tarballs fetched over HTTPS from InfluxData's CDN
#: (``https://dl.influxdata.com/telegraf/releases/telegraf-<ver>_linux_<arch>.tar.gz``)
#: and computed with ``sha256sum``. The verify step is wired unconditionally
#: (`worker_telemetry_runtime` refuses to install a binary whose local or remote
#: sha256 does not match this constant), so a swapped or truncated artifact is
#: never installed silently. Bump both together when ``TELEGRAF_VERSION`` moves.
TELEGRAF_ARTIFACTS: Dict[str, Dict[str, str]] = {
    "amd64": {
        "filename": "telegraf-{}_linux_amd64.tar.gz".format(TELEGRAF_VERSION),
        "sha256": "260bc3170dbd6cce67575c1215a0b89b8447945106e2943d74e617d06b750c03",
        # Where the ``telegraf`` binary sits inside the release tarball.
        "member": "./telegraf-{}/usr/bin/telegraf".format(TELEGRAF_VERSION),
    },
    "arm64": {
        "filename": "telegraf-{}_linux_arm64.tar.gz".format(TELEGRAF_VERSION),
        "sha256": "f0d8ccae539afa04b171d5268dbab21eef58bc51b5437689e347619e2097c824",
        "member": "./telegraf-{}/usr/bin/telegraf".format(TELEGRAF_VERSION),
    },
}

#: Maps a probe's ``uname -m`` / architecture class onto a Telegraf artifact key.
_ARCH_ALIASES = {
    "amd64": "amd64", "x86_64": "amd64", "x86-64": "amd64",
    "arm64": "arm64", "aarch64": "arm64",
}


def telegraf_artifact(architecture: str) -> Dict[str, str]:
    """The pinned Telegraf artifact for an architecture, or a refusal.

    The architecture comes from the enrolled node's own probe, so an
    unrecognised value is a node the controller cannot ship a reviewed binary
    to rather than a guess at one.
    """
    key = _ARCH_ALIASES.get(str(architecture or "").strip().lower())
    if key is None or key not in TELEGRAF_ARTIFACTS:
        raise ValueError(
            "No pinned Telegraf binary for architecture {!r}.".format(architecture)
        )
    return TELEGRAF_ARTIFACTS[key]


def ingest_url(address: str, port: int) -> str:
    """The controller's keyed ingest URL a worker posts to.

    ``https`` because the appliance serves TLS; the worker verifies it against
    the pinned controller certificate (see :func:`render_config`).
    """
    clean = str(address or "").strip()
    if not clean:
        raise ValueError("The controller has no cluster address to post telemetry to.")
    return "https://{}:{}{}".format(clean, int(port), INGEST_ROUTE)


def emitter_command(node_id: str, *, emitter_path: str = EMITTER_PATH,
                    python_path: str = WORKER_PYTHON) -> str:
    """The one ``inputs.exec`` command line: run the emitter for this node.

    ``node_id`` is a validated cluster id (letters/digits/-/_), so it carries no
    space or quote and Telegraf's word split cannot break the argv apart.
    """
    return "{} {} --node {}".format(python_path, emitter_path, node_id)


def render_unit() -> str:
    """The ``vaelor-telegraf.service`` unit that supervises the agent.

    ``ExecStart`` names the config on argv — the config *path*, never the key,
    which lives inside that ``0600`` file. Assembled line by line rather than
    from a shared template, matching `gpu_pool_units.unit_text`.
    """
    lines = [
        "[Unit]",
        "Description=Vaelor worker telemetry agent (Telegraf)",
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        "ExecStart={} --config {}".format(TELEGRAF_BINARY_PATH, CONFIG_PATH),
        "Restart=on-failure",
        "RestartSec=5",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ]
    return "\n".join(lines)


def render_sampler_unit() -> str:
    """The ``vaelor-gpu-sampler.service`` unit: a non-root, long-running loop.

    ``ExecStart`` is a fixed argv that takes no input. The service restarts on
    failure; systemd creates its runtime directory, hands it to the dynamic
    user, and removes it when the service stops.
    """
    lines = [
        "[Unit]",
        "Description=Vaelor GPU vendor sampler (amd-smi, non-root)",
        "",
        "[Service]",
        "Type=simple",
        "ExecStart={} {}".format(WORKER_PYTHON, SAMPLER_PATH),
        "Restart=on-failure",
        "RestartSec=10",
    ]
    lines += ["{}={}".format(name, value) for name, value in SAMPLER_SANDBOX]
    lines += ["", "[Install]", "WantedBy=multi-user.target", ""]
    return "\n".join(lines)


def sampler_probe_arguments() -> list:
    """The ``systemd-run`` property arguments that reproduce the sampler's sandbox.

    For a privilege probe (``systemd-run --wait --pipe <these> amd-smi metric
    --json``): derived from :data:`SAMPLER_SANDBOX`, the same table the unit is
    rendered from, so a probe cannot test a different sandbox from the one
    shipped.
    """
    arguments = []
    for name, value in SAMPLER_SANDBOX:
        arguments += ["-p", "{}={}".format(name, value)]
    return arguments


def render_config(
    *,
    node_id: str,
    ingest_url: str,
    ingest_key: str,
    emitter_path: str = EMITTER_PATH,
    python_path: str = WORKER_PYTHON,
    tls_ca_path: Optional[str] = CONTROLLER_CA_PATH,
    allow_insecure_tls: bool = False,
) -> str:
    """Render the ``0600`` ``telegraf.conf`` for one worker.

    * ``[agent]`` sets ``interval``/``flush_interval`` to ``1s`` (the ingest cap
      is one accepted post per node per second, so the agent batches exactly one
      line per second) and ``omit_hostname = true``. There is **no**
      ``[global_tags]`` and no processor, so the only tag on the wire is the
      emitter's ``node=<id>``.
    * ``[[inputs.exec]]`` runs the emitter and parses its ``influx`` output.
    * ``[[outputs.http]]`` POSTs to the keyed ingest route; the key is the
      ``Authorization: Bearer`` header, and it appears **only** here.
    * TLS: the controller's cert is self-signed, so verification is pinned to the
      shipped controller CA (``tls_ca``). Only when no CA path is available and
      the caller has explicitly allowed it does this fall back to
      ``insecure_skip_verify`` — and it says so in a comment, flagged as a
      follow-up. It never silently disables verification.
    * A commented ``[[inputs.prometheus]]`` placeholder marks where E4's local
      vLLM/llama.cpp ``/metrics`` scrape will go. It is not wired now.
    """
    command = emitter_command(node_id, emitter_path=emitter_path, python_path=python_path)
    lines = [
        "# Vaelor worker telemetry agent (Phase E2b). Rendered 0600 by the",
        "# controller. The ingest key below appears ONLY in this file — never on",
        "# a command line and never in the systemd unit.",
        "",
        "[agent]",
        '  interval = "1s"',
        '  flush_interval = "1s"',
        "  omit_hostname = true",
        "  # No [global_tags]. The ingest route refuses any tag whose key is not",
        "  # `node`; omit_hostname suppresses Telegraf's default `host` tag, and",
        "  # nothing here adds another, so the only tag on the wire is the",
        "  # emitter's own node=<id>.",
        "",
        "[[inputs.exec]]",
        '  commands = ["{}"]'.format(command),
        '  timeout = "5s"',
        _INFLUX_FORMAT,
        "",
        "# E4 (not wired yet): a future [[inputs.prometheus]] will scrape a local",
        "# vLLM/llama.cpp /metrics endpoint into this same agent. Left as a marked",
        "# placeholder so serving metrics ride the transport host telemetry uses.",
        "# [[inputs.prometheus]]",
        '#   urls = ["http://127.0.0.1:8000/metrics"]',
        "",
        "[[outputs.http]]",
        '  url = "{}"'.format(ingest_url),
        '  method = "POST"',
        _INFLUX_FORMAT,
    ]
    if tls_ca_path:
        lines += [
            "  # Self-signed appliance cert, pinned: verify against the controller",
            "  # CA shipped beside this config rather than trusting the system store.",
            '  tls_ca = "{}"'.format(tls_ca_path),
        ]
    elif allow_insecure_tls:
        lines += [
            "  # FOLLOW-UP (E2b): no controller CA was pinned, so certificate",
            "  # verification is disabled here. Ship and point tls_ca at the",
            "  # controller cert to close this — do not leave it in production.",
            "  insecure_skip_verify = true",
        ]
    else:
        raise ValueError(
            "Refusing to render a Telegraf output with neither a pinned "
            "controller CA nor an explicit insecure-TLS opt-in."
        )
    lines += [
        "  [outputs.http.headers]",
        '    Authorization = "Bearer {}"'.format(ingest_key),
        '    Content-Type = "text/plain; charset=utf-8"',
        "",
    ]
    return "\n".join(lines)
