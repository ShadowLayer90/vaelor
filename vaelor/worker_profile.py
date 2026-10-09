"""The slim worker's profile: what a cluster worker holds, as data (VD-194, P1).

A slim worker is the OS plus a few pieces its controller lays down over SSH.
This module is the one statement of those pieces - the manifest - and of the
profile's version, which is a content digest and never a hand-bumped number
(LESSONS 6: a number somebody must remember to bump is a second answer to "did
it change?"). It does no I/O of its own: it takes facts in and returns data.
The only files it touches are this package's own sources, through the bundle
digests it reuses.

**Every content source is reused, never restated** (LESSONS 6). The emitter and
sampler bundles are digested by `worker_telemetry_bundle`, the units and the
config are rendered by `worker_telemetry_config`, the Telegraf tarballs are
pinned there, the gate-config and Ray-token directory modes are the constants
their deploy paths use, and the sensor-module line is `wmi_sensors`'.

Three entry points:

* :func:`release_manifest` / :func:`release_profile_digest` - every component
  for every architecture and capability branch, with per-node, per-controller
  and secret values replaced by fixed placeholders, so the digest depends only
  on this release's code.
* :func:`node_expected` - the subset one worker should hold, with the
  controller's own inputs (its CA's digest, its ingest address) filled in. The
  ingest key and the node id stay placeholders, so a key rotation is not a
  profile change and no secret is ever hashed.
* :func:`canonical_digest` - the one hash both the expected side and the
  measured side (`worker_profile_compare`) are put through, so "the worker
  holds profile X" and "the controller expects profile X" are the same sum.

Nothing here does I/O; the apply job (`worker_profile_apply`,
`worker_profile_job`) lays the profile down from this manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import worker_telemetry_config as telemetry
from .gpu_pool_units import DOCKER, GATE_CONFIG_ROOT, GATE_CONFIG_ROOT_MODE
from .gpu_ray_plane import NFT_BINARY, RAY_TOKEN_ROOT, RAY_TOKEN_ROOT_MODE
from .platforms.graphics_software import integrated_amd_gpu
from .wmi_sensors import WMI_SENSOR_GUID, WMI_SENSOR_MODULE
from .worker_telemetry_bundle import bundle_digest, sampler_bundle_digest

#: Fixed stand-ins for what must never enter a digest: the node's own id, its
#: ingest key, and (in the release digest only) the controller's address and
#: certificate. The probe redacts the same two values on the worker before it
#: hashes the config, so a match means "the same text, whoever's key it holds".
PLACEHOLDER_NODE = "<node-id>"
PLACEHOLDER_KEY = "<ingest-key>"
PLACEHOLDER_INGEST_URL = "<controller-ingest-url>"
PLACEHOLDER_CA = "<controller-ca-sha256>"

#: Component kinds (design §3a.1). ``marker`` is the installer fence; its
#: content records an applied time, so only its presence, owner and mode count.
FILE, UNIT, DIRECTORY, PREREQUISITE, MODULE_CONF, PACKAGE, MARKER = (
    "file", "unit", "directory", "prerequisite", "module-conf", "package", "marker",
)

#: Where the controller-written fence lives (design §2.1 item 6). P2 writes it.
MARKER_PATH = telemetry.CONFIG_DIR + "/worker-profile.json"

#: The appliance's state root and its model cache (design §2.1 item 2).
STATE_ROOT = "/var/lib/vaelor"
MODELS_ROOT = STATE_ROOT + "/models"

#: Where the sensor-module setting lands, and its text. The installer writes the
#: same two lines (`deploy/install-vaelor.sh`, `install_wmi_sensors`); a test
#: ties the two, so the profile adopts the installer's file rather than
#: fighting it (LESSONS 6).
WMI_MODULE_CONF_PATH = "/etc/modules-load.d/vaelor-hp-wmi-sensors.conf"
WMI_MODULE_CONF_TEXT = (
    "# Fans and board temperatures for HP hardware. See vaelor/wmi_sensors.py.\n"
    + WMI_SENSOR_MODULE + "\n"
)

#: AMD's amd-smi, alone and pinned (design §2.5; owner, 2026-10-04). P1 only
#: describes it: the probe reads what is installed, nothing installs it.
AMD_SMI_SERIES = "7.14"
AMD_SMI_PIN = "7.14.1-0"
AMD_SMI_PACKAGES = ("amdrocm-amdsmi" + AMD_SMI_SERIES, "amdrocm-sysdeps" + AMD_SMI_SERIES)
#: The primary fingerprint of AMD's package-signing key. The installer's
#: ``ROCM_GFX1151_KEY_FINGERPRINT`` is the other half of this pair, and a test
#: pins the two equal (design G10, LESSONS 6).
AMD_KEY_FINGERPRINT = "D0F004A0025A1145C7807FCD0701EAC4D5E02107"
AMD_KEYRING_PATH = "/etc/apt/keyrings/amdrocm.gpg"
AMD_SOURCE_PATH = "/etc/apt/sources.list.d/amdrocm.list"
AMD_PREFERENCES_PATH = "/etc/apt/preferences.d/vaelor-amd-smi"
#: The Ubuntu releases AMD publishes this channel for, by ``VERSION_ID``.
AMD_PUBLISHED_UBUNTU = ("26.04",)
_X86_64 = ("x86_64", "amd64")


def amd_source_line(version_id: str) -> str:
    """The apt source line, the installer's own (`install_rocm_gfx1151_runtime`)."""
    return (
        "deb [arch=amd64 signed-by={}] https://repo.amd.com/rocm/packages-multi-arch/"
        "ubuntu{} stable main".format(AMD_KEYRING_PATH, str(version_id).replace(".", ""))
    )


def amd_preferences_text() -> str:
    """Nothing from AMD's repository is chosen as an upgrade or pulled in implicitly."""
    return (
        "# Vaelor worker profile (VD-194): amd-smi is installed from AMD's\n"
        "# repository explicitly, by version. Nothing else from it is ever\n"
        "# chosen as an upgrade or pulled in implicitly.\n"
        "Package: *\n"
        "Pin: origin repo.amd.com\n"
        "Pin-Priority: 100\n"
    )


@dataclass(frozen=True)
class Component:
    """One piece of the profile. ``applies`` names a predicate in :data:`PREDICATES`."""

    id: str
    kind: str
    path: str
    label: str
    applies: str = "always"
    owner: str = "root"
    group: str = "root"
    mode: str = ""
    unit: str = ""
    #: While the full appliance is installed, a difference here is the
    #: appliance's own layout, reported and not repaired (design §2.1 item 2).
    appliance_owned: bool = False
    #: Applying it would interrupt serving; held while a deployment is loaded.
    disruptive: bool = False


#: The manifest, in apply order. No component names an appliance unit or the
#: FLM tree (design G3, `tests/test_worker_profile.py`).
MANIFEST: Tuple[Component, ...] = (
    Component("python3", PREREQUISITE, telemetry.WORKER_PYTHON, "Python 3"),
    Component("nft", PREREQUISITE, NFT_BINARY, "nftables"),
    Component("docker", PREREQUISITE, DOCKER, "Docker"),
    Component("lib-dir", DIRECTORY, telemetry.TELEGRAF_LIB_DIR, "Vaelor library folder", mode="0755"),
    Component("config-dir", DIRECTORY, telemetry.CONFIG_DIR, "Vaelor settings folder", mode="0755"),
    Component("gate-config-dir", DIRECTORY, GATE_CONFIG_ROOT, "Model gate settings folder",
              mode=GATE_CONFIG_ROOT_MODE),
    Component("ray-token-dir", DIRECTORY, RAY_TOKEN_ROOT, "Split-model token folder",
              mode=RAY_TOKEN_ROOT_MODE),
    Component("state-dir", DIRECTORY, STATE_ROOT, "Vaelor data folder", mode="0755",
              appliance_owned=True),
    Component("models-dir", DIRECTORY, MODELS_ROOT, "Model cache folder", mode="0755",
              appliance_owned=True),
    Component("models-pull-dir", DIRECTORY, MODELS_ROOT + "/pull", "Model download folder", mode="0755"),
    Component("models-compile-dir", DIRECTORY, MODELS_ROOT + "/compile", "Model compile cache folder",
              mode="0755"),
    Component("telegraf", FILE, telemetry.TELEGRAF_BINARY_PATH, "Telemetry agent (Telegraf)", mode="0755"),
    Component("emitter", FILE, telemetry.EMITTER_PATH, "Telemetry emitter", mode="0755"),
    Component("controller-ca", FILE, telemetry.CONTROLLER_CA_PATH, "Controller certificate",
              applies="controller_ca", mode="0644"),
    Component("telegraf-config", FILE, telemetry.CONFIG_PATH, "Telemetry agent settings", mode="0600"),
    Component("telegraf-unit", UNIT, telemetry.UNIT_PATH, "Telemetry agent service", mode="0644",
              unit=telemetry.UNIT_NAME),
    Component("sampler", FILE, telemetry.SAMPLER_PATH, "GPU sampler", applies="gpu_sampler", mode="0755"),
    Component("sampler-unit", UNIT, telemetry.SAMPLER_UNIT_PATH, "GPU sampler service",
              applies="gpu_sampler", mode="0644", unit=telemetry.SAMPLER_UNIT_NAME),
    Component("amd-smi", PACKAGE, AMD_PREFERENCES_PATH, "AMD amd-smi", applies="amd_smi", mode="0644"),
    Component("hp-wmi-sensors", MODULE_CONF, WMI_MODULE_CONF_PATH, "Fan and board sensor module",
              applies="hp_wmi", mode="0644"),
    Component("marker", MARKER, MARKER_PATH, "Worker profile marker", mode="0644"),
)

COMPONENTS: Dict[str, Component] = {component.id: component for component in MANIFEST}


def _text_sha(text: str) -> str:
    """The digest of a rendered text, stripped as the runtime compares units."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _bytes_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_text(ingest_url: str, ca_pinned: bool) -> str:
    """The Telegraf config as the controller renders it, node and key as placeholders."""
    return telemetry.render_config(
        node_id=PLACEHOLDER_NODE, ingest_url=ingest_url, ingest_key=PLACEHOLDER_KEY,
        tls_ca_path=telemetry.CONTROLLER_CA_PATH if ca_pinned else None,
        allow_insecure_tls=not ca_pinned,
    )


def key_line_frame() -> Tuple[str, str]:
    """The text either side of the ingest key on its config line, from the renderer.

    The probe replaces whatever sits between these on the worker with
    :data:`PLACEHOLDER_KEY` before hashing, so the key never leaves the worker,
    and the frame is derived from `render_config` rather than restated.
    """
    sentinel = "\x00"
    text = telemetry.render_config(
        node_id=PLACEHOLDER_NODE, ingest_url=PLACEHOLDER_INGEST_URL, ingest_key=sentinel,
        tls_ca_path=telemetry.CONTROLLER_CA_PATH,
    )
    line = next(line for line in text.splitlines() if sentinel in line)
    prefix, _, suffix = line.partition(sentinel)
    return prefix, suffix


# -- capability predicates ---------------------------------------------------

Verdict = Tuple[Optional[bool], str]


def _always(facts: Mapping[str, Any]) -> Verdict:
    return True, ""


def _gpu_sampler(facts: Mapping[str, Any]) -> Verdict:
    wanted = facts.get("gpu_sampler")
    if wanted is None:
        return None, "whether this machine has an AMD integrated GPU was not read"
    if not wanted:
        return False, "this machine has no AMD integrated GPU"
    return True, ""


def _amd_smi(facts: Mapping[str, Any]) -> Verdict:
    architecture = str(facts.get("architecture") or "").lower()
    if not architecture:
        return None, "this machine's architecture was not read"
    if architecture not in _X86_64:
        return False, "AMD publishes amd-smi for x86-64 only; this machine is {}".format(architecture)
    os_id, version = str(facts.get("os_id") or ""), str(facts.get("os_version") or "")
    if not os_id:
        return None, "this machine's operating system was not read"
    if os_id != "ubuntu" or version not in AMD_PUBLISHED_UBUNTU:
        return False, "AMD does not publish amd-smi for {} {}".format(os_id, version).rstrip()
    bound = facts.get("amdgpu_bound")
    if bound is None:
        return None, "whether an AMD GPU is bound to its driver was not read"
    if not bound:
        return False, "no AMD GPU is bound to the amdgpu driver on this machine"
    return True, ""


def _hp_wmi(facts: Mapping[str, Any]) -> Verdict:
    present = facts.get("wmi_guid")
    if present is None:
        return None, "whether the firmware publishes the HP sensor interface was not read"
    if not present:
        return False, "this machine's firmware does not publish the HP sensor interface"
    return True, ""


def _controller_ca(facts: Mapping[str, Any]) -> Verdict:
    if not facts.get("controller_ca_sha256"):
        return False, "this controller has no certificate to pin"
    return True, ""


PREDICATES = {
    "always": _always, "gpu_sampler": _gpu_sampler, "amd_smi": _amd_smi,
    "hp_wmi": _hp_wmi, "controller_ca": _controller_ca,
}


def node_facts(inventory: Mapping[str, Any], probed: Optional[Mapping[str, Any]] = None,
               controller_inputs: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The facts the predicates read: the probe's fresh ones over the enrolled inventory.

    The GPU-sampler fact is the telemetry module's own decision when the read
    path supplies it (``probed["gpu_sampler"]``), and otherwise the same
    inventory predicate `cluster_worker_telemetry.wants_gpu_sampler` falls back
    to - one answer to "does this worker run a sampler?" (LESSONS 6).
    """
    probed = dict(probed or {})
    inventory = dict(inventory or {})
    gpu = inventory.get("gpu") if isinstance(inventory.get("gpu"), dict) else {}
    sampler = probed.get("gpu_sampler")
    if sampler is None and gpu:
        sampler = integrated_amd_gpu({"unified_memory": gpu.get("unified_memory"),
                                      "vendor": inventory.get("gpu_vendor")})
    return {
        "architecture": probed.get("architecture") or inventory.get("architecture") or "",
        "os_id": probed.get("os_id") or inventory.get("os_id") or "",
        "os_version": probed.get("os_version") or inventory.get("os_version") or "",
        "gpu_sampler": sampler,
        "amdgpu_bound": probed.get("amdgpu_bound"),
        "wmi_guid": probed.get("wmi_guid"),
        "controller_ca_sha256": (controller_inputs or {}).get("controller_ca_sha256") or "",
    }


# -- expected values ---------------------------------------------------------

def _expected_fields(component: Component, facts: Mapping[str, Any],
                     inputs: Mapping[str, Any], release: bool) -> Dict[str, Any]:
    """What a worker holding this component reads, field by field.

    ``sha256`` is ``None`` where the content is not compared (the marker) and
    ``""`` with a ``cannot_compare`` reason where the controller itself lacks
    the value to compare against - never a guess.
    """
    fields: Dict[str, Any] = {}
    if component.kind == PREREQUISITE:
        return {"present": True}
    fields.update({"present": True, "owner": component.owner, "group": component.group,
                   "mode": component.mode})
    if component.kind == DIRECTORY:
        fields["acl"] = False
        return fields
    if component.kind == MARKER:
        fields["sha256"] = None
        return fields
    if component.kind == PACKAGE:
        version = str(facts.get("os_version") or "")
        fields.update({
            "sha256": _bytes_sha(amd_preferences_text()),
            "version": AMD_SMI_PIN, "packages": list(AMD_SMI_PACKAGES), "held": True,
            "source_sha256": (
                {release_version: _text_sha(amd_source_line(release_version))
                 for release_version in AMD_PUBLISHED_UBUNTU}
                if release else _text_sha(amd_source_line(version))
            ),
            "key_fingerprint": AMD_KEY_FINGERPRINT,
        })
        return fields
    if component.kind == UNIT:
        text = (telemetry.render_unit() if component.id == "telegraf-unit"
                else telemetry.render_sampler_unit())
        fields.update({"text_sha256": _text_sha(text), "active": "active", "enabled": "enabled"})
        return fields
    fields["sha256"] = _content_sha(component, inputs, release)
    if not fields["sha256"] and not release:
        fields["cannot_compare"] = _cannot_compare(component)
    return fields


def _content_sha(component: Component, inputs: Mapping[str, Any], release: bool) -> Any:
    if component.id == "emitter":
        return bundle_digest()
    if component.id == "sampler":
        return sampler_bundle_digest()
    if component.id == "hp-wmi-sensors":
        return _bytes_sha(WMI_MODULE_CONF_TEXT)
    if component.id == "telegraf":
        if release:
            # The binary's own sum is read from the staged tarball at run time;
            # what the release pins is the tarball itself, per architecture.
            return {arch: artifact["sha256"] for arch, artifact in
                    sorted(telemetry.TELEGRAF_ARTIFACTS.items())}
        return str(inputs.get("telegraf_binary_sha256") or "")
    if component.id == "controller-ca":
        return PLACEHOLDER_CA if release else str(inputs.get("controller_ca_sha256") or "")
    if component.id == "telegraf-config":
        if release:
            return {"pinned": _bytes_sha(config_text(PLACEHOLDER_INGEST_URL, True)),
                    "unpinned": _bytes_sha(config_text(PLACEHOLDER_INGEST_URL, False))}
        url = str(inputs.get("ingest_url") or "")
        if not url:
            return ""
        return _bytes_sha(config_text(url, bool(inputs.get("controller_ca_sha256"))))
    raise ValueError("No content source for component {!r}.".format(component.id))


def _cannot_compare(component: Component) -> str:
    if component.id == "telegraf":
        return ("this controller has no verified Telegraf tarball staged, so the "
                "installed binary cannot be compared")
    if component.id == "telegraf-config":
        return "this controller has no cluster address yet, so the settings cannot be compared"
    return "this controller cannot say what the file should hold"


def release_manifest() -> List[Dict[str, Any]]:
    """Every component for every architecture and capability branch, placeholders in."""
    return [
        {"id": component.id, "kind": component.kind, "path": component.path,
         "unit": component.unit, "applies": component.applies,
         "disruptive": component.disruptive,
         **_expected_fields(component, {}, {}, release=True)}
        for component in MANIFEST
    ]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def release_profile_digest() -> str:
    """SHA-256 of this release's whole profile; depends only on code."""
    return canonical_digest(release_manifest())


def short(digest: Optional[str]) -> str:
    """The first eight hex of a digest, as the console shows it; ``""`` for none."""
    return str(digest or "")[:8]


#: The fields that make up a component's fingerprint, per kind. The measured
#: side (`worker_profile_compare.measured_fields`) fills exactly these keys, so
#: a worker that matches everywhere hashes to :func:`node_expected`'s digest.
FINGERPRINT_FIELDS = {
    PREREQUISITE: ("present",),
    DIRECTORY: ("present", "owner", "group", "mode", "acl"),
    FILE: ("present", "owner", "group", "mode", "sha256"),
    MODULE_CONF: ("present", "owner", "group", "mode", "sha256"),
    MARKER: ("present", "owner", "group", "mode"),
    UNIT: ("present", "owner", "group", "mode", "text_sha256", "active", "enabled"),
    PACKAGE: ("present", "owner", "group", "mode", "sha256", "version", "held", "source_sha256"),
}


def node_expected(inventory: Mapping[str, Any], controller_inputs: Mapping[str, Any],
                  probed: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """What one worker should hold, and the digest of it.

    ``controller_inputs``: ``controller_ca_sha256``, ``ingest_url`` and
    ``telegraf_binary_sha256`` (for this worker's architecture), each ``""``
    when the controller cannot say. ``probed`` is the profile probe's facts.
    Returns ``{"components": [...], "digest": hex}``; a component that does not
    apply carries its reason and is left out of the digest.
    """
    facts = node_facts(inventory, probed, controller_inputs)
    components = []
    for component in MANIFEST:
        applies, why = PREDICATES[component.applies](facts)
        entry: Dict[str, Any] = {
            "id": component.id, "kind": component.kind, "label": component.label,
            "path": component.path, "unit": component.unit, "applies": applies, "why": why,
            "appliance_owned": component.appliance_owned, "disruptive": component.disruptive,
        }
        if applies:
            entry["fields"] = _expected_fields(component, facts, controller_inputs, release=False)
        components.append(entry)
    return {"components": components, "digest": fingerprint_digest(
        (entry["id"], entry["kind"], entry.get("fields") or {})
        for entry in components if entry["applies"]
    )}


def fingerprint_digest(entries) -> str:
    """The canonical digest of ``(id, kind, fields)`` triples, fingerprint keys only."""
    rows = []
    for component_id, kind, fields in entries:
        rows.append({"id": component_id, **{
            key: fields.get(key) for key in FINGERPRINT_FIELDS[kind]
        }})
    return canonical_digest(sorted(rows, key=lambda row: row["id"]))
