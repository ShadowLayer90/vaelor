"""Re-apply the current release's settings to models already deployed.

An upgrade replaces the code that decides how a model is served and leaves the
*deployed* compose alone, so every measured setting in
:mod:`vaelor.model_service_compose` - the pinned engine digest, the prompt-cache
bound, the KV window, the slot-save path, the container limit - stays at the
previous release's value until somebody redeploys that model by hand. Nothing
says so. On 2026-08-10 a Pi served an alpha-29 rendering while `pip show`, the
API and every file on disk reported alpha 35; three measured fixes were
installed and inert, and the only symptom was memory climbing (VD-081).

**This queues the same job the UI queues.** It does not render anything itself.
A second renderer is how two paths drift, and the whole defect above is a
deployed artifact drifting from the code that produces it - reproducing that
shape inside the fix would be absurd. `model.deploy` recomputes everything from
the model path, the port and the runtime mode. The first two the deployed
compose states outright. The third it does not, and cannot: `mode` is the
owner's choice between memory profiles, not a measurement, so it is recovered
from the job ledger instead - see `_last_effective_mode`. This docstring
asserted for one release that path and port "are all that has to be
recovered", which read as a completeness claim and was not one; the cost was
an installer quietly moving owners off `efficient`.

**It runs after the services restart, never before.** The executor performs the
deploy, and an executor still running the previous release renders the previous
release's compose *while reporting success* - which is exactly how the alpha-29
rendering came back on a box where alpha 35 was installed. Order is the whole
correctness argument here, so `install-vaelor.sh` calls this after its health
gate rather than beside the pip install.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .runtime_paths import DATA_ROOT, env_value

#: Where a deployed workload's compose lives, one directory per workload.
WORKLOAD_ROOT = "workloads"

#: The one workload a ``model.deploy`` job can actually target.
#:
#: `ExecutorModelDeployMixin._deploy_model` hard-codes
#: ``_project_directory("model-assistant")`` and the payload carries no project
#: field, so a payload built from some *other* workload's compose does not
#: refresh that workload - it is applied to this one. Scanning every workload
#: therefore repointed the Assistant at another app's model, on another app's
#: port, rather than refreshing anything.
#:
#: It was reachable: ``compose.import`` writes operator-supplied compose text
#: verbatim, so any imported app running llama.cpp matched, and directories are
#: walked in sorted order, so the alphabetically last match is the one that
#: won. The port was wrong too - the regex takes the first published port in
#: the file, which on a multi-service stack belongs to whichever service is
#: listed first, not to the model server.
MANAGED_PROJECT = "model-assistant"

#: The environment line that identifies a compose as a model deployment and
#: names the artifact inside the container.
MODEL_LINE = re.compile(r'^\s*LLAMA_ARG_MODEL:\s*"([^"]+)"', re.MULTILINE)

#: The published port, from the host side of the mapping.
PORT_LINE = re.compile(r'^\s*-\s*"(?:[\d.]+:)?(\d+):\d+"', re.MULTILINE)

#: The read-only model mount, which carries the host directory the artifact
#: actually lives in. The environment line above is a container path.
MOUNT_LINE = re.compile(r'^\s*-\s*"([^"]+):/models:ro"', re.MULTILINE)


def data_root() -> Path:
    """The state root, from the one definition that already honours the legacy
    alias. Recomputing it here called `env_value` with two arguments where it
    takes three - a crash that every unit test missed, because they all passed
    `--root` and never exercised the default. The appliance found it."""
    return DATA_ROOT


def deployed_models(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every deployed model workload, with the facts a redeploy needs.

    Returns the *host* path of the artifact rather than the container path, so
    the value can be handed straight to a `model.deploy` payload.
    """
    base = (root or data_root()) / WORKLOAD_ROOT
    found: List[Dict[str, Any]] = []
    if not base.is_dir():
        return found
    for directory in sorted(base.iterdir()):
        # Only the workload a redeploy can be aimed at; see `MANAGED_PROJECT`.
        if directory.name != MANAGED_PROJECT:
            continue
        compose = directory / "compose.yaml"
        if not compose.is_file():
            continue
        try:
            text = compose.read_text(encoding="utf-8")
        except OSError:
            continue
        model = MODEL_LINE.search(text)
        if not model:
            continue  # not a model deployment
        mount = MOUNT_LINE.search(text)
        port = PORT_LINE.search(text)
        # `/models/x.gguf` inside the container is `<mounted dir>/x.gguf` on the
        # host. Resolved rather than guessed: the deploy path verifies the file
        # exists and a container path would simply not be there.
        name = model.group(1).rsplit("/", 1)[-1]
        found.append({
            "workload": directory.name,
            "compose": str(compose),
            "path": "{}/{}".format(mount.group(1).rstrip("/"), name)
            if mount else "",
            "port": int(port.group(1)) if port else 0,
        })
    return found


def refresh_payloads(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """`model.deploy` payloads for every deployment that can be described.

    A deployment whose compose does not state both a host mount and a published
    port is skipped and reported rather than guessed at - a payload built from
    half a compose would deploy something nobody asked for.
    """
    payloads = []
    for entry in deployed_models(root):
        if not entry["path"] or not entry["port"]:
            continue
        payloads.append({
            "path": entry["path"],
            "port": entry["port"],
            "surface": "assistant",
            # Marks the job as machine-initiated so the ledger reads honestly:
            # nobody clicked anything.
            "reason": "post-upgrade refresh",
        })
    return payloads


def _incomplete(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    return [
        entry for entry in deployed_models(root)
        if not entry["path"] or not entry["port"]
    ]


def _last_effective_mode(store: Any) -> str:
    """The runtime mode actually in effect, recovered from the job ledger.

    **The compose does not state it, and it is not derivable from what does.**
    This module's own docstring claimed a redeploy "recomputes everything from
    the model path and the port, both of which the deployed compose already
    states, so those two facts are all that has to be recovered". That was
    wrong: ``mode`` is a third input, both UI call sites always send it, and it
    is the operator's decision rather than a measurement.

    Omitting it made `_deploy_model` fall back to `RECOMMENDED_RUNTIME_MODE`,
    so an owner who deliberately chose `efficient` on a memory-tight Pi had
    their window doubled (2,048 to 4,096) and their container limit raised by
    512 MiB - silently, by an installer, in the direction the appliance has
    been OOM-killed and power-cycled from before.

    **Only the newest completed deploy is consulted, and only about itself
    (#140).** An older completed deploy describes a deployment that was since
    replaced, and a failed one describes a configuration that did not take -
    reading either resurrects a choice the machine is not running. Within
    that one record, three fields in order:

    - ``result.requested_mode`` - the operator's own ask, which the deploy
      records beside what it effected. Preferred over the effective mode
      because the selector only ever walks *down* from a request: re-issuing
      the walked-down result would ratchet, so an owner's `quality` that a
      pessimistic release once trimmed to `balanced` could never return even
      after the estimate was corrected or RAM was added.
    - ``result.mode`` - the effective mode, for results written before the
      request was recorded beside it.
    - ``payload.mode`` - the same deploy's own request, for ledgers older
      than result recording entirely.

    A newest completed deploy carrying none of the three ran at
    `RECOMMENDED_RUNTIME_MODE`, and "" - the deploy's own default - is the
    honest recovery; reaching past it to an older record's payload is exactly
    the resurrect this function was rewritten to stop. "" is likewise the
    answer for a ledger with no completed deploy at all.
    """
    try:
        records = store.list(limit=200)
    except (OSError, ValueError):
        return ""
    for record in records:
        if record.get("type") != "model.deploy":
            continue
        if record.get("state") != "completed":
            continue
        result = record.get("result") or {}
        for value in (
            result.get("requested_mode"),
            result.get("mode"),
            (record.get("payload") or {}).get("mode"),
        ):
            mode = str(value or "").strip()
            if mode:
                return mode
        return ""
    return ""


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Queue a redeploy for every deployed model, so an upgrade's "
                    "measured settings reach what is actually running.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--apply", action="store_true",
        help="queue the jobs; without it the payloads are printed and nothing "
             "is changed")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payloads = refresh_payloads(args.root)
    skipped = _incomplete(args.root)
    for entry in skipped:
        print("skipped {}: compose states no {}".format(
            entry["workload"],
            "model directory" if not entry["path"] else "published port"))
    if not payloads:
        print("no deployed model to refresh")
        return 0
    for payload in payloads:
        print("refresh {} on port {}".format(payload["path"], payload["port"]))
    if not args.apply:
        print("(dry run; pass --apply to queue)")
        return 0

    from .jobs import JobStore

    store = JobStore(env_value(
        "VAELOR_JOBS_DB", "PM_JOBS_DB",
        str(data_root() / "jobs" / "jobs.sqlite3")))
    mode = _last_effective_mode(store)
    for payload in payloads:
        if mode:
            payload["mode"] = mode
        job = store.create("model.deploy", "system", payload)
        print("queued {} for {}{}".format(
            job["id"], payload["path"],
            " in {} mode".format(mode) if mode else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
