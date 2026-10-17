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

**Cluster GPU deployments are refreshed too (W4-D1).** A pooled vLLM deployment
is not a compose and no ``model.deploy`` reaches it, so an upgrade used to leave
its units at the previous release's text - VD-165's environment line installed
and inert on both machines of a replicated deployment. Each such row records
what its units were rendered from (`gpu_render_ledger`); this reads the rows
read-only, asks which ones this release renders differently, says which and
why, and queues one ``cluster.gpu.refresh`` per serving row that does - the
same Unload and Load the owner would run (`gpu_pool_refresh`). It renders
nothing itself here either: the comparison re-renders through the runtime's
one template, and the job runs in the executor, after the restart. A row
whose units still read as this release renders them is refreshed too when a
Load would now serve it with different options - this release's measured
settings for a deployment made before them (VD-217, kb G71).

**Run as root with ``--apply``, it first runs the release's root build steps**
(`post_install_steps`, VD-217), so the reloads it queues serve what they
build - on the installer and on the console update alike.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

# The cluster store's file under the state root is `ClusterStore`'s own
# constant, so the installer's read and the store cannot disagree (review A4).
from .cluster_store import CLUSTER_DATABASE
from .runtime_paths import DATA_ROOT, env_value, state_path
# The cluster-agent reconcile lived here beside the model refresh it mirrors;
# it now has its own module (ACC-070) and is re-exported for its callers.
from .agent_reconcile import reconcile_cluster_agents  # noqa: F401

#: What every job this command queues says about itself in the ledger, so the
#: record reads honestly: nobody clicked anything.
REFRESH_REASON = "post-upgrade refresh"

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
            "reason": REFRESH_REASON,
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


#: How long the installer waits on a store another process holds locked. Short
#: on purpose: a lock says the GPU check could not run, at once, rather than
#: holding the install (review of W4-D1: a 12 s lock read as "no deployment").
STORE_BUSY_SECONDS = 2


class StoreUnread(RuntimeError):
    """The cluster store exists but could not be read; the message says why."""


def cluster_database(root: Optional[Path] = None) -> Path:
    """The cluster store's file: ``--root``'s own when one is given, else the
    store's own setting, honouring its legacy alias exactly as the store does."""
    if root is not None:
        return root / CLUSTER_DATABASE
    return Path(env_value("VAELOR_CLUSTER_DB", "PM_CLUSTER_DB", state_path(CLUSTER_DATABASE)))


def read_only_store(database: Path):
    """A connection to the store that cannot write: an SQLite read-only URI.

    The installer runs as root, and a store file (or a journal) root created
    beside the services' own would lock them out. The path is percent-encoded
    so a ``?`` or ``#`` in it cannot become a URI parameter.
    """
    import sqlite3
    from urllib.parse import quote

    connection = sqlite3.connect(
        "file:{}?mode=ro".format(quote(database.as_posix(), safe="/:")),
        uri=True, timeout=STORE_BUSY_SECONDS,
    )
    connection.row_factory = sqlite3.Row
    return connection


def _controller_gpu() -> Dict[str, Any]:
    """This controller's GPU, read as the deploy reads it (`cluster_operations._controller_gpu_facts`)."""
    from .cluster_capacity import gpu_facts_from_accelerators
    from .copilot_setup import hardware_inventory

    hardware = hardware_inventory() or {}
    return dict(gpu_facts_from_accelerators(
        hardware.get("accelerators"), int(hardware.get("memory_total_bytes", 0) or 0) or None))


def would_load(record: Dict[str, Any], inventories: Dict[str, Any],
               controller_gpu: Callable[[], Dict[str, Any]] = _controller_gpu) -> Any:
    """The options a Load would serve this serving row with on the fleet as recorded (VD-217).

    The Load's own decision (`vllm_serve_options.options_for_load`, release
    defaults and item 2's speed-ups included) on each machine's GPU type as
    the deploy reads it: a worker's from its last probe in the store, this
    controller's from its own probe. Without the fit, which needs the running
    control plane; the refresh job asks the Load again, fit included, before it
    stops anything (`gpu_pool_refresh`). ``None`` when it cannot be decided -
    a row not serving, or a fact that cannot be read - so the unit text
    alone decides, as before.
    """
    import json

    from .cluster_placement import CONTROLLER_PLACEMENT_ID
    from .gpu_cluster_mode import HEALTHY_STATE
    from .gpu_pool_refit import gpu_targets_of
    from .gpu_pool_units import is_replicated
    from .vllm_serve_options import options_for_load

    if str(record.get("state", "")) != HEALTHY_STATE:
        return None
    try:
        nodes = []
        for node_id in record.get("node_ids") or []:
            if str(node_id) == CONTROLLER_PLACEMENT_ID:
                gpu = controller_gpu()
            else:
                gpu = (json.loads(inventories.get(str(node_id)) or "{}") or {}).get("gpu") or {}
            nodes.append({"inventory": {"gpu": gpu}})
        units = record.get("units") or {}
        return options_for_load(
            units, str(units.get("repo", "") or record.get("model_id", "") or ""),
            replicated=is_replicated(record), facts=None, gpu_targets=gpu_targets_of(nodes),
        )[0]
    except Exception:  # noqa: BLE001 - an unread fact is not a change (LESSONS 8)
        return None


def gpu_refresh_plan(
    database: Path, controller_gpu: Callable[[], Dict[str, Any]] = _controller_gpu,
) -> List[Dict[str, Any]]:
    """Every vLLM deployment with whether this release must re-render it, and why.

    Read only, and only when the store exists (:func:`read_only_store`); a
    store that exists and cannot be read raises :class:`StoreUnread` with the
    reason, never "nothing to refresh". Each row is judged on its own: one
    that cannot be read is reported as such (``error``) and the rest are still
    planned. Nothing in a row is changed here - the job does that.
    """
    import sqlite3
    from contextlib import closing

    from .cluster_store import POOLED_ROWS_QUERY, ClusterStore
    from .gpu_render_ledger import render_drift

    if not database.is_file():
        return []
    try:
        with closing(read_only_store(database)) as connection:
            rows = connection.execute(POOLED_ROWS_QUERY).fetchall()
            try:
                inventories = {
                    str(node["id"]): node["inventory_json"] for node in connection.execute(
                        "SELECT id, inventory_json FROM cluster_nodes")
                }
            except sqlite3.Error:
                # No machine facts: every row is judged on its unit text alone.
                inventories = {}
    except sqlite3.Error as error:
        raise StoreUnread("the cluster store could not be read ({})".format(error)) from error
    plan = []
    for row in rows:
        try:
            record = ClusterStore._pooled(row)
            if (record.get("units") or {}).get("engine") != "vllm":
                continue
            plan.append({"name": record["name"], **render_drift(
                record, would_load(record, inventories, controller_gpu))})
        except Exception as error:  # noqa: BLE001 - one bad row reports itself
            name = row["name"] if "name" in row.keys() else "a row"
            plan.append({"name": str(name), "refresh": False, "changed": [],
                         "reason": "", "error": "{}: {}".format(type(error).__name__, error)[:200]})
    return plan


def _queue_gpu_refreshes(plan: List[Dict[str, Any]], store: Any, record: Any = None) -> None:
    from .cluster_job_confirmations import confirmation_for
    from .job_vocabulary import CLUSTER_GPU_REFRESH_JOB

    for entry in plan:
        if not entry["refresh"]:
            continue
        job = store.create(CLUSTER_GPU_REFRESH_JOB, "system", {
            "name": entry["name"],
            "confirm": confirmation_for(CLUSTER_GPU_REFRESH_JOB),
            "reason": REFRESH_REASON,
        })
        if record is not None:
            record.queued(job, entry["name"])
        print("queued {} for GPU deployment {}; it stops and loads again, so it "
              "does not answer until that Load is healthy".format(job["id"], entry["name"]))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Queue a redeploy for every deployed model, and a refresh for "
                    "every serving cluster GPU deployment this release renders "
                    "differently, so an upgrade's settings reach what is running.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--apply", action="store_true",
        help="queue the jobs; without it the payloads are printed and nothing "
             "is changed")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # VD-217: the release's root builds land BEFORE anything below queues a
    # reload, on every path that runs this - the installer and the console
    # update's broker alike (`post_install_steps`: one list, no new door). Only
    # as root, only when queueing, and only on the box's own state root.
    held = None
    record = None
    if args.apply and args.root is None:
        from .post_install_steps import is_root, run_steps

        if is_root():
            # R102: one apply at a time, from the first update on - this runs
            # from the new package even when the broker that started it is
            # the previous release's (`upgrade_apply_lock`).
            from . import upgrade_apply_lock as apply_lock
            from .appliance_upgrade import RETAINED_DIR

            if os.environ.get(apply_lock.HELD_ENV) != "1":
                held = apply_lock.acquire(RETAINED_DIR, apply_lock.REFRESH_WAIT_SECONDS)
                if held is None:
                    print(apply_lock.BUSY)
                    return 1
            # R104: each step's outcome, then the reloads queued, written as
            # they happen for the update panel (`upgrade_progress`).
            from .upgrade_progress import RefreshRecord

            record = RefreshRecord()
            run_steps(record=record.step)
            record.steps_ended()
            # R105: a broker from before the lock never restarts itself, so
            # this - the one command it runs from the new package - has it
            # restarted onto the new code once it is idle (`upgrade_broker_restart`).
            from .upgrade_broker_restart import schedule

            print(schedule())
    try:
        return _refresh(args, record)
    finally:
        if record is not None:
            record.ended()


def _refresh(args: Any, record: Any) -> int:
    """The refresh itself, after any build steps: report, then queue with ``--apply``."""
    payloads = refresh_payloads(args.root)
    skipped = _incomplete(args.root)
    for entry in skipped:
        print("skipped {}: compose states no {}".format(
            entry["workload"],
            "model directory" if not entry["path"] else "published port"))
    for payload in payloads:
        print("refresh {} on port {}".format(payload["path"], payload["port"]))
    try:
        plan = gpu_refresh_plan(cluster_database(args.root))
    except Exception as error:  # noqa: BLE001 - never at the Assistant's expense
        # The Assistant's refresh never depended on the cluster store; a store
        # this cannot read must not cost it that (VD-081), and is said - never
        # passed off as "nothing to refresh".
        print("GPU deployments were not checked: {}".format(
            str(error) if isinstance(error, StoreUnread) else type(error).__name__))
        plan = []
    for entry in plan:
        if entry.get("error"):
            print("GPU deployment {} was not checked: {}".format(entry["name"], entry["error"]))
            continue
        print("{} GPU deployment {}: {}".format(
            "refresh" if entry["refresh"] else "skipped", entry["name"], entry["reason"]))
    if not payloads:
        print("Assistant model: none deployed here, so none to refresh")
    if not payloads and not any(entry["refresh"] for entry in plan):
        return 0
    if not args.apply:
        print("(dry run; pass --apply to queue)")
        return 0

    from .jobs import JobStore

    store = JobStore(env_value(
        "VAELOR_JOBS_DB", "PM_JOBS_DB",
        str(data_root() / "jobs" / "jobs.sqlite3")))
    mode = _last_effective_mode(store) if payloads else ""
    for payload in payloads:
        if mode:
            payload["mode"] = mode
        job = store.create("model.deploy", "system", payload)
        if record is not None:
            record.queued(job, str(payload["path"]).rsplit("/", 1)[-1])
        print("queued {} for {}{}".format(
            job["id"], payload["path"],
            " in {} mode".format(mode) if mode else ""))
    _queue_gpu_refreshes(plan, store, record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
