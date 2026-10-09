"""The model-library pull on a GPU node: its unit, its start/poll pair, its paths.

Moved out of `gpu_pool_runtime` (ACC-163's review) when the Ray plane took that
module to its 1,000-line ceiling, and cohesive on its own: the pull is the one
unit the GPU tier writes that serves no model. `GpuPoolRuntime` inherits
:class:`PullUnitsMixin`, so every caller still asks the runtime; the names
below are re-exported there for the readers that have always used them.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from .bridge_transport import root_renders_units
from .gpu_pool_pull import classify_pull, pull_outcome
from .gpu_pool_units import (
    DOCKER, pull_container_name, pull_unit_name, pull_unit_text, repo_slug,
    require_pull_unit,
)
from .gpu_pull_program import PULL_SCRIPT, PULL_SCRIPT_DIGEST, PULL_SCRIPT_NAME
from .ssh_transport import SshTransportError
from .vllm_container import COMPILE_CACHE_ROOT, MODEL_CACHE_ROOT, MODELS_MOUNT
from .vllm_entry_program import ENTRY_SCRIPT_NAME, write_program_on_worker
from .vllm_images import IMAGE_FIELD, RECORDED_IMAGE_KEY, image_profile

#: The pull container runs the image's own Python by name (`gpu_pool_runtime`
#: names Ray, shell and vLLM the same way).
PYTHON_ENTRYPOINT = "python"

#: The pull's managed-unit kind (`gpu_pool_runtime.MANAGED_UNIT_PARAMETERS`).
MODEL_PULL_KIND = "model-pull"

#: Where every managed unit file lives, on a worker and on the controller alike.
UNIT_DIRECTORY = "/etc/systemd/system"

#: Where the pull script and its progress JSON live, on the host and inside the
#: container. Under the cache root so one bind mount covers both, and so a poll
#: can ``cat`` the progress file on the HOST path while the container writes it
#: through the mount.
_PULL_PROGRESS_ROOT = MODEL_CACHE_ROOT + "/pull"
_CONTAINER_PULL_ROOT = MODELS_MOUNT + "/pull"

#: The systemd properties the poll reads in one ``systemctl show`` to decide
#: whether a pull is queued, running, finished cleanly, or failed.
#: ``ExecMainStartTimestampMonotonic`` separates the first two: ``systemctl start
#: --no-block`` returns on ENQUEUE, and a unit that never ran reports
#: ``ActiveState=inactive`` with systemd's default ``Result=success`` -
#: indistinguishable from a clean finish without it.
_PULL_UNIT_PROPERTIES = (
    "ActiveState,SubState,Result,ExecMainStatus,ExecMainStartTimestampMonotonic"
)

#: Where the content-hashed pull program lands, on the host and inside the
#: container. `gpu_pull_program` owns the program and the name its digest gives
#: it; this module owns the paths, because it owns the mount they are expressed
#: against.
_PULL_SCRIPT_PATH = f"{_PULL_PROGRESS_ROOT}/{PULL_SCRIPT_NAME}"
#: Where a WORKER keeps the MoE entry program (`vllm_entry_program`), beside it.
_ENTRY_SCRIPT_PATH = f"{_PULL_PROGRESS_ROOT}/{ENTRY_SCRIPT_NAME}"
_CONTAINER_PULL_SCRIPT = f"{_CONTAINER_PULL_ROOT}/{PULL_SCRIPT_NAME}"

#: Where the controller's pull container finds the program when the ROOT bridge
#: mounts it read-only out of Vaelor's own installed package (VD-143), rather than
#: out of the model store every ``vaelor-jobs`` account can write. A worker still
#: reads it from the store, where `_ensure_pull_script` puts it over SSH.
_CONTAINER_PACKAGED_PROGRAM = "/opt/vaelor-pull/vaelor_pull.py"

#: The characters a packaged program path may carry into a pull unit's mount.
_PROGRAM_PATH = r"/[A-Za-z0-9._/-]{1,255}\.py"


class PullUnitsMixin:
    """The pull half of `GpuPoolRuntime`: start, poll, stop, remove."""

    # --- Model-library pull: a robust start/poll pair -----------------------
    #
    # The model library (Phase B2) needs truthful, granular progress and to
    # survive a reconnect, so it pulls through this start/poll pair rather than
    # one blocking ``transport.run`` (a dropped SSH channel mid-multi-GB pull
    # would otherwise take the whole deploy down). The pull runs as a
    # backgrounded systemd oneshot (``systemctl start --no-block``) whose
    # container writes a progress JSON through the model-cache mount; callers
    # read that HOST file and the unit's state through `pull_verdict`, which
    # owns the order. On the controller the store is touched only by the root
    # bridge's own no-follow file operations (VD-143), never by an argv.
    #
    # The progress-file contract is documented once, beside the program, in
    # :mod:`vaelor.gpu_pull_program`; the runtime never fabricates a percent.

    def pull_unit_name(self, repo: str) -> str:
        """The pull oneshot for a validated repo: `gpu_pool_units` spells it."""
        return pull_unit_name(self._model(repo))

    def _progress_path(self, repo: str) -> str:
        return f"{_PULL_PROGRESS_ROOT}/{repo_slug(self._model(repo))}.json"

    def _hub_cache_dir(self, repo: str) -> str:
        """The Hugging Face hub directory a repo's weights live in under the cache.

        The HF cache lays a repo out as ``<HF_HOME>/hub/models--org--name``, so
        this is derived from the validated repo and is what ``remove`` deletes,
        strictly under the cache root so an ``rm -rf`` cannot be aimed elsewhere.
        """
        model = self._model(repo)
        return f"{MODEL_CACHE_ROOT}/hub/models--" + model.replace("/", "--")

    _require_pull_unit = staticmethod(require_pull_unit)

    def render_model_pull(
        self, *, repo: str, revision: Optional[str] = None,
        pull_program: Optional[str] = None, vllm_image: Optional[str] = None,
    ) -> Tuple[str, str]:
        """``(unit, text)`` of the oneshot that pulls ``repo`` into the node cache.

        The pull runs the image's own Python on :data:`PULL_SCRIPT` and needs no
        GPU devices, so it uses a plain ``docker run`` rather than the
        accelerator prefix. ``vllm_image`` is the pinned image it runs in - the
        deployment's own, so a deploy never brings a second image onto a
        machine to fetch weights; a pull that names none runs in the image a
        pull always ran in (`vllm_images.RECORDED_IMAGE_KEY`). Without ``pull_program`` (a worker) the program is
        read from the store's pull directory, where `_ensure_pull_script` put
        it; with one (the controller's root bridge, VD-143) that host file is
        mounted read-only and run instead, so the program a root container runs
        is never one an account other than root could have written.
        """
        model = self._model(repo)
        pinned = self._revision(revision)
        slug = repo_slug(model)
        container = pull_container_name(model)
        command = [
            DOCKER, "run", "--rm", "--name", container,
            "--network", "host",
            "-e", f"HF_HOME={MODELS_MOUNT}",
            "-v", f"{MODEL_CACHE_ROOT}:{MODELS_MOUNT}",
        ]
        program = _CONTAINER_PULL_SCRIPT
        if pull_program is not None:
            host = str(pull_program)
            if not re.fullmatch(_PROGRAM_PATH, host) or ".." in host:
                raise ValueError("The model pull program path is invalid.")
            command += ["-v", f"{host}:{_CONTAINER_PACKAGED_PROGRAM}:ro"]
            program = _CONTAINER_PACKAGED_PROGRAM
        command += [
            "--entrypoint", PYTHON_ENTRYPOINT,
            image_profile(RECORDED_IMAGE_KEY if vllm_image is None else vllm_image).image,
            program,
            "--model", model,
            "--cache-dir", MODELS_MOUNT,
            "--progress-file", f"{_CONTAINER_PULL_ROOT}/{slug}.json",
        ]
        if pinned is not None:
            command += ["--revision", pinned]
        return pull_unit_name(model), pull_unit_text(
            description=f"Vaelor model pull ({model})",
            container_name=container, exec_start=" ".join(command),
        )

    def start_model_pull(
        self, transport, *, repo: str, revision: Optional[str] = None,
        image: Optional[str] = None,
    ) -> Dict[str, str]:
        """Start a backgrounded oneshot that pulls ``repo`` into the node cache.

        Returns ``{unit, progress_path}`` for the poll to follow. ``--no-block``
        so the call returns immediately and systemd supervises the multi-GB
        fetch rather than the SSH channel. The previous run's progress file is
        deleted first, so a poll can never read an earlier pull's stale
        ``done`` as this one's state - over SSH on a worker, and inside the root
        bridge's install on the controller, which also makes the store's pull
        and compile directories itself.
        """
        model = self._model(repo)
        values = {"repo": model, "revision": self._revision(revision)}
        # ``image`` is the pinned image the pull runs in (a `vllm_images`
        # key), which the caller has already made present (`ensure_image`):
        # nothing here fetches one. It is named in the values only when it is
        # not the image an unnamed pull runs in, so a pull in that image
        # crosses to an older bridge exactly as it always did.
        if image is not None and image_profile(image).key != RECORDED_IMAGE_KEY:
            values[IMAGE_FIELD] = image_profile(image).key
        if not root_renders_units(transport):
            self._ensure_cache_dirs(transport)
            transport.run(["rm", "-f", self._progress_path(model)], sudo=True)
            self._ensure_pull_script(transport)
        unit = self._install_unit(transport, MODEL_PULL_KIND, values)
        return {"unit": unit, "progress_path": self._progress_path(model)}

    @staticmethod
    def _ensure_cache_dirs(transport) -> bool:
        """Make a WORKER's model store and its pull and compile directories,
        without re-moding the store.

        On a worker the store is Vaelor's to make; on an appliance it is already
        the shared ``root:vaelor-jobs 3770`` directory (`vaelor.state_root_layout`),
        and ``install -d -m 0755`` RESETS an existing directory's mode, dropping
        the setgid bit and group write the workload executor depends on. So the
        mode is only ever set on a directory this call CREATES; the pull and
        compile SUBdirectories are Vaelor's alone everywhere and are always
        ``install``ed - the compile cache (:data:`COMPILE_CACHE_ROOT`) here
        rather than left to docker's implicit bind-mount creation, so its
        ownership is decided in the same place as the pull directory's. The
        controller's store is never touched from here: the root bridge makes
        its folders itself (VD-143).

        **The probe needs ``sudo``**: ``/var/lib/vaelor`` grants nothing to an
        account outside group ``vaelor``, and the enrolled login is one, so an
        unprivileged ``stat`` gets EACCES - which reads here as "absent" and sent
        the fallback on to reset an appliance's store to ``root:root 0755``.

        **A store this call creates is root-owned**: a runtime cannot know the
        installer's owner and group (a worker has no ``vaelor-workloads`` user).
        The deploy's own containers run as root and use it fine, and an
        installer-made store - the shape Mode A's executor needs - is untouched.
        """
        make = ["install", "-d", "-m", "0755"]
        created = True
        try:
            transport.run(["stat", "-c", "%a", MODEL_CACHE_ROOT], sudo=True)
            created = False
        except SshTransportError:
            transport.run(make + [MODEL_CACHE_ROOT], sudo=True)
        transport.run(make + [_PULL_PROGRESS_ROOT], sudo=True)
        transport.run(make + [COMPILE_CACHE_ROOT], sudo=True)
        return created

    @staticmethod
    def _ensure_pull_script(transport) -> str:
        """Put :data:`PULL_SCRIPT` on a WORKER under its content-hashed name.

        The name carries the script's own digest, so a new version is a new file
        and can never truncate the one a still-running pull container is reading.
        The probe-then-write is `vllm_entry_program.write_program_on_worker`'s.
        """
        return write_program_on_worker(
            transport, _PULL_SCRIPT_PATH, PULL_SCRIPT, PULL_SCRIPT_DIGEST,
        )

    def read_pull_progress(self, transport, repo: str) -> Dict[str, Any]:
        """The pull script's last progress object, or ``{}`` when none is readable.

        A missing file (the container has not written yet) or unparseable content
        is an empty dict, never a guessed percent - the caller reports the honest
        phase. The container writes through the ``/models`` mount and runs as
        root, so the HOST file is read as root: with ``sudo`` over SSH, and by
        the bridge's own no-follow open on the controller (VD-143), where a
        link planted in the store must not become a root read of another file.
        """
        try:
            if root_renders_units(transport):
                raw = transport.read_pull_progress(self._model(repo))
            else:
                raw = transport.run(["cat", self._progress_path(repo)], sudo=True)
        except SshTransportError:
            return {}
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def pull_unit_status(self, transport, unit: str) -> Dict[str, str]:
        """The pull unit's systemd state, as a small property dict.

        One ``systemctl show`` (exit 0 even for a failed unit, unlike
        ``is-active``) so a poll can tell running from clean from failed without
        a raising probe.
        """
        self._require_pull_unit(unit)
        return self._show(transport, unit, _PULL_UNIT_PROPERTIES)

    @staticmethod
    def _show(transport, unit: str, properties: str) -> Dict[str, str]:
        """One ``systemctl show --property=…`` of ``unit``, as a property dict.

        The one spelling of that read and of its parse, for the pull oneshot
        and the serving unit alike; the callers own WHICH properties they ask
        for and what the answer means.
        """
        raw = transport.run(["systemctl", "show", unit, f"--property={properties}"])
        parsed: Dict[str, str] = {}
        for line in str(raw).splitlines():
            key, separator, value = line.partition("=")
            if separator:
                parsed[key.strip()] = value.strip()
        return parsed

    def pull_verdict(
        self, transport, unit: str, repo: str,
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """``(outcome, status, progress)`` for one poll, in the ONE right order.

        Both pollers ask this rather than sequencing the two remote reads
        themselves, because the order decides correctness and they had it
        opposite ways round. Progress FIRST loses a pull that completes between
        the round-trips: the file still says ``fetching``, the unit already says
        ``success``, and :func:`pull_outcome`'s contradiction rule then records
        an error over a fully cached model. So the unit is read first, and a
        contradiction surviving that order re-reads the file ONCE before it is
        believed - the program writes its terminal record atomically, so a
        contradiction that repeats is the real half-finished pull. ``status``
        rides along because a failure's WORDING needs it: an incomplete clean
        exit and a crashed unit read differently.
        """
        status = self.pull_unit_status(transport, unit)
        progress = self.read_pull_progress(transport, repo)
        outcome = pull_outcome(status, progress)
        if classify_pull(status) == "ok" and outcome == "failed":
            progress = self.read_pull_progress(transport, repo)
            outcome = pull_outcome(status, progress)
        return outcome, status, progress

    def stop_pull_unit(self, transport, unit: str) -> None:
        """Stop a pull's oneshot unit, remove it, and clear its state.

        Called once a pull has reached ``ready`` or ``error``; the weights stay
        in the cache, only the transient unit is cleaned up. The ``stop`` is what
        actually ends a still-running job - deleting the unit file would leave
        the container fetching - and reaches the program's SIGTERM handler,
        because the foreground ``docker run`` proxies the signal in. It and
        ``reset-failed`` are tolerated failing: the unit may already be gone.
        """
        self._require_pull_unit(unit)
        for action in ("stop", "reset-failed"):
            try:
                transport.run(["systemctl", action, unit], sudo=True)
            except SshTransportError:
                pass
        transport.run(["rm", "-f", f"{UNIT_DIRECTORY}/{unit}"], sudo=True)
        transport.run(["systemctl", "daemon-reload"], sudo=True)

    def remove_model_cache(self, transport, repo: str) -> str:
        """Delete a repo's cached weights from the node and return the path removed.

        The path is derived from the validated repo and asserted to sit under the
        cache root, so the ``rm -rf`` a worker runs can only remove one model's
        hub directory. On the controller the root bridge removes it itself
        (VD-143), walking the store without following a single link, so a link
        planted anywhere in the store cannot aim the removal outside it.
        """
        cache_dir = self._hub_cache_dir(repo)
        if not cache_dir.startswith(f"{MODEL_CACHE_ROOT}/hub/models--"):
            raise ValueError("The model cache path is invalid.")
        if root_renders_units(transport):
            transport.remove_model_cache(self._model(repo))
        else:
            transport.run(["rm", "-rf", cache_dir], sudo=True)
        return cache_dir
