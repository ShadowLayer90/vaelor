"""Deployment sequencing for the managed local model server.

Split out of :mod:`vaelor.executor` because it is a different job from
dispatching work: this file decides *what to launch and in what order* — size
the runtime, render the Compose file, validate it, pull, replace the running
container, wait for health, and activate the credential — while
:mod:`vaelor.model_start_verification` asks what actually came up afterwards.
The two were interleaved in one method, and the seam between them is where the
redeploy defect lived: a reading taken at the wrong point in the sequence.

The rollback rule is the one thing to preserve when editing here. Every failure
between writing the new Compose file and reporting success restores the
previous one and brings it back up, because a half-replaced model server is an
appliance with no assistant and no explanation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .copilot_setup import local_runtime_settings
from .credential_broker import CredentialBrokerClient, CredentialError
from .hardware_bridge import HardwareBridgeError
from .executor_network import available_model_port, parse_memory_usage
from .inference_tuning import (
    DEFAULT_MODEL_FLAGS,
    RECOMMENDED_RUNTIME_MODE,
)
from .model_sizing import loaded_model_capacity
from .job_control import JobCancelled
from .managed_local_credentials import activate_managed_chat, activate_managed_local
from .model_catalog import catalog_engine, catalog_surface
from .model_service_compose import accelerator_plan, build_model_compose
from .model_start_verification import accelerator_baseline, verify_started_model


#: The broker label a managed-local credential carries, and the fallback detail
#: when its connection test fails without a message. One home each, because both
#: deploy paths write them - the llama.cpp path and the flm-real NPU path - and a
#: sentence written in two places in one module is the shape #98 records
#: (LESSONS 6 / VD-090). `{}` is filled with the model's short name.
MANAGED_LOCAL_CREDENTIAL_LABEL = "Managed local model · {}"
CONNECTION_TEST_FAILED_DETAIL = "connection test failed"


#: The compose project the stock GPU AI-Chat server runs in, kept DISTINCT from
#: the Assistant's ``model-assistant`` (#247m): a non-NPU GPU box may run both,
#: so they must not share a project, and keeping them apart lets GPU mutual
#: exclusion be *targetable* - retirement ``compose down``s this without touching
#: the Assistant's.
GPU_CHAT_COMPOSE_PROJECT = "model-chat"


#: Boot-time reconcile (VD-001): how many times, and how long apart, the NPU
#: autostart re-tries a launch that failed only because the privileged hardware
#: bridge's socket was not yet accepting. systemd `After=` orders the bridge's
#: START before the executor's, but NOT the moment its Unix socket is bound, so
#: at boot the first `flm_start` can lose that race by a beat. A bounded wait
#: (6 x 5s = 30s, well under a minute) rides the gap out without stalling the
#: job loop, which is why the whole reconcile runs off the loop's critical path.
#: A launch that fails for any OTHER reason - notably flm-real starting but never
#: becoming healthy - is NOT retried; the supervisor already stopped it.
NPU_AUTOSTART_ATTEMPTS = 6
NPU_AUTOSTART_RETRY_SECONDS = 5.0


def _loopback_port(base_url: str) -> Optional[int]:
    """The loopback port carried in a managed-local credential's ``base_url``.

    The credential is ``http://127.0.0.1:PORT/v1`` (or the IPv6
    ``http://[::1]:PORT/...``). Reusing that exact port on relaunch is the whole
    point of the VD-001 reconcile: flm-real comes back on the SAME port the
    stored credential already names, so the credential stays valid with no
    re-registration and no churn. Returns None for anything outside the
    unprivileged range (1024-65535) rather than handing the supervisor a port it
    would reject anyway - and None is the non-fatal "nothing to supervise" answer
    everywhere it is read.
    """
    try:
        port = urllib.parse.urlparse(str(base_url or "")).port
    except ValueError:
        return None
    if port is None or not 1024 <= port <= 65535:
        return None
    return port


def _artifact_identity(model: Path) -> Dict[str, str]:
    """The `repo` and `file` that name this artifact in the catalog.

    Recovered from where the download path put it: `executor.py` writes into
    `models_root / repo.replace("/", "--")`, so the directory name reverses to
    the repository. Reversing the convention keeps this in step with the one
    place that establishes it - and a wrong guess is not dangerous, because
    `identify()` resolves through the catalog and returns None for a pair it
    does not recognise, which is exactly the unmeasured case.

    Returns a mapping so the call site can splat it: an artifact outside the
    managed layout contributes nothing rather than a misleading half-identity.
    """
    parent = model.parent.name
    if not parent or "--" not in parent:
        return {}
    return {"repo": parent.replace("--", "/"), "file": model.name}


class ExecutorModelDeployMixin:
    """Deploy, replace, and roll back the managed llama.cpp service."""

    def _restore_model_service(
        self, project: Path, compose_file: Path, previous_content: Optional[str]
    ) -> None:
        if previous_content is None:
            self._compose(project, "down", "--remove-orphans", timeout=120)
            compose_file.unlink(missing_ok=True)
            return
        temporary = project / ".compose.rollback.tmp"
        temporary.write_text(previous_content, encoding="utf-8")
        temporary.chmod(0o660)
        temporary.replace(compose_file)
        self._compose(project, "up", "-d", "--remove-orphans", timeout=180)

    def _accelerator_baseline(
        self, project: Path, *, replacing: bool
    ) -> Optional[int]:
        """Accelerator memory in use with no managed model of ours resident.

        **The previous container is stopped first.** ``compose up`` replaces it,
        so a baseline read before that still counts the outgoing model's
        ~5.9 GB — the new model holds the same amount, the delta is zero, and a
        healthy redeploy tells the owner "Running on CPU". That is the
        accelerator check inverted: asserting a degradation it has not
        established. Only a redeploy needs the stop; a first deploy has nothing
        of ours resident to clear.

        It is taken after ``pull`` rather than before, so the outage is the
        container replacement that was going to happen anyway and not the image
        download in front of it.
        """
        if replacing:
            self._compose(project, "down", "--remove-orphans", timeout=120)
        return accelerator_baseline(
            self._model_hardware_budget().get("accelerators")
        )

    def _compose_stats(self, containers: list) -> list:
        """One `docker stats` memory reading per container, as raw strings.

        `--no-stream` so it samples once and exits; without it `docker stats`
        streams forever and the deploy never gets past sizing.
        """
        docker = shutil.which("docker")
        if docker is None or not containers:
            return []
        result = subprocess.run(
            [docker, "stats", "--no-stream", "--format", "{{.MemUsage}}"]
            + [str(container) for container in containers],
            capture_output=True, check=False, text=True, timeout=30,
        )
        if result.returncode != 0:
            return []
        return [line for line in (result.stdout or "").splitlines() if line.strip()]

    def _outgoing_model_bytes(self, project: Path) -> Optional[int]:
        """Memory the model this deploy is replacing is still holding.

        Returns 0 when nothing of ours is running, and ``None`` when the
        reading could not be taken. Those were both 0 until 2026-08-11, which
        made them indistinguishable at the one point where they mean opposite
        things: 0 says the machine's free memory is already correct, and
        ``None`` says it is understated by an unknown amount. Collapsing them
        let an unreadable machine be sized as though it were an idle one.

        `docker stats` under memory pressure is the case this exists for, and
        `subprocess.TimeoutExpired` is not an `OSError` - so the timeout that
        this method's own 20 and 30 second bounds exist to produce was the one
        exception the handlers below did not catch, and it escaped a caller
        that has no handler either.

        **Actual usage, never the configured limit.** A cgroup limit is a cap,
        not a reservation, so what the replacement hands back is what the
        container is *using* - and under VD-073 a model that has been idle for
        fifteen minutes is using almost none of its limit. Adding the limit
        back would invent headroom that does not exist and size the next model
        past what the machine can hold.

        **No compose file means no project, which means nothing to subtract.**
        Sizing runs before ``_deploy_model`` writes ``compose.yaml`` or makes
        the project directory, so on the first deploy on a clean box there is
        no file for ``docker compose ps`` to read: it exits non-zero, ``_compose``
        raises ``ValueError``, and the handler below turns that into the ``None``
        that refuses the deploy. That is the first-deploy regression the Pi
        caught - a genuinely idle machine (0 to subtract) reported as an
        unmeasurable one. The discriminator is the compose file, not the ``ps``
        exit code: absent file -> 0 and sizing proceeds; a present file whose
        ``ps``/``stats`` still fails -> ``None`` (VD-073, a running model whose
        size could not be read must not be sized as 0).
        """
        if not (project / "compose.yaml").is_file():
            return 0
        try:
            listed = self._compose(project, "ps", "-q", timeout=20)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
            return None
        containers = [line.strip() for line in str(
            listed.get("output", "")
        ).splitlines() if line.strip()]
        if not containers:
            return 0
        try:
            stats = self._compose_stats(containers)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
            return None
        return sum(parse_memory_usage(line) for line in stats)

    def _sizing_budget(
        self, hardware: Dict[str, Any], project: Path
    ) -> Dict[str, Any]:
        """Hardware facts corrected for the model that is about to go away.

        Sizing reads the machine's available memory, and at this point in the
        sequence the outgoing llama-server still holds its full footprint.
        Measured on the appliance on 2026-08-09: redeploying the same 4B saw
        2,422 MiB available where 5,727 MiB was free a minute later, and sized
        an 8,192-token window down to 4,096 without being asked to.

        The window it landed on turned out to be the right one - measuring the
        standing brief the next day put it at 1,344 tokens, not the 4,668 the
        constant claimed (VD-075). That does not make the reading sound: it
        arrived at a defensible window through a memory figure that happened to
        be wrong by 3.3 GB, and the same reading on a different day lands
        somewhere arbitrary. What is under test here is the reading.

        This is the hazard :meth:`_accelerator_baseline` records one method up,
        and it solves it by stopping the container before it reads. Sizing
        cannot do that: the Compose file it produces is what ``up`` is given,
        and stopping before the image pull would put the pull inside the
        outage. So the reading is corrected rather than retimed.

        The correction only ever *adds*, and only what a running container of
        ours is measurably using. **When the reading fails, the result is not
        the conservative sizing this had before** - that sentence stood here
        until 2026-08-11 and was wrong in the direction that matters. An
        uncorrected reading is the *depressed* one. On the appliance that is
        2,422 MiB against a real 5,727, and it does not size the window down:
        it refuses the deploy outright, with "This model does not fit while
        preserving memory for the operating system", on a machine that is
        running that very model at that very window while it says so.

        So a reading that could not be taken is marked rather than assumed,
        and the caller reports the measurement it could not make instead of
        telling the owner their machine is too small.

        **Expected observable, not a bug (#205 finding 5).** Redeploying the
        *identical* model raises the rendered container limit once - a live pass
        saw 4352M on the first deploy step up to 5120M on the redeploy. This is
        this correction working, not drift: the first deploy sizes from raw free
        memory (nothing of ours is running, so :meth:`_outgoing_model_bytes`
        returns 0), while a redeploy adds back the outgoing model's measured
        steady-state footprint and so sizes from a larger, truer free figure.
        It is a one-time step to steady state, not per-redeploy growth - a third
        identical redeploy measures the same resident footprint and lands on the
        same limit. Intentional; left as-is with this note.
        """
        outgoing = self._outgoing_model_bytes(project)
        if outgoing is None:
            unverified = dict(hardware)
            unverified["memory_reading_unverified"] = True
            return unverified
        if outgoing <= 0:
            return hardware
        corrected = dict(hardware)
        corrected["memory_available_bytes"] = int(
            hardware.get("memory_available_bytes") or 0
        ) + outgoing
        # Recorded because a wrong correction here is otherwise invisible: it
        # would show up only as a window that is quietly smaller or larger
        # than the machine deserves, which is exactly how the defect this
        # method exists for went unnoticed.
        corrected["replaced_model_bytes"] = outgoing
        return corrected

    def _npu_serving_decision(
        self, payload: Dict[str, Any], hardware: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Whether this deploy routes to the NPU, and the plan/verdict behind it.

        Kept apart from :meth:`_deploy_model` so the routing predicate is
        testable on its own and so the llama.cpp path can assert it was *not*
        taken. Returns the plan and capability verdict alongside the verdict so
        the NPU deploy does not recompute them.
        """
        from .flm_service import discover_npu_serving
        from .flm_supervisor import should_serve_on_npu
        from .inference_tuning import npu_tier_plan

        neural = hardware.get("neural_accelerators") or []
        if not neural:
            # The Pi-isolation short-circuit: no device, no discovery, no NPU.
            return {"serve_on_npu": False, "plan": None, "capability": None}
        plan = npu_tier_plan()
        capability = discover_npu_serving(
            str(plan.get("flm_tag") or ""),
            device_node=str(neural[0].get("device_node") or "") or None,
        )
        return {
            "serve_on_npu": should_serve_on_npu(hardware, payload, plan, capability),
            "plan": plan,
            "capability": capability,
        }

    def _flm_supervisor(self):
        """The NPU supervisor, wired to the privileged bridge.

        A seam so a test can drive :meth:`_deploy_npu_assistant` with a faked
        flm-real. Production launches through the root hardware bridge, which is
        the only account that may run flm-real with CAP_IPC_LOCK (VD-001).
        """
        from .flm_supervisor import BridgeLauncher, FlmSupervisor
        from .hardware_bridge import HardwareBridgeClient

        return FlmSupervisor(BridgeLauncher(HardwareBridgeClient()))

    def ensure_npu_assistant_served(
        self, *, sleep: Callable[[float], None] = time.sleep
    ) -> Optional[Dict[str, Any]]:
        """Relaunch the NPU Assistant after a reboot, on its pinned port (VD-001).

        The counterpart of :meth:`ensure_web_research_started` for the NPU tier,
        and it runs OFF the job loop's critical path for the same reason: flm-real
        is health-gated for up to 90s while the model loads, and blocking the
        executor on that would stall the first job after a reboot for a minute or
        more. flm-real is a child of the hardware bridge; a reboot or fleet
        restart kills it and nothing else brings it back, so the Assistant is DOWN
        after every reboot/deploy until this reconcile - or a manual
        ``model.deploy`` - runs.

        Idempotent, gated, and NON-FATAL, the same contract as the web-research
        autostart:

        * Gated - it acts only when this machine can serve on the NPU (the
          decision below already ANDs a neural accelerator present, the assistant
          surface, capability available, and a usable plan, which is also the
          Pi-isolation short-circuit) AND an active managed-local Assistant
          credential exists to supervise. A Pi, a hosted-provider Assistant, or a
          box that never deployed the NPU Assistant all no-op.
        * Idempotent - a server already answering on the stored port is left
          untouched (relaunching would evict a loaded model to no purpose).
        * Non-fatal - nothing escapes. Any failure returns None so the reconcile
          can never keep the executor from processing jobs.

        Returns a dict describing what it did, or None when it no-oped.
        """
        try:
            hardware = self._model_hardware_budget()
            decision = self._npu_serving_decision({"surface": "assistant"}, hardware)
            if not decision["serve_on_npu"]:
                # Pi isolation, NPU tier unavailable, or an unusable model: the
                # decision already ANDs all three, so there is nothing to serve.
                return None
            try:
                lease = self.credential_broker.resolve_active("deployment-agent")
            except CredentialError:
                # A fresh box that never deployed the NPU Assistant has no active
                # deployment-agent credential - nothing to bring back.
                return None
            from .provider_runtime import managed_local_connection

            if not managed_local_connection(lease):
                # The Assistant is on a hosted provider; nothing local to run.
                return None
            port = _loopback_port(str(lease.get("base_url") or ""))
            if port is None:
                return None
            plan = decision["plan"]
            # Positively identify THIS NPU tier's credential, do not just fall
            # back (VD-001, FIX 2). Since a63 the NPU credential is pinned to the
            # served flm tag, so the pinned model IS the discriminator: a
            # llama.cpp Assistant credential has selected_model=="" (resolve_active
            # returns model==""), and a foreign non-empty tag is some other
            # model. Relaunching flm-real onto this port for an empty or foreign
            # tag would serve the wrong model on a stranger's port, so no-op
            # unless the pin equals the plan's flm_tag.
            pinned = str(lease.get("model") or "").strip()
            if not pinned or pinned != str(plan["flm_tag"]):
                return None
            tag = pinned
            ctx_len = int(plan["context_tokens"])
            supervisor = self._flm_supervisor()
            if supervisor.healthy(port):
                return {"served": False, "reason": "already-healthy", "port": port}
            return self._launch_npu_with_boot_retry(
                supervisor, tag=tag, ctx_len=ctx_len, port=port, sleep=sleep
            )
        except Exception:
            # Never let a boot-time reconcile keep the executor from processing
            # jobs, under ANY error (FIX 4). The reachable set is wider than it
            # looks - resolve_active can raise sqlite3.OperationalError (DB locked
            # or absent at boot), and plan dict access / int() can raise
            # KeyError/TypeError - so this catches broadly rather than by a tuple
            # that a future seam could outgrow. Not BaseException: a
            # KeyboardInterrupt/SystemExit must still stop the process. The inner
            # `except CredentialError` above keeps the no-active-credential no-op.
            return None

    def _launch_npu_with_boot_retry(
        self, supervisor: Any, *, tag: str, ctx_len: int, port: int,
        sleep: Callable[[float], None],
    ) -> Dict[str, Any]:
        """Serve flm-real on ``port``, retrying ONLY the transient socket race.

        The retry taxonomy is by ``__cause__`` (FIX 3), because the bridge client
        raises the same `HardwareBridgeError` type for two very different things:

        * TRANSIENT - the bridge socket is not yet accepting at boot. The client
          builds it as ``raise HardwareBridgeError(...) from OSError/TimeoutError``
          (`_request`'s `except (OSError, TimeoutError)`), so ``__cause__`` is an
          `OSError` (`TimeoutError` subclasses it). This is the boot race
          systemd `After=` cannot close - it orders start, not socket readiness -
          so wait and try again, bounded by :data:`NPU_AUTOSTART_ATTEMPTS`.
        * PERMANENT - a bridge-side rejection (bad tag, missing binary): a
          `HardwareBridgeError` with NO `__cause__`. Retrying 30s changes
          nothing, so re-raise at once to the non-fatal outer catch.

        A plain ``RuntimeError`` from ``serve`` (flm-real started but never became
        healthy; the supervisor already stopped it) is likewise not retried - it
        is not a `HardwareBridgeError` and falls straight through. The final
        exhausted transient attempt re-raises, which the caller swallows.
        """
        last_error: HardwareBridgeError
        for attempt in range(NPU_AUTOSTART_ATTEMPTS):
            try:
                served = supervisor.serve(tag, ctx_len=ctx_len, port=port)
                return {"served": True, **served}
            except HardwareBridgeError as error:
                if not isinstance(error.__cause__, OSError):
                    # Permanent bridge-side failure, not the socket race: no retry.
                    raise
                last_error = error
                if attempt + 1 < NPU_AUTOSTART_ATTEMPTS:
                    sleep(NPU_AUTOSTART_RETRY_SECONDS)
        raise last_error

    def _deploy_npu_assistant(
        self, payload: Dict[str, Any], decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Serve the Assistant on the NPU via flm-real (VD-001 / VD-002).

        The counterpart of the llama.cpp path for a machine that has the NPU
        tier: launch flm-real through the privileged bridge, wait for it to
        answer, register the endpoint as the managed local connection - the same
        credential shape the llama.cpp path registers, so `chat_inference`
        treats it as managed-local with no change - and only then retire the
        llama.cpp assistant container.

        **The order is the fail-safe (F1).** flm-real is launched and confirmed
        HEALTHY before the llama.cpp assistant container is touched, so if it
        never comes up - the exact outcome if the launch environment, the
        ``serve`` flag names or the health-endpoint assumption is wrong on the
        hardware - llama.cpp is left running and the deploy raises an honest
        error. The Assistant is never left with no server. flm-real takes a
        freshly allocated loopback port rather than the payload's, which on the
        refresh path is the *llama.cpp* container's port: the two must coexist
        until flm-real is proven healthy.
        """
        plan = decision["plan"]
        tag = str(plan["flm_tag"])
        ctx_len = int(plan["context_tokens"])
        supervisor = self._flm_supervisor()
        port = supervisor.allocate_port(0)
        project = self._project_directory("model-assistant")
        self._checkpoint(45, "Starting the neural processor model server", "starting")
        # If flm-real never answers, `serve` stops it and raises here - before
        # anything below runs, so the llama.cpp container is still serving.
        served = supervisor.serve(
            tag, ctx_len=ctx_len, port=port,
            announce=lambda detail: self._checkpoint(75, detail, "starting"),
        )
        endpoint = served["endpoint"]
        candidate_id = "cred_managed_local_{}".format(
            hashlib.sha256(
                "npu:{}:{}:{}".format(tag, port, time.time_ns()).encode()
            ).hexdigest()[:12]
        )
        broker = CredentialBrokerClient()
        # Register the profile already pinned to the served tag, NOT model="".
        # The connection test below fires before `select_model`, and a reader
        # that finds an empty model falls back to flm-real's FIRST catalog entry,
        # deepseek-r1:8b. On a box where deepseek is not already cached (every
        # clean install), that empty-model test asks flm-real to serve a tag it
        # does not have, triggering a multi-GB HuggingFace download that the test
        # then times out on - the deploy fails with "chat API failed" even though
        # the served model is healthy (reproduced live on the Z2, VD-111). Pinning
        # the served tag up front makes the test target the loaded model, and the
        # explicit select_model below keeps the pin unambiguous.
        profile = json.dumps(
            {"base_url": endpoint, "model": tag, "api_key": ""},
            separators=(",", ":"),
        )
        # Rollback discipline, the same one the llama.cpp path keeps: a failure
        # after flm-real is up must not leave it running unregistered. Any
        # credential error stops the server and removes the half-written
        # credential before re-raising, so a failed deploy leaves no orphan.
        try:
            credential = broker.put(
                "openai-compatible",
                MANAGED_LOCAL_CREDENTIAL_LABEL.format(str(plan.get("model") or tag)[:48]),
                profile,
                credential_id=candidate_id,
            )
            connection_test = broker.test(credential["id"])
            if not connection_test.get("ok"):
                broker.delete(credential["id"])
                supervisor.stop()
                raise RuntimeError(
                    "flm-real started, but its chat API failed: {}".format(
                        connection_test.get("message", CONNECTION_TEST_FAILED_DETAIL)
                    )
                )
            # Pin the credential to the model flm-real is actually serving.
            # The profile is registered with model="" (FLM serves one loaded
            # model and answers /models with its whole catalog), and every
            # downstream reader that finds an empty model falls back to the
            # FIRST catalog entry - which on flm-real is deepseek-r1:8b, a
            # reasoning model. Two failures follow from that one empty string,
            # both reproduced live on the Z2 (VD-108): its chat template ignores
            # enable_thinking:False, so the Assistant's own _model_answer leaks
            # thinking-out-loud into every reply (leak 3/3 on the deepseek tag,
            # 0/3 on qwen3.5:4b); and _model_billions("") == 0 makes
            # model_capability label the 4B NPU model "Basic local intelligence".
            # Pinning the served tag is what makes the Assistant answer from the
            # NPU model cleanly, so it happens before activation - the credential
            # is never active-but-unpinned.
            broker.select_model(credential["id"], tag)
            # activate_managed_local capability-gates its fresh-box ai-chat
            # adoption on the accelerator inventory it discovers: a
            # dual-accelerator Z2 (gfx1151 GPU) keeps ai-chat off the NPU
            # Assistant for the user's own GPU pick (VD-007/VD-042).
            activate_managed_local(broker, credential["id"])
        except CredentialError:
            try:
                CredentialBrokerClient().delete(candidate_id)
            except CredentialError:
                pass
            supervisor.stop()
            raise
        # flm-real is healthy and is now the Assistant's endpoint, so the
        # llama.cpp assistant container is a second server no longer needed and
        # is retired (the GPU chat tier is a different project and is untouched).
        # This runs ONLY after the NPU tier is proven serving (F1), and a
        # teardown failure is not fatal now that the Assistant already has a
        # working endpoint.
        self._checkpoint(90, "Retiring the llama.cpp assistant container")
        self._retire_assistant_container(project)
        self._checkpoint(95, "Neural processor model server is healthy", "starting")
        return {
            "project": project.name,
            "endpoint": endpoint,
            "health": served["health_url"],
            "context": ctx_len,
            "context_reason": str(plan.get("context_reason") or ""),
            "model": str(plan.get("model") or ""),
            "flm_tag": tag,
            "backend": "flm-real",
            "device": "npu",
            # The Assistant runs on the neural processor, orchestrated through
            # structured JSON intent (FLM has no usable native tool calling).
            "native_tool_calling": bool(plan.get("native_tool_calling")),
            "runtime": "flm-real",
            "assistant_active": True,
            "credential_id": credential["id"],
            "loaded_models": loaded_model_capacity(
                self._model_hardware_budget()
            ),
        }

    def _retire_assistant_container(self, project: Path) -> None:
        """Retire the OLD llama.cpp assistant, compose file present or not.

        Runs only after flm-real is proven healthy (F1). The old guard was
        ``if (project/"compose.yaml").is_file(): compose down`` — which SKIPPED
        the teardown when the compose file was absent. That is not a hypothetical
        state: the deployed ``model-assistant`` compose directory was found
        deleted on the appliance while its ``llama.cpp`` container kept running,
        so the skip left a stale llama.cpp assistant serving alongside the new
        NPU flm-real (VD-002). Retirement no longer depends on the file: with it,
        ``compose down`` also tears the network down; without it, the container
        is removed by the assistant project's OWN compose label.

        **This touches ONLY the assistant.** Both branches are scoped to the
        ``model-assistant`` project — the ``compose down`` by
        ``--project-name``, the fallback by an exact
        ``com.docker.compose.project=model-assistant`` label filter. The GPU
        chat tier (gpt-oss llama-server) is a DIFFERENT compose project, still
        needed for AI Chat, and its label never matches this one, so it is never
        stopped or removed. A teardown failure is swallowed: the Assistant
        already has a working NPU endpoint, so a failed retire is not worth
        failing the deploy over.
        """
        try:
            if (project / "compose.yaml").is_file():
                self._compose(project, "down", "--remove-orphans", timeout=120)
            else:
                self._remove_compose_project_containers(project.name)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
            pass

    def _remove_compose_project_containers(self, project_name: str) -> None:
        """``docker rm -f`` every container of ONE compose project, by its label.

        The scalpel for the compose-file-missing case. A compose ``down`` needs
        the file; when it is gone the container is still labelled
        ``com.docker.compose.project=<project_name>`` (Docker Compose sets it on
        every container it starts), so this lists exactly the containers bearing
        THIS project's label and force-removes them. The label filter is an exact
        match, so a differently named project — the GPU chat tier among them — is
        never listed and never touched (VD-002 constraint: retire only the
        assistant, never the chat tier).
        """
        docker = shutil.which("docker")
        if docker is None:
            return
        listed = subprocess.run(
            [
                docker, "ps", "-aq",
                "--filter",
                "label=com.docker.compose.project={}".format(project_name),
            ],
            capture_output=True, check=False, text=True, timeout=30,
        )
        if listed.returncode != 0:
            return
        containers = [
            line.strip() for line in (listed.stdout or "").splitlines()
            if line.strip()
        ]
        if not containers:
            return
        subprocess.run(
            [docker, "rm", "-f", *containers],
            capture_output=True, check=False, timeout=60,
        )

    def _deploy_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # **The GPU-engine (ROCmFP4) model is routed out FIRST, before the NPU
        # decision.** Its FP4 tensors load only in the ROCmFPX fork on the GPU,
        # and it is a distinct AI-Chat tier from the NPU Assistant. The NPU
        # decision below keys on the *presence* of a neural accelerator, not on
        # the requested model, so on a box that has an NPU it would otherwise
        # claim this deploy and re-serve the Assistant's own flm tag instead -
        # the exact misroute a live install hit (the 27B came back as
        # `backend: flm-real, device: npu`).
        #
        # Guarded on a NON-EMPTY path, because the model file is resolved here
        # only to read its catalog engine. A pathless deploy - the NPU assistant
        # reconcile arrives as `{"surface": "assistant"}` with no model file -
        # must NOT be forced through `_model_file` (it would raise before the NPU
        # decision is even reached). A model outside the managed layout has no
        # catalog identity (an empty dict) and is never splatted into
        # `catalog_engine` (whose repo/file are required), so it too falls
        # straight through to the NPU/llama.cpp routing unchanged.
        path = str(payload.get("path", ""))
        model = self._model_file(path) if path else None
        identity: Dict[str, str] = {}
        if model is not None:
            identity = _artifact_identity(model)
            if identity and catalog_engine(**identity) == "rocmfpx":
                return self._deploy_gpu_rocm_chat(payload, model)
        # **Which serving surface this deploy is for - the Assistant, or AI
        # Chat (#247m).** A stock catalog GGUF marked ``surface="ai-chat"`` is
        # served on the GPU as the AI-Chat model through the SAME generic compose
        # path the Assistant uses, differing only in the credential it claims
        # (``ai-chat``, not ``deployment-agent``) and in not being intercepted to
        # the NPU. The surface is DATA on the entry, so there is no per-model
        # branch; a pair the catalog does not stock falls back to the payload's
        # own surface (default ``assistant``), byte-for-byte as before. The
        # resolved surface replaces the payload's, so every reader below - the
        # NPU gate and the sizing - sees one consistent value.
        entry_surface = catalog_surface(**identity) if identity else ""
        surface = entry_surface or str(payload.get("surface") or "assistant")
        payload = {**payload, "surface": surface}
        serve_as_chat = surface == "ai-chat"
        # Route the Assistant to the NPU when this machine serves it (VD-001):
        # a neural accelerator is present, flm-real and the model are installed,
        # and the selected model is fit to run. Every other machine - every
        # Raspberry Pi - falls through to the unchanged llama.cpp path below,
        # byte-for-byte, because the first clause of the decision requires a
        # device the Pi does not have. An ``ai-chat`` surface is exempt from this
        # gate (`should_serve_on_npu` returns False for any non-assistant
        # surface), so a stock GPU chat GGUF is never re-served on the NPU - it
        # takes the compose GPU path below and claims ``ai-chat``.
        decision = self._npu_serving_decision(payload, self._model_hardware_budget())
        if decision["serve_on_npu"]:
            return self._deploy_npu_assistant(payload, decision)
        # The llama.cpp/compose path below needs the model file. It is resolved
        # above only when the payload named a path; a pathless deploy that
        # reaches here (neither GPU nor NPU) resolves it now, which raises the
        # same "no model file" error the original single call did.
        if model is None:
            model = self._model_file(path)
        requested_port = int(payload.get("port") or 0)
        port = requested_port or available_model_port(socket.socket)
        if not 1024 <= port <= 65535 or port in {34001, 34002}:
            raise ValueError("Choose an available port from 1024 to 65535.")
        # An AI-Chat GPU deploy gets its own compose project (#247m), so it
        # never collides with the Assistant's and retirement can target one.
        project = self._project_directory(
            GPU_CHAT_COMPOSE_PROJECT if serve_as_chat else "model-assistant"
        )
        hardware = self._sizing_budget(self._model_hardware_budget(), project)
        if hardware.get("memory_reading_unverified"):
            # Refusing to size beats sizing from a reading known to be wrong in
            # a known direction. The outgoing model is still holding its
            # footprint, so an uncorrected figure is understated, and
            # `local_runtime_settings` would refuse with "This model does not
            # fit while preserving memory for the operating system" - naming
            # the machine's size as the cause when the cause is a measurement
            # that failed. Nothing has been written at this point, so this is
            # also the last place the deploy can stop cleanly.
            raise RuntimeError(
                "Could not measure the model that is currently running, so the "
                "free memory this deploy would size from is understated by an "
                "unknown amount. Nothing was changed. Try again in a moment."
            )
        model_bytes = model.stat().st_size
        # KV quantization was measured on the accelerated backends, not on a
        # CPU appliance. Applying the measured defaults to a Raspberry Pi would
        # be generalising one machine's numbers onto hardware nobody benchmarked,
        # so an unaccelerated host keeps f16 until it has its own measurement.
        measured_cache = (
            DEFAULT_MODEL_FLAGS if hardware.get("accelerators")
            else {"cache_type_k": "f16", "cache_type_v": "f16"}
        )
        runtime = local_runtime_settings(
            hardware,
            model_bytes,
            # No mode named in the request means "the best this machine can
            # afford", not the middle setting. Defaulting to `balanced` pinned
            # every deploy to the middle rung on a machine with a ~34 GB budget,
            # and the selector can only walk downwards from what it is given.
            # (The original argument was that the middle rung is below the
            # appliance's own agent prompt. That rested on the unmeasured 4,668
            # and no longer holds - the walks-downwards defect is the whole
            # reason now, and it was always the load-bearing half.)
            str(payload.get("mode") or RECOMMENDED_RUNTIME_MODE),
            cache_type_k=str(payload.get("cache_type_k", measured_cache["cache_type_k"])),
            cache_type_v=str(payload.get("cache_type_v", measured_cache["cache_type_v"])),
            # This server answers the Assistant: one owner at one keyboard, so
            # one slot holding the whole window. The slot count is *stated*
            # whatever the surface, because the alternative is not "the default"
            # — it is four full copies of the requested window and a container
            # killed for exceeding a limit that was correct for one.
            surface=str(payload.get("surface") or "assistant"),
            # **The identity is what switches the measured footprint on**, and
            # this call - the only one of the three that sets a container limit
            # - was the one not passing it. `local_runtime_settings` falls back
            # to `model_size * 1.25 + overhead` without it, and that estimate is
            # documented in its own body as 18% low in the direction that caps a
            # container below what the process needs.
            #
            # The download preview (`executor.py`) and the catalog fit check
            # (`model_discovery.py`) both named it, so the screen told the owner
            # "Fits, from a measured footprint for this model on this platform"
            # while the deploy that built the container sized it from the guess.
            # Two mechanisms, one question, and the answer that shipped was the
            # worse one.
            #
            # Measured consequence, 2026-08-09: the 4B at 8,192 was given a
            # 5,120 MiB limit and the kernel killed it at 5,024 MiB of anonymous
            # memory after 36 sustained requests - CONSTRAINT_MEMCG, the same
            # signature as the three kills #117 and VD-070 record.
            **_artifact_identity(model),
        )
        project.mkdir(parents=True, exist_ok=True)
        project.chmod(0o2770)
        plan = accelerator_plan(
            hardware,
            model_bytes=model_bytes,
            kv_cache_bytes=int(runtime.get("kv_cache_bytes") or 0),
        )
        content = build_model_compose(
            model_dir=str(model.parent),
            model_name=model.name,
            port=port,
            runtime=runtime,
            plan=plan,
        )
        compose_file = project / "compose.yaml"
        previous_content = (
            compose_file.read_text(encoding="utf-8") if compose_file.exists() else None
        )
        candidate_id = "cred_managed_local_{}".format(
            hashlib.sha256(
                "{}:{}:{}".format(model, port, time.time_ns()).encode()
            ).hexdigest()[:12]
        )
        temporary = project / ".compose.yaml.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(0o660)
        temporary.replace(compose_file)
        try:
            self._checkpoint(20, "Validating local AI configuration")
            self._compose(project, "config", "--quiet", timeout=30)
            self._checkpoint(35, "Downloading llama.cpp server", "downloading")
            self._compose(project, "pull", timeout=900)
            if serve_as_chat and hardware.get("accelerators"):
                # #247m Finding 1: retire the OTHER GPU chat server (the fork
                # 27B) before starting this one, so the two do not both hold GPU
                # memory and OOM the box. Gated on a GPU (a CPU box has no fork
                # to conflict with); never touches the Assistant or the NPU.
                self._retire_gpu_chat_except("compose")
            baseline_gpu_bytes = self._accelerator_baseline(
                project, replacing=previous_content is not None
            )
            self._checkpoint(75, "Starting local AI server", "starting")
            self._compose(project, "up", "-d", "--remove-orphans", timeout=180)
        except (
            JobCancelled,
            OSError,
            RuntimeError,
            ValueError,
            subprocess.SubprocessError,
        ) as error:
            # subprocess.SubprocessError covers TimeoutExpired, which _compose
            # raises on a wall-clock timeout (executor.py). Without it a redeploy
            # that times out during config/pull/up would escape past the restore
            # and leave the AI server half-replaced with no reconcile to heal the
            # llama.cpp path (#Recovery-3), matching the broad rollback except of
            # the sibling GPU deploy and executor _install_template/_lifecycle.
            self._restore_model_service(project, compose_file, previous_content)
            detail = str(error).lower()
            if "port is already allocated" in detail or "address already in use" in detail:
                raise ValueError(
                    "The local AI port became busy. The previous model was restored; retry to choose another port."
                ) from error
            raise
        health_url = "http://127.0.0.1:{}/health".format(port)
        deadline = time.monotonic() + 180
        last_error = ""
        while time.monotonic() < deadline:
            if self._active_job_id and self.store.is_cancelling(self._active_job_id):
                self._restore_model_service(project, compose_file, previous_content)
                raise JobCancelled()
            try:
                with urllib.request.urlopen(health_url, timeout=3) as response:
                    if response.status == 200:
                        endpoint = "http://127.0.0.1:{}/v1".format(port)
                        broker = CredentialBrokerClient()
                        profile = json.dumps({
                            "base_url": endpoint,
                            "model": "",
                            "api_key": "",
                        }, separators=(",", ":"))
                        credential = broker.put(
                            "openai-compatible",
                            MANAGED_LOCAL_CREDENTIAL_LABEL.format(model.stem[:48]),
                            profile,
                            credential_id=candidate_id,
                        )
                        connection_test = broker.test(credential["id"])
                        if not connection_test.get("ok"):
                            broker.delete(credential["id"])
                            self._restore_model_service(
                                project, compose_file, previous_content
                            )
                            raise RuntimeError(
                                "The model started, but its chat API failed: {}".format(
                                    connection_test.get("message", CONNECTION_TEST_FAILED_DETAIL)
                                )
                            )
                        # #247m: an ``ai-chat`` surface claims ``ai-chat`` and
                        # leaves ``deployment-agent`` alone (`activate_managed_chat`,
                        # the same handoff the fork 27B uses), so a stock GPU chat
                        # GGUF becomes the AI-Chat model without stealing the
                        # Assistant's connection. The Assistant surface keeps its
                        # unchanged `activate_managed_local`.
                        if serve_as_chat:
                            activate_managed_chat(broker, credential["id"])
                        else:
                            activate_managed_local(broker, credential["id"])
                        # A running process is not an accelerated one, and a
                        # server that answers is not one that built the window
                        # it was asked for. Both readings are taken from the
                        # server that came up rather than inferred from what was
                        # requested, and both announce themselves where the user
                        # is looking.
                        verified = verify_started_model(
                            backend=plan["backend"],
                            # What was asked for, not what the gates granted.
                            # A ROCm decision that fell back to the CPU image
                            # is a degradation, and reading `expected` off the
                            # granted backend reported it as "no accelerator
                            # was expected" beside a payload that said `rocm`.
                            requested_backend=plan.get("requested_backend", ""),
                            gpu_layers_requested=plan.get(
                                "requested_gpu_layers", 0
                            ),
                            declined_reason=plan.get("reason", ""),
                            accelerators=self._model_hardware_budget().get(
                                "accelerators"
                            ),
                            baseline_bytes=baseline_gpu_bytes,
                            base_url="http://127.0.0.1:{}".format(port),
                            requested_context=runtime["context"],
                            # Both halves of the request. A server that honours
                            # the window and starts four slots of it has not
                            # honoured the request, and the check cannot see
                            # that without knowing what was asked for.
                            requested_slots=runtime["parallel_slots"],
                            announce=lambda detail: self._checkpoint(
                                95, detail, "starting"
                            ),
                        )
                        return {
                            "project": project.name, "path": str(model),
                            "endpoint": endpoint,
                            "health": health_url,
                            "context": runtime["context"],
                            # Per slot, and how many slots — the pair, because
                            # either one alone is a window size that cannot be
                            # turned into a memory figure.
                            "parallel_slots": runtime["parallel_slots"],
                            "parallel_slots_reason": runtime["parallel_slots_reason"],
                            "context_tokens_total": runtime["context_tokens_total"],
                            "threads": runtime["threads"],
                            "mode": runtime["mode"],
                            "requested_mode": runtime["requested_mode"],
                            "memory_limit_mib": runtime["memory_limit_mib"],
                            "adjusted_for_memory": runtime["adjusted"],
                            "cache_type_k": runtime["cache_type_k"],
                            "cache_type_v": runtime["cache_type_v"],
                            "flash_attention": runtime["flash_attention"],
                            "flash_attention_mode": runtime["flash_attention_mode"],
                            "fits_agent_prompt": runtime["fits_agent_prompt"],
                            "context_reason": runtime["context_reason"],
                            "budget_source": runtime["budget_source"],
                            # Which backend was chosen and on what numbers, so
                            # the Assistant and the UI can report the decision
                            # rather than the bare name of a backend.
                            "backend": plan["backend"],
                            "backend_reason": plan.get("reason", ""),
                            "backend_decision": plan.get("decision", {}),
                            # Whether the accelerator is *in use*, not whether
                            # it was requested. `in_use: None` means the check
                            # could not be made and must not render as either
                            # outcome.
                            "acceleration": verified["acceleration"],
                            # What the server built, not what it was asked
                            # for. `unknown` is load-bearing here exactly as
                            # `in_use: None` is above.
                            "context_built": verified["context_built"],
                            "rocm": plan.get("rocm", {}),
                            # Gated on this machine's hardware: a CPU-only Pi
                            # keeps one model, not the Z2's two (#205, VD-071).
                            "loaded_models": loaded_model_capacity(hardware),
                            "runtime": "llama.cpp",
                            # Which surface this server holds. An ai-chat deploy
                            # is the AI-Chat model, not the Assistant, so it holds
                            # ``ai-chat`` and never claims to be the Assistant's
                            # connection (mirrors `_deploy_gpu_rocm_chat`).
                            "surface": surface,
                            "assistant_active": not serve_as_chat,
                            "ai_chat_active": serve_as_chat,
                            "credential_id": credential["id"],
                        }
            except (OSError, urllib.error.URLError) as error:
                last_error = str(error)[:300]
            except CredentialError:
                try:
                    CredentialBrokerClient().delete(candidate_id)
                except CredentialError:
                    pass
                self._restore_model_service(project, compose_file, previous_content)
                raise
            self._checkpoint(90, "Waiting for the model to become ready", "starting")
            time.sleep(3)
        self._restore_model_service(project, compose_file, previous_content)
        raise RuntimeError("The local AI server did not become healthy: {}".format(last_error))
