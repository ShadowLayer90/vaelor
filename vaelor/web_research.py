"""Lifecycle owner for Vaelor's optional, loopback-only search capability."""

from __future__ import annotations

import json
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

from .runtime_paths import data_path


SEARCH_IMAGE = (
    "searxng/searxng@sha256:"
    "79c2be18a18367484474bae9b18a8cd9085114ab3dcd49cac091cad8c548a0a9"
)
SEARCH_URL = "http://127.0.0.1:8888"
SEARCH_PORT = 8888
SEARCH_PROJECT = "system-web-research"
CONFIRMATIONS = {
    "install": "install-guarded-web-research",
    "repair": "repair-guarded-web-research",
    "remove": "remove-guarded-web-research",
}


class WebResearchError(ValueError):
    """A bounded, user-displayable lifecycle failure."""


class WebResearchManager:
    """Install, inspect, repair, and remove the appliance-owned search stack."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
        probe: Callable[[], bool] | None = None,
        live: Callable[[], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = Path(root) if root is not None else Path(
            data_path("workloads"),
        ) / "system-web-research"
        self.runner = runner or self._run
        # Two deliberately different checks (#212 / VD-104): `probe` decides
        # status().ready and requires the search to actually return results;
        # `live` gates install/bring-up and only asks whether the backend is up
        # (cheap /healthz, no engine query). Gating install on results would fail
        # on a transient empty first query AND make the bring-up loop's own query
        # burst trip the very rate-limiter it waits on.
        self.probe = probe or self._probe
        self.live = live or self._live
        self.sleep = sleep

    def plan(self, action: str) -> dict[str, Any]:
        if action not in CONFIRMATIONS:
            raise WebResearchError("Choose install, repair, or remove.")
        removing = action == "remove"
        return {
            "action": action,
            "confirmation": CONFIRMATIONS[action],
            "title": (
                "Remove guarded web research" if removing else
                f"{action.title()} guarded web research"
            ),
            "changes": [
                (
                    "Stop and unregister Vaelor's private search service."
                    if removing else
                    "Run a digest-pinned SearXNG service on this Vaelor node."
                ),
                (
                    "Retain configuration for recovery unless permanent removal is selected."
                    if removing else
                    "Bind it only to 127.0.0.1:8888; it is never exposed to the LAN."
                ),
            ],
            "image": SEARCH_IMAGE,
            "endpoint": SEARCH_URL,
            "data_path": str(self.root),
            "approval_required": True,
            "recovery": (
                "Reinstall from this same setup surface."
                if removing else
                "Repair rewrites the owned definition and restarts only this service."
            ),
        }

    def status(self) -> dict[str, Any]:
        compose = self.root / "compose.yaml"
        settings = self.root / "settings.yml"
        configured = compose.is_file() and settings.is_file()
        pinned = configured and SEARCH_IMAGE in compose.read_text(
            encoding="utf-8", errors="replace",
        )
        ready = bool(configured and pinned and self.probe())
        port_busy = self._port_open()
        if ready:
            state, reason = "ready", "Guarded web research is reachable on loopback."
        elif configured and not pinned:
            state, reason = "degraded", "The installed image no longer matches Vaelor policy."
        elif configured:
            state, reason = "degraded", "The managed service is configured but not responding."
        elif port_busy:
            state, reason = "blocked", "Port 8888 is in use by an unmanaged process."
        else:
            state, reason = "not_installed", "Optional guarded web research is not installed."
        return {
            "state": state,
            "reason": reason,
            "installed": configured,
            "ready": ready,
            "managed": configured,
            "digest_pinned": bool(pinned),
            "endpoint": SEARCH_URL,
            "network_scope": "loopback-only",
            "image": SEARCH_IMAGE,
            "actions": self._actions(state),
        }

    def execute(self, action: str, confirmation: str, *, purge: bool = False) -> dict[str, Any]:
        expected = CONFIRMATIONS.get(action)
        if not expected or confirmation != expected:
            raise WebResearchError("Review the plan and type its exact confirmation.")
        if action == "remove":
            return self._remove(purge=purge)
        # install / repair always rewrite the owned definition and restart, so
        # the pinned digest is re-enforced even over a healthy service.
        self._guard_unmanaged_port()
        return self._bring_up_and_wait()

    def ensure_running(self) -> dict[str, Any]:
        """Idempotently deploy and start the pinned search backend - the enable.

        #212 / owner decision (flipped to auto-start-on-enable): enabling
        web research must bring the search backend up automatically, so a granted
        agent fetches end-to-end with no separate manual container deploy. Unlike
        `execute("install")` this needs no typed confirmation and is a no-op when
        the service is already reachable, so it is safe to call automatically
        (executor start-up recovery). It uses the one owned compose project -
        loopback-only, digest-pinned, cap_drop, the 512 MB cap - and opens no
        second docker mechanism.
        """
        if self.status()["ready"]:
            return self.status()
        self._guard_unmanaged_port()
        return self._bring_up_and_wait()

    def autoprovision(self) -> dict[str, Any] | None:
        """Install the guarded search backend ONCE on a clean box - the auto-enable.

        Owner decision (auto-provision): a fresh appliance should have custom-app
        web research ready without the operator first discovering, then approving,
        the separate "Set up web research" install - the capability was otherwise
        hidden behind a failed research attempt. This performs the same bring-up
        as :meth:`ensure_running` (digest-pinned image, loopback-only bind, marked
        enabled) but needs no typed confirmation, exactly once: a
        ``.autoprovisioned`` marker is written on success, so a later deliberate
        ``remove`` is respected and never fought.

        Idempotent and safe on every reconcile - a no-op once the marker exists or
        the service is already installed by the operator. It raises only what
        ``ensure_running`` raises (notably Docker not yet ready at boot); the
        marker is written only AFTER a successful bring-up, so a transient failure
        leaves it unset and the next reconcile pass retries.
        """
        marker = self.root / ".autoprovisioned"
        if marker.is_file():
            return None
        if self.status()["state"] != "not_installed":
            # Already installed/managed (or the operator removed it after an
            # earlier provision): record that and never re-provision.
            self._atomic_write(marker, "1\n", 0o640)
            return None
        result = self.ensure_running()
        self._atomic_write(marker, "1\n", 0o640)
        return result

    def _guard_unmanaged_port(self) -> None:
        if self._port_open() and not (self.root / "compose.yaml").is_file():
            raise WebResearchError(
                "Port 8888 is already used by another service; nothing was changed."
            )

    def _bring_up_and_wait(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_owned_files()
        # Record intent to run before the container converges, so the executor
        # will re-ensure it after a restart even if this first boot is slow.
        self._mark_enabled()
        self._compose("up", "-d", "--remove-orphans")
        # First boot may need to unpack the pinned image and initialize its
        # internal data on slower edge storage. Keep polling while Docker owns
        # startup instead of reporting a failure while it is still converging.
        # Gate on LIVENESS, not results (#212 / VD-104): a fresh backend that is
        # up but whose first search transiently returns nothing must still count
        # as installed, and polling /healthz (not a search) keeps this 90-query
        # loop from tripping the external engines' rate-limiter. status() below
        # then reports `ready` only once a real search returns results.
        for _ in range(90):
            if self.live():
                return self.status()
            self.sleep(1)
        raise WebResearchError(
            "The private search service did not become ready within 90 seconds. "
            "Its configuration was retained; review the failed operation, then use Repair."
        )

    def is_enabled(self) -> bool:
        """Whether web research is *enabled* (should be running), not merely
        configured. A `remove` that retains configuration for recovery is
        disabled, so auto-start must not resurrect it - this marker is how the
        executor tells the two apart (#212, auto-start-on-enable)."""
        return (self.root / "enabled").is_file()

    def _mark_enabled(self) -> None:
        self._atomic_write(self.root / "enabled", "1\n", 0o640)

    def _clear_enabled(self) -> None:
        marker = self.root / "enabled"
        if marker.exists():
            marker.unlink()

    def _remove(self, *, purge: bool) -> dict[str, Any]:
        # Disable first: clear the marker so the executor's start-up auto-start
        # cannot bring a deliberately-stopped service back (#212).
        self._clear_enabled()
        if (self.root / "compose.yaml").is_file():
            self._compose("down", "--remove-orphans")
        if purge:
            for name in ("settings.yml", "compose.yaml"):
                path = self.root / name
                if path.exists():
                    path.unlink()
            try:
                self.root.rmdir()
            except OSError:
                pass
        result = self.status()
        result["configuration_retained"] = not purge and self.root.exists()
        return result

    def _write_owned_files(self) -> None:
        settings = self.root / "settings.yml"
        if not settings.exists():
            self._atomic_write(settings, self._settings(secrets.token_urlsafe(32)), 0o600)
        self._atomic_write(self.root / "compose.yaml", self._compose_text(), 0o660)

    @staticmethod
    def _settings(secret: str) -> str:
        # #212 / VD-104, verified on the Pi 2026-08-14: SearXNG's default general
        # engines (brave, duckduckgo, google, startpage, qwant) all captcha- or
        # rate-limit-block a self-hosted instance, so `use_default_settings` alone
        # returns `results: []` on every query while the container plainly has
        # internet (Wikipedia infoboxes still resolve). A granted research agent
        # then pulls nothing - the exact #212 symptom, one layer past the
        # backend-absent cause. bing and mojeek were measured to return results
        # headless from this network; the five blockers returned CAPTCHA / "too
        # many requests". Pin the two that work and silence the blockers. The
        # ENABLE of a working engine is load-bearing; disabling the blockers only
        # trims per-query latency and log noise, and a name that drifts across
        # SearXNG versions fails safe (the blocker merely fails gracefully again).
        return (
            "use_default_settings: true\n"
            "server:\n"
            f"  secret_key: \"{secret}\"\n"
            "  limiter: false\n"
            "  image_proxy: false\n"
            "search:\n"
            "  safe_search: 1\n"
            "  formats:\n"
            "    - html\n"
            "    - json\n"
            "engines:\n"
            "  - name: bing\n"
            "    disabled: false\n"
            "  - name: mojeek\n"
            "    disabled: false\n"
            "  - name: duckduckgo\n"
            "    disabled: true\n"
            "  - name: brave\n"
            "    disabled: true\n"
            "  - name: startpage\n"
            "    disabled: true\n"
            "  - name: qwant\n"
            "    disabled: true\n"
            "  - name: google cse\n"
            "    disabled: true\n"
        )

    @staticmethod
    def _compose_text() -> str:
        return f"""services:
  search:
    image: {SEARCH_IMAGE}
    restart: unless-stopped
    ports:
      - \"127.0.0.1:{SEARCH_PORT}:8080\"
    volumes:
      - ./settings.yml:/etc/searxng/settings.yml:ro
    labels:
      io.vaelor.owner: web-research
    cap_drop: [ALL]
    cap_add: [CHOWN, SETGID, SETUID, DAC_OVERRIDE]
    security_opt: [no-new-privileges:true]
    mem_limit: 512m
    cpus: 1.0
"""

    @staticmethod
    def _atomic_write(path: Path, content: str, mode: int) -> None:
        temporary = path.with_suffix(path.suffix + ".next")
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(mode)
        temporary.replace(path)

    def _compose(self, *arguments: str) -> None:
        result = self.runner([
            "docker", "compose", "--project-name", SEARCH_PROJECT,
            "-f", str(self.root / "compose.yaml"), *arguments,
        ])
        if result.returncode:
            detail = " ".join((result.stderr or result.stdout or "").split())[:400]
            raise WebResearchError(detail or "Docker could not manage guarded web research.")

    @staticmethod
    def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(command), capture_output=True, text=True, check=False, timeout=180,
        )

    @staticmethod
    def _probe() -> bool:
        # #212 / VD-104: "ready" must mean the search actually RETURNS RESULTS,
        # not merely that the backend answered 200. The old check required the
        # body be <= 4096 bytes, which only ever passed because the default
        # engines were captcha-blocked and returned `results: []` (a tiny body);
        # once a working engine set is pinned a real response is ~11 KB / 20
        # results (measured on the Pi 2026-08-14) and that cap rejected it as
        # not-ready. Read a generous bound, parse, and require a non-empty
        # results list so a backend that is reachable but returns nothing reports
        # `degraded`, never honest-green over an empty search. This couples
        # readiness to the external engines by design - the whole point of #212
        # is that a granted capability that returns no data is not "ready"; the
        # 90-iteration bring-up loop absorbs a cold first query.
        url = SEARCH_URL + "/search?q=vaelor+health&format=json"
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                if response.status != 200:
                    return False
                raw = response.read(1_048_576)  # 1 MiB bound; a real body is ~11 KB
            body = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError):
            return False
        return isinstance(body, dict) and bool(body.get("results"))

    @staticmethod
    def _live() -> bool:
        # Liveness only (#212 / VD-104): is the backend up and serving? Hits the
        # cheap /healthz endpoint, which does NOT query any search engine - so the
        # 90-iteration bring-up loop cannot trip the external rate-limiter the way
        # a burst of real search queries would. status().ready uses the stricter
        # `_probe` (results present) instead; install success only needs liveness.
        try:
            with urllib.request.urlopen(SEARCH_URL + "/healthz", timeout=3) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    @staticmethod
    def _port_open() -> bool:
        try:
            with socket.create_connection(("127.0.0.1", SEARCH_PORT), timeout=0.2):
                return True
        except OSError:
            return False

    @staticmethod
    def _actions(state: str) -> list[str]:
        if state == "ready":
            return ["repair", "remove"]
        if state == "degraded":
            return ["repair", "remove"]
        if state == "blocked":
            return []
        return ["install"]
