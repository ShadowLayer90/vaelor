"""Lifecycle owner for Vaelor's optional, loopback-only search capability."""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

from .credential_broker import WEB_SEARCH_PURPOSE as WEB_SEARCH_KEY_PURPOSE
from .credential_broker_client import (
    CredentialBrokerClient, CredentialError, broker_did_not_answer,
)
from .runtime_paths import data_path

LOGGER = logging.getLogger(__name__)


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
#: The one service in the owned compose project. A settings change recreates
#: this container and nothing else (VD-222 item 6).
SEARCH_SERVICE = "search"
#: Write the owner's key into settings.yml, or take it out, and recreate the
#: search container. Queued only by the owner's web-search-key routes, never
#: offered by `plan` (VD-222 item 2). The intent travels in the action, so a
#: removal never depends on the credential broker answering.
KEY_SET_ACTION = "search-key-set"
KEY_REMOVE_ACTION = "search-key-remove"
SETTINGS_CONFIRMATION = "apply-guarded-web-research-settings"
#: A set whose key cannot be read fails its job, visibly (adversarial F2).
KEY_UNREAD = (
    "The web search key could not be read from Vaelor's secure storage, so it "
    "was not applied. The search service keeps its earlier settings; save the "
    "key again."
)
#: Said by a process that may not write the search service's files: only the
#: workload executor, which owns them, writes them (adversarial F1).
NOT_THE_WRITER = "The workload executor starts the search service, and it is not up yet."
SETTINGS_UNREADABLE = (
    "The search service's settings.yml cannot be read by this process, so it "
    "was left exactly as it is."
)

# VD-222 item 3. Measured 2026-10-19 from the Z2 on the pinned image
# (SearXNG 2026.7.31+6bfd82705): a temporary instance, 53 engines one at a
# time, six queries, three rounds over 30 minutes. Results per query:
#
#   engine            results  every round?               here
#   yep               20       yes, on topic              on
#   zapmeta           9        yes, on topic              on
#   seznam            7-12     yes, fair                  on
#   duckduckgo web    0-10     best answers; blocked r3   on
#   bing              7-10     yes, often off topic       on
#   bing news         1-9      yes                        on (news)
#   duckduckgo news   1-22     yes                        on (news)
#   yandex            10-15    yes, on topic              OFF: the owner's choice
#   mojeek, qwant     0        "access denied"            off
#   brave, brave.news 0        "too many requests"        off
#   startpage (+news), google news   0, CAPTCHA           off
#
# The keyless mix replaces #212's bing + mojeek (Pi, 2026-08-14); Mojeek now
# refuses the box. Yep and DuckDuckGo also rate-limit the box: Yep answered
# "access denied" on some runs that day, and DuckDuckGo web fails with
# "parsing error" after a few minutes of queries, so the mix leans on more
# than one engine on purpose. Bing keeps the default weight 1: on the mix
# itself (two temporary instances side by side, three rounds of eight queries,
# with and without Yep) Bing at 0.5 put as many on-topic pages in the top five
# as at 1 (112 of 120 both with Yep; 107 against 106 without), so a lower
# weight bought nothing measurable. DuckDuckGo web and Bing often return the
# same page, so their agreement is not independent. Plain "duckduckgo" is a
# second client for the same DuckDuckGo block, so it stays off.
# Default-enabled engines not named here (wikipedia, wikidata, currency, the
# translators) are left as the image ships them; the other refusing engines
# are off by default already. The box writes settings.yml itself and rewrites
# it on every update and install (item 6), so a hand edit is not kept.
SEARCH_ENGINES_ON = (
    "yep", "zapmeta", "seznam", "duckduckgo web", "bing",
    "bing news", "duckduckgo news",
)
SEARCH_ENGINES_OFF = (
    "yandex", "mojeek", "qwant", "duckduckgo", "brave", "brave.news",
    "startpage", "startpage news", "google", "google news", "google cse",
)
#: The engine an owner-added search key turns on (VD-222 item 2). It is only
#: ever written with a key the owner added; queries then also go to Brave.
OWNER_KEY_ENGINE = "braveapi"
#: A search API key is one opaque token. Anything else is refused before it can
#: reach settings.yml, so a key can never add YAML of its own.
SEARCH_KEY_SHAPE = re.compile(r"[A-Za-z0-9_-]{16,200}")
_SECRET_LINE = re.compile(r'^  secret_key: "([A-Za-z0-9_-]{16,200})"$', re.M)
_KEY_LINE = re.compile(
    r'^  - name: "' + OWNER_KEY_ENGINE + r'"\n(?:    .*\n)*?    api_key: "([^"\n]*)"$',
    re.M,
)


#: One sentence per state, read by both `status` and `availability` (VD-221).
READY_REASON = "Guarded web research is reachable on loopback."
UNPINNED_REASON = "The installed image no longer matches Vaelor policy."
NOT_RESPONDING_REASON = "The managed service is configured but not responding."
NOT_INSTALLED_REASON = "Optional guarded web research is not installed."


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
        docker_healthy: Callable[[], bool] | None = None,
        search_key: Callable[[], str | None] | None = None,
        writer: bool | None = None,
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
        # Optional, injected by the executor: is Docker able to start containers
        # at all? When it is not (a wiped/broken data-root), a search backend
        # that is "configured but not responding" is not the search service's
        # fault, and status() should say the true cause instead of blaming it.
        # Default None preserves the original behavior exactly.
        self._docker_healthy = docker_healthy
        # The owner's search key from the credential broker. Each read is an
        # audited lease, so it is read on an explicit action, on the first
        # write of settings.yml, and once per process on the reconcile (a
        # restore or an update restarts the executor) - never on every pass.
        # None means "could not read", and keeps the key settings.yml carries.
        self._search_key = search_key or _broker_search_key
        self._key_synced = False
        # Whether this process may write the owned files. None: only the
        # account that owns the directory may (the executor); the control
        # plane shares the group and must never rewrite them (adversarial F1).
        self._writer = writer

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

    def status(self, *, search: bool = True) -> dict[str, Any]:
        """The service's state. ``search=False`` reads readiness from the
        loopback ``/healthz`` instead of a real query, for a screen that must
        not send anything off this appliance just by opening (VD-221)."""
        compose = self.root / "compose.yaml"
        settings = self.root / "settings.yml"
        configured = compose.is_file() and settings.is_file()
        pinned = configured and SEARCH_IMAGE in compose.read_text(
            encoding="utf-8", errors="replace",
        )
        ready = bool(configured and pinned and (self.probe() if search else self.live()))
        port_busy = self._port_open()
        if ready:
            state, reason = "ready", READY_REASON
        elif configured and not pinned:
            state, reason = "degraded", UNPINNED_REASON
        elif configured:
            state, reason = "degraded", NOT_RESPONDING_REASON
        elif port_busy:
            state, reason = "blocked", "Port 8888 is in use by an unmanaged process."
        else:
            state, reason = "not_installed", NOT_INSTALLED_REASON
        # When the service is not ready because Docker itself cannot start
        # containers (a wiped/broken data-root), the search stack is a victim,
        # not the cause. Override only the human `reason` in that specific case
        # - never the `state`, so the state machine and its tests are untouched
        # - to point the operator at the Repair Docker control (#docker-health,
        # MEMORY: a wiped Docker data-root breaks all containers).
        if (
            state in ("degraded", "blocked")
            and self._docker_healthy is not None
            and not self._docker_healthy()
        ):
            reason = (
                "Docker is not healthy on this node, so the search service "
                "cannot run. Repair Docker on the Apps screen and web research "
                "starts automatically."
            )
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

    def availability(self) -> dict[str, Any]:
        """Whether a search could run now, read WITHOUT searching (VD-221).

        AI Chat's Web search control reads this every time AI Chat opens, and
        the owner's rule is that nothing leaves the appliance for a search
        unless the toggle is on. ``status`` cannot answer it: its readiness
        probe runs a real query against the public engines. This asks only the
        owned files and the loopback ``/healthz``.

        A running service is available whatever its ``enabled`` marker says, as
        ``status`` reads it (review round 1: the setup row said ready while this
        said disabled); a deliberate remove stops the container, so a stopped
        one without the marker is the one "turned off".
        """
        compose = self.root / "compose.yaml"
        configured = compose.is_file() and (self.root / "settings.yml").is_file()
        pinned = configured and SEARCH_IMAGE in compose.read_text(
            encoding="utf-8", errors="replace",
        )
        if not configured:
            state, reason = "not_installed", NOT_INSTALLED_REASON
        elif not pinned:
            state, reason = "degraded", UNPINNED_REASON
        elif self.live():
            state, reason = "ready", READY_REASON
        elif not self.is_enabled():
            state, reason = "disabled", "Guarded web research was turned off on this appliance."
        else:
            state, reason = "degraded", NOT_RESPONDING_REASON
        return {"available": state == "ready", "state": state, "reason": reason}

    def may_write(self) -> bool:
        """Whether this process may write the owned files: the account that owns
        the service's directory (the workload executor), or the nearest existing
        parent before it is made. The control plane shares the directory's group
        and reads status through this class, but never writes (adversarial F1:
        it could not read the 0600 settings.yml and replaced it)."""
        if self._writer is not None:
            return self._writer
        effective_uid = getattr(os, "geteuid", None)
        if effective_uid is None:
            return True  # no POSIX owners on this host (a development machine)
        path = self.root
        while not path.exists() and path.parent != path:
            path = path.parent
        return path.stat().st_uid == effective_uid()

    def _require_writer(self) -> None:
        if not self.may_write():
            raise WebResearchError(NOT_THE_WRITER)

    def execute(self, action: str, confirmation: str, *, purge: bool = False) -> dict[str, Any]:
        self._require_writer()
        if action in (KEY_SET_ACTION, KEY_REMOVE_ACTION) and confirmation == SETTINGS_CONFIRMATION:
            return self.apply_key(remove=action == KEY_REMOVE_ACTION)
        expected = CONFIRMATIONS.get(action)
        if not expected or confirmation != expected:
            raise WebResearchError("Review the plan and type its exact confirmation.")
        if action == "remove":
            return self._remove(purge=purge)
        # install / repair always rewrite the owned definition and restart, so
        # the pinned digest is re-enforced even over a healthy service.
        self._guard_unmanaged_port()
        return self._bring_up_and_wait(search_key=self._search_key())

    def apply_key(self, *, remove: bool) -> dict[str, Any]:
        """Write the owner's key into settings.yml, or take it out, and recreate
        the search container when the file changed (VD-222 item 2).

        A removal writes no key whatever the broker says, so a broker that does
        not answer cannot keep queries going to Brave. A set reads the key, and
        one it cannot read fails the job (KEY_UNREAD) rather than completing.
        A service not yet installed reads the key on its first write; one turned
        off stays stopped and gets the new file for when it is turned back on.
        """
        self._require_writer()
        key = "" if remove else self._search_key()
        if key is None or (not remove and not key):
            raise WebResearchError(KEY_UNREAD)
        if not (self.root / "compose.yaml").is_file():
            return self.status(search=False)
        if not self.is_enabled():
            self._write_owned_files(key)
            return self.status(search=False)
        return self._bring_up_and_wait(search_key=key)

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
        status = self.status()
        if not self.may_write():
            # The control plane: it reads, the executor's reconcile brings the
            # service up. Nothing here writes a file or runs compose.
            if status["ready"]:
                return status
            raise WebResearchError(NOT_THE_WRITER)
        # Once per process (an update, a restore or a reboot restarts the
        # executor), the broker's key is the truth; afterwards the file's.
        key = None if self._key_synced else self._search_key()
        if status["ready"] and not self._settings_stale(key):
            self._key_synced = self._key_synced or key is not None
            return status
        # VD-222 item 6: a ready service whose settings.yml is not what this
        # release writes (an older engine mix) is recreated here, so a console
        # update and an install leave the same file and the same running
        # engines. The key settings.yml already carries is kept.
        self._guard_unmanaged_port()
        result = self._bring_up_and_wait(search_key=key)
        self._key_synced = self._key_synced or key is not None
        return result

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
        if marker.is_file() or not self.may_write():
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

    def _bring_up_and_wait(self, *, search_key: str | None = None) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        changed = self._write_owned_files(search_key)
        # Record intent to run before the container converges, so the executor
        # will re-ensure it after a restart even if this first boot is slow.
        self._mark_enabled()
        # A changed settings.yml is read only when the container starts, and
        # `up -d` alone keeps a running container whose compose did not change
        # (VD-222 item 6). Recreate the search service, and only it: it holds
        # nothing of its own beyond a cache.
        recreate = ("--force-recreate", SEARCH_SERVICE) if changed else ()
        self._compose("up", "-d", "--remove-orphans", *recreate)
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

    def _desired_settings(self, search_key: str | None) -> tuple[str, str]:
        """``(current, desired)`` settings.yml text. The server secret and, when
        ``search_key`` is None, the owner's search key are kept from the current
        file; a missing file gets a new secret. A file that exists and cannot be
        read raises SETTINGS_UNREADABLE: it is never treated as missing, or a
        process without read access would replace it (adversarial F1)."""
        path = self.root / "settings.yml"
        current = ""
        if path.exists():
            try:
                current = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                raise WebResearchError(SETTINGS_UNREADABLE) from error
        secret_match = _SECRET_LINE.search(current)
        secret = secret_match.group(1) if secret_match else secrets.token_urlsafe(32)
        if search_key is None:
            key_match = _KEY_LINE.search(current)
            search_key = key_match.group(1) if key_match else ""
            if not SEARCH_KEY_SHAPE.fullmatch(search_key):
                search_key = ""  # a hand-edited, malformed key is dropped
        return current, self._settings(secret, search_key)

    def _settings_stale(self, search_key: str | None = None) -> bool:
        try:
            current, desired = self._desired_settings(search_key)
        except WebResearchError:
            LOGGER.warning(SETTINGS_UNREADABLE)
            return False
        return current != desired

    def _write_owned_files(self, search_key: str | None = None) -> bool:
        """Write the owned files; True when an existing settings.yml was rewritten.

        settings.yml is rewritten whenever this release would write something
        different (VD-222 item 6: it used to be written only when missing, so an
        existing box kept its first engine mix for good, and a hand edit is now
        replaced on the next update). Mode 0600, since it holds the server
        secret and any owner search key: the executor's account writes it, the
        container's entrypoint reads it as root with DAC_OVERRIDE, and the
        control plane in the directory's group cannot. A first write reads the
        owner's key from the broker, so a key added before web research was
        installed reaches it (adversarial F6). A first write needs no recreate,
        and a compose.yaml change is one `up -d` acts on by itself.
        """
        self._require_writer()
        if search_key is None and not (self.root / "settings.yml").exists():
            search_key = self._search_key() or ""
        current, desired = self._desired_settings(search_key)
        if current != desired:
            self._atomic_write(self.root / "settings.yml", desired, 0o600)
        self._atomic_write(self.root / "compose.yaml", self._compose_text(), 0o660)
        return bool(current) and current != desired

    @staticmethod
    def _settings(secret: str, search_key: str = "") -> str:
        """settings.yml for the measured engine mix above. ``search_key`` turns
        on the owner's key engine; a key not shaped like one token is refused."""
        if search_key and not SEARCH_KEY_SHAPE.fullmatch(search_key):
            raise WebResearchError("The web search key is not in a form Vaelor accepts.")
        lines = [
            "use_default_settings: true",
            "server:",
            f"  secret_key: \"{secret}\"",
            "  limiter: false",
            "  image_proxy: false",
            "search:",
            "  safe_search: 1",
            "  formats:",
            "    - html",
            "    - json",
            "engines:",
        ]
        turned_on = ("    disabled: false", "    inactive: false")
        for name in SEARCH_ENGINES_ON:
            lines += [f"  - name: {json.dumps(name)}", *turned_on]
        for name in SEARCH_ENGINES_OFF:
            lines += [f"  - name: {json.dumps(name)}", "    disabled: true"]
        if search_key:
            lines += [
                f"  - name: {json.dumps(OWNER_KEY_ENGINE)}",
                *turned_on,
                f"    api_key: \"{search_key}\"",
            ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _compose_text() -> str:
        return f"""services:
  {SEARCH_SERVICE}:
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
        # Created with its final mode, never readable wider for a moment
        # (adversarial F9); LF on every host.
        temporary = path.with_suffix(path.suffix + ".next")
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(temporary, flags, mode), "wb") as handle:
            handle.write(content.encode("utf-8"))
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


def _broker_search_key() -> str | None:
    """The owner's web search key from the credential broker: the key, "" when
    none was added, or None when the broker gave no answer (LESSONS 8: an
    unread key is not an absent one, so the caller keeps the current file)."""
    try:
        lease = CredentialBrokerClient(timeout_seconds=10).resolve_active(
            WEB_SEARCH_KEY_PURPOSE,
        )
    except CredentialError as error:
        if broker_did_not_answer(error):
            LOGGER.warning("The web search key could not be read: %s", error)
            return None
        return ""
    except Exception:  # noqa: BLE001 - no socket at all (a test host) is "unread", said
        LOGGER.warning("The web search key could not be read.", exc_info=True)
        return None
    token = str((lease or {}).get("token", "")).strip()
    return token if SEARCH_KEY_SHAPE.fullmatch(token) else ""
