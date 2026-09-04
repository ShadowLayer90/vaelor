"""Short-lived, one-use VNC session tokens for the isolated websockify gateway."""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import socket
import subprocess
import struct
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Optional

from .host_desktop import HostDesktopClient
from .platform_drivers import default_platform_drivers
from .runtime_paths import data_path, env_value

#: The console polls the host status route, so a broker round-trip inside it
#: has to be bounded well under the browser's own 8-second GET timeout. The
#: broker's grdctl call is capped tighter still, in `host_desktop_tls`.
CERTIFICATE_CLIENT_TIMEOUT = 5
#: The shape the console reads when there is no certificate to describe.
#: Emitted as a stable dictionary rather than an omitted key so that "RDP is
#: off" and "the fingerprint could not be read" are different payloads.
NO_CERTIFICATE = {"fingerprint": "", "algorithm": "", "detail": ""}


class VncSessionStore:
    def __init__(self, database_path: Optional[str] = None, lifetime_seconds: int = 90):
        self.database_path = database_path or env_value(
            "VAELOR_VNC_SESSIONS_DB", "PM_VNC_SESSIONS_DB",
            data_path("vnc/sessions.sqlite3"),
        )
        self.lifetime_seconds = max(30, min(int(lifetime_seconds), 300))
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _connect(self):
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o660)
            os.close(descriptor)
        try:
            os.chmod(path, 0o660)
        except PermissionError:
            # The control plane and gateway intentionally share this database
            # through the vaelor-vnc group. A group member may update SQLite
            # but cannot chmod a file owned by the gateway account.
            mode = path.stat().st_mode & 0o777
            if mode & 0o007 or mode & 0o060 != 0o060:
                raise PermissionError(
                    "The browser ticket database does not have a secure "
                    "shared mode."
                )
        connection = sqlite3.connect(str(path), timeout=10)
        connection.row_factory = sqlite3.Row
        self._ensure_schema(connection)
        return connection

    def _ensure_schema(self, connection) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS vnc_sessions (
                    token_hash TEXT PRIMARY KEY,
                    session_id TEXT,
                    actor TEXT NOT NULL,
                    app_id TEXT NOT NULL,
                    target_host TEXT NOT NULL,
                    target_port INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                    , consumed_at INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_vnc_sessions_expiry
                    ON vnc_sessions(expires_at);
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(vnc_sessions)")
            }
            if "session_id" not in columns:
                connection.execute("ALTER TABLE vnc_sessions ADD COLUMN session_id TEXT")
            if "consumed_at" not in columns:
                connection.execute("ALTER TABLE vnc_sessions ADD COLUMN consumed_at INTEGER")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_vnc_sessions_session_id ON vnc_sessions(session_id)"
            )
            connection.commit()
            self._schema_ready = True

    def create(self, actor: str, app_id: str, target_port: int) -> dict:
        port = int(target_port)
        if not 5900 <= port <= 65535:
            raise ValueError("The app does not expose a valid VNC port.")
        token = secrets.token_urlsafe(32)
        session_id = secrets.token_urlsafe(18)
        now = int(time.time())
        expires_at = now + self.lifetime_seconds
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM vnc_sessions WHERE expires_at < ? AND consumed_at IS NULL", (now,))
            connection.execute("DELETE FROM vnc_sessions WHERE consumed_at IS NOT NULL AND consumed_at < ?", (now - 300,))
            connection.execute(
                """
                INSERT INTO vnc_sessions
                    (token_hash, session_id, actor, app_id, target_host, target_port, created_at, expires_at)
                VALUES (?, ?, ?, ?, '127.0.0.1', ?, ?, ?)
                """,
                (self._hash(token), session_id, actor[:64], app_id[:64], port, now, expires_at),
            )
            connection.commit()
        return {"token": token, "session_id": session_id, "expires_at": expires_at * 1000}

    def revoke_all(self, actor: str) -> int:
        """Drop every unconsumed session token this actor holds.

        Called when the owner ends the desktop on the appliance. Without it
        the desktop is gone but the viewing tokens are not, so a URL someone
        kept open would still be accepted for a session that no longer exists
        — a connection to whatever came up on that port next. Returns the
        number of tokens removed so the caller can report honestly rather
        than assume.
        """
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM vnc_sessions WHERE actor = ? AND consumed_at IS NULL",
                (actor[:64],),
            )
            connection.commit()
            return int(cursor.rowcount or 0)

    def consume(self, token: str) -> Optional[tuple[str, int]]:
        if not token or len(token) > 128:
            return None
        now = int(time.time())
        token_hash = self._hash(token)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            row = connection.execute(
                """
                SELECT target_host, target_port
                FROM vnc_sessions
                WHERE token_hash = ? AND expires_at >= ? AND consumed_at IS NULL
                """,
                (token_hash, now),
            ).fetchone()
            if row is not None:
                connection.execute(
                    "UPDATE vnc_sessions SET consumed_at = ? WHERE token_hash = ?",
                    (now, token_hash),
                )
            connection.execute("DELETE FROM vnc_sessions WHERE expires_at < ? AND consumed_at IS NULL", (now,))
            connection.commit()
        if row is None:
            return None
        return str(row["target_host"]), int(row["target_port"])

    def status(self, session_id: str, actor: str) -> dict:
        if not session_id or len(session_id) > 64:
            return {"state": "failed", "message": "The browser desktop session is invalid."}
        now = int(time.time())
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT expires_at, consumed_at FROM vnc_sessions WHERE session_id = ? AND actor = ?",
                (session_id, actor[:64]),
            ).fetchone()
        if row is None:
            return {"state": "failed", "message": "The browser desktop session is no longer available."}
        if row["consumed_at"] is not None:
            return {"state": "connected", "message": "The protected desktop connection is active."}
        if int(row["expires_at"]) < now:
            return {"state": "failed", "message": "The browser desktop ticket expired before it connected."}
        return {"state": "connecting", "message": "Waiting for the protected desktop connection."}


class HostVncProbe:
    """Detect the loopback browser-desktop listener for this host."""

    def __init__(
        self,
        port: Optional[int] = None,
        remote_access_provider=None,
    ):
        configured = port if port is not None else env_value(
            "VAELOR_HOST_VNC_PORT", "PM_HOST_VNC_PORT", "5901"
        )
        try:
            self.port = int(configured)
        except (TypeError, ValueError):
            self.port = 5901
        if not 5900 <= self.port <= 65535:
            self.port = 5901
        self.remote_access_provider = (
            remote_access_provider
            or default_platform_drivers()["remote_access_provider"]
        )

    def status(self) -> dict:
        listening = self._compatible_listener()
        remote = self.remote_access_provider.capabilities()
        os_info = remote["os"]
        setup_supported = remote["browser_desktop_setup"]
        return {
            "available": listening,
            "setup_supported": setup_supported,
            "port": self.port,
            "name": f"{os_info.get('name') or 'Linux'} desktop",
            "kind": "host-vnc",
            "detail": (
                "The local browser desktop service is ready."
                if listening
                else (
                    "The appliance-managed browser desktop is ready for one-click setup."
                    if setup_supported
                    else "Browser desktop setup is not supported on this operating system."
                )
            ),
        }

    def _compatible_listener(self) -> bool:
        """Require the loopback RFB server mode protected by Vaelor tickets."""
        def receive_exact(connection, size: int) -> bytes:
            payload = b""
            while len(payload) < size:
                part = connection.recv(size - len(payload))
                if not part:
                    raise OSError("RFB server closed the health handshake.")
                payload += part
            return payload

        try:
            with socket.create_connection(
                ("127.0.0.1", self.port), timeout=0.6
            ) as connection:
                connection.settimeout(0.6)
                banner = receive_exact(connection, 12)
                if not banner.startswith(b"RFB 003."):
                    return False
                connection.sendall(banner)
                if banner.startswith(b"RFB 003.003"):
                    security = receive_exact(connection, 4)
                    if struct.unpack(">I", security)[0] != 1:
                        return False
                else:
                    count = receive_exact(connection, 1)[0]
                    offered = receive_exact(connection, count)
                    if 1 not in offered:
                        return False
                    connection.sendall(b"\x01")
                    if struct.unpack(">I", receive_exact(connection, 4))[0] != 0:
                        return False
                # Complete ClientInit/ServerInit so recurring health probes do
                # not accumulate as failed security handshakes in TigerVNC.
                connection.sendall(b"\x01")
                server_init = receive_exact(connection, 24)
                width, height = struct.unpack(">HH", server_init[:4])
                name_length = struct.unpack(">I", server_init[20:24])[0]
                if not width or not height or name_length > 65536:
                    return False
                receive_exact(connection, name_length)
                return True
        except (OSError, ValueError, struct.error):
            return False

    def target_port(self) -> int:
        status = self.status()
        if not status["available"]:
            raise ValueError(
                "The browser desktop is not commissioned. Use the one-click "
                "setup first."
            )
        return self.port


class HostRemoteDesktopProbe:
    """Report native GNOME RDP and optional loopback browser VNC separately."""

    def __init__(
        self,
        rdp_port: int = 3389,
        vnc_port: Optional[int] = None,
        remote_access_provider=None,
        certificate_source=None,
    ):
        self.rdp_port = int(rdp_port)
        self.remote_access_provider = (
            remote_access_provider
            or default_platform_drivers()["remote_access_provider"]
        )
        self.certificate_source = certificate_source or HostDesktopClient(
            timeout=CERTIFICATE_CLIENT_TIMEOUT
        )
        self.vnc = HostVncProbe(vnc_port, self.remote_access_provider)

    def _certificate(self) -> dict:
        """The fingerprint of the certificate the RDP listener is serving.

        Asked of the privileged broker, because both routes to the value need
        root, and asked only when there is a listener to have one — which is
        also the only moment an owner needs it. On an appliance with RDP off,
        the common case, this costs nothing.

        A failure is reported rather than flattened to an empty string:
        "there is no fingerprint" and "the broker did not answer" are
        different facts and the console says which (LESSONS pattern 8).
        `AttributeError` is in the list because `socket.AF_UNIX` does not
        exist off Linux, where the whole broker is absent.
        """
        try:
            return dict(NO_CERTIFICATE, **self.certificate_source.rdp_certificate())
        except (AttributeError, OSError, RuntimeError, ValueError) as error:
            return dict(NO_CERTIFICATE, detail=(
                "The certificate fingerprint could not be read from this "
                "appliance: {}".format(str(error)[:160])
            ))

    @staticmethod
    def _listening(port: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.4):
                return True
        except OSError:
            return False

    @staticmethod
    def _unit_active(unit: str) -> bool:
        try:
            result = subprocess.run(
                [
                    "/usr/bin/systemctl", "is-active", "--quiet",
                    unit,
                ],
                capture_output=True,
                check=False,
                timeout=2,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    @classmethod
    def _service_active(cls) -> bool:
        return cls._unit_active("gnome-remote-desktop.service")

    @staticmethod
    def _desktop_process_active() -> bool:
        try:
            result = subprocess.run(
                [
                    "/usr/bin/pgrep",
                    "-f",
                    "gnome-shell|gdm-wayland-session|gdm-x-session|Xorg|Xwayland",
                ],
                capture_output=True,
                check=False,
                timeout=2,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    @classmethod
    def _desktop_health(cls) -> dict:
        display_manager = cls._unit_active("display-manager.service")
        graphical_target = cls._unit_active("graphical.target")
        session = cls._desktop_process_active()
        available = display_manager and graphical_target and session
        return {
            "available": available,
            "display_manager": display_manager,
            "graphical_target": graphical_target,
            "session": session,
            "detail": (
                "The graphical login and desktop stack are healthy."
                if available
                else "No usable graphical login is running. Console access remains available."
            ),
        }

    def status(self) -> dict:
        remote = self.remote_access_provider.capabilities()
        os_info = remote["os"]
        os_name = os_info.get("name") or "Linux"
        rdp_setup_supported = remote["native_rdp_setup"]
        remote_name = remote["native_rdp_name"]
        desktop = self._desktop_health()
        rdp_ready = (
            desktop["available"]
            and self._service_active()
            and self._listening(self.rdp_port)
        )
        console_ready = self._listening(22)
        browser = self.vnc.status()
        return {
            "available": rdp_ready,
            "port": self.rdp_port,
            "name": remote_name,
            "kind": "host-rdp",
            "os": os_info,
            "preferred_access": "rdp" if desktop["available"] else "console",
            "desktop": desktop,
            "console": {
                "available": console_ready,
                "port": 22,
                "kind": "ssh",
                "detail": (
                    "Secure Shell is accepting console connections."
                    if console_ready
                    else "No SSH console listener was detected."
                ),
            },
            "detail": (
                f"Native {os_name} Remote Login is accepting secure RDP connections."
                if rdp_ready
                else (
                    f"{remote_name} is ready to configure."
                    if rdp_setup_supported
                    else f"Native RDP setup is not available on {os_name}; browser desktop availability is checked separately."
                )
            ),
            "rdp": {
                "available": rdp_ready,
                "port": self.rdp_port,
                "certificate": self._certificate() if rdp_ready else dict(
                    NO_CERTIFICATE
                ),
                "setup_supported": rdp_setup_supported and desktop["available"],
                "detail": (
                    "GNOME Remote Desktop is listening on the local network."
                    if rdp_ready
                    else (
                        "RDP is currently disabled."
                        if desktop["available"]
                        else "RDP is unavailable because the graphical desktop is not healthy."
                    )
                ),
            },
            "browser_vnc": browser,
        }


class VncTokenPlugin:
    """websockify token plugin. Source is the session SQLite path."""

    def __init__(self, source):
        self.source = source

    def lookup(self, token):
        # websockify serializes the plugin when it creates connection workers,
        # so keep only the database path here—not locks or live connections.
        return VncSessionStore(self.source).consume(token)
