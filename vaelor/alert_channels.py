"""Storage for alert-delivery channel configuration.

A channel is where a fired alert is sent: an email over SMTP, or a JSON POST to
a webhook. This store holds only *configuration* - host, port, addresses, the
webhook URL, and a reference to the channel's secret. The secret itself (an SMTP
password or a webhook auth token) never lands here: it lives in the encrypted
credential broker under a per-channel purpose, and only that ``secret_purpose``
string is stored. The store also records the outcome of the last delivery so the
management UI can show "delivered" or "failed: reason" beside each channel.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import uuid
from contextlib import closing
from typing import Optional

from .alert_delivery import EMAIL_SECURITIES, deliver_alert, summarize_results
from .credential_broker import ALERT_PURPOSE_PREFIX
from .runtime_paths import env_value, state_path

CHANNEL_KINDS = ("email", "webhook")
CHANNEL_NOT_FOUND = "Delivery channel not found."
_EMAIL_SHAPE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AlertChannelError(ValueError):
    """A rejected channel configuration, safe to return to the client."""


def _is_loopback_host(host: str) -> bool:
    """Whether a host is unambiguously this machine, by string (no DNS)."""
    lowered = host.strip().lower().strip("[]")
    return (
        lowered in ("localhost", "127.0.0.1", "::1")
        or lowered.startswith("127.")
    )


def channel_purpose(channel_id: str) -> str:
    """The broker purpose that holds one channel's secret."""
    return "{}{}".format(ALERT_PURPOSE_PREFIX, channel_id)


class AlertChannelStore:
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_ALERT_CHANNELS_DB", "PM_ALERT_CHANNELS_DB",
            state_path("assistant/alert_channels.sqlite3"),
        )
        parent = os.path.dirname(self.database_path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        self._initialize()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS alert_channels (
                    id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    smtp_host TEXT NOT NULL DEFAULT '',
                    smtp_port INTEGER NOT NULL DEFAULT 0,
                    security TEXT NOT NULL DEFAULT 'starttls',
                    from_address TEXT NOT NULL DEFAULT '',
                    to_address TEXT NOT NULL DEFAULT '',
                    username TEXT NOT NULL DEFAULT '',
                    url TEXT NOT NULL DEFAULT '',
                    auth_header TEXT NOT NULL DEFAULT '',
                    secret_purpose TEXT NOT NULL DEFAULT '',
                    has_secret INTEGER NOT NULL DEFAULT 0,
                    last_delivery_status TEXT NOT NULL DEFAULT '',
                    last_delivery_error TEXT NOT NULL DEFAULT '',
                    last_delivery_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            connection.commit()

    def _row(self, row) -> dict:
        item = dict(row)
        item["enabled"] = bool(item["enabled"])
        item["has_secret"] = bool(item["has_secret"])
        return item

    def _validate(self, kind: str, body: dict) -> dict:
        if kind not in CHANNEL_KINDS:
            raise AlertChannelError("Choose an email or webhook channel.")
        name = str(body.get("name", "")).strip()[:100]
        if not name:
            raise AlertChannelError("Give the delivery channel a name.")
        clean = {
            "kind": kind, "name": name, "smtp_host": "", "smtp_port": 0,
            "security": "starttls", "from_address": "", "to_address": "",
            "username": "", "url": "", "auth_header": "",
        }
        if kind == "email":
            self._validate_email(body, clean)
        else:
            self._validate_webhook(body, clean)
        return clean

    def _validate_email(self, body: dict, clean: dict) -> None:
        host = str(body.get("smtp_host", "")).strip()[:255]
        if not host:
            raise AlertChannelError("Enter the outgoing mail server host.")
        security = str(body.get("security", "starttls")).strip().lower()
        if security not in EMAIL_SECURITIES:
            raise AlertChannelError("Choose STARTTLS, SSL, or no transport security.")
        try:
            port = int(body.get("smtp_port") or (465 if security == "ssl" else 587))
        except (TypeError, ValueError) as error:
            raise AlertChannelError("The mail server port must be a number.") from error
        if not 1 <= port <= 65535:
            raise AlertChannelError("The mail server port is out of range.")
        sender = str(body.get("from_address", "")).strip()[:255]
        recipient = str(body.get("to_address", "")).strip()[:255]
        if not _EMAIL_SHAPE.match(sender) or not _EMAIL_SHAPE.match(recipient):
            raise AlertChannelError("Enter a valid sender and recipient email address.")
        username = str(body.get("username", "")).strip()[:255]
        if security == "none" and username and not _is_loopback_host(host):
            # A username means a password will be sent; with no transport
            # security that password goes on the wire in the clear. Only permit
            # it for a loopback relay, which the docstring already scopes it to.
            raise AlertChannelError(
                "Sending a username without transport security would put the "
                "SMTP password on the wire in the clear. Use STARTTLS or SSL, "
                "or a loopback relay."
            )
        clean.update({
            "smtp_host": host, "smtp_port": port, "security": security,
            "from_address": sender, "to_address": recipient,
            "username": username,
        })

    def _validate_webhook(self, body: dict, clean: dict) -> None:
        url = str(body.get("url", "")).strip()[:2000]
        if not url.startswith(("http://", "https://")):
            raise AlertChannelError("Enter an http or https webhook URL.")
        clean.update({
            "url": url,
            "auth_header": str(body.get("auth_header", "")).strip()[:100],
        })

    def create(self, actor: str, kind: str, body: dict) -> dict:
        clean = self._validate(kind, body)
        now = time.time()
        channel_id = uuid.uuid4().hex
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO alert_channels
                (id,actor,kind,name,enabled,smtp_host,smtp_port,security,
                 from_address,to_address,username,url,auth_header,
                 secret_purpose,has_secret,created_at,updated_at)
                VALUES(?,?,?,?,1,?,?,?,?,?,?,?,?,?,0,?,?)
                """,
                (channel_id, actor, clean["kind"], clean["name"],
                 clean["smtp_host"], clean["smtp_port"], clean["security"],
                 clean["from_address"], clean["to_address"], clean["username"],
                 clean["url"], clean["auth_header"], channel_purpose(channel_id),
                 now, now),
            )
            connection.commit()
        return self.get(channel_id, actor)

    def update(self, channel_id: str, actor: str, body: dict) -> dict:
        existing = self.get(channel_id, actor)
        if existing is None:
            raise AlertChannelError(CHANNEL_NOT_FOUND)
        clean = self._validate(existing["kind"], body)
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE alert_channels SET name=?,smtp_host=?,smtp_port=?,security=?,
                    from_address=?,to_address=?,username=?,url=?,auth_header=?,updated_at=?
                WHERE id=? AND actor=?
                """,
                (clean["name"], clean["smtp_host"], clean["smtp_port"],
                 clean["security"], clean["from_address"], clean["to_address"],
                 clean["username"], clean["url"], clean["auth_header"], now,
                 channel_id, actor),
            )
            connection.commit()
        return self.get(channel_id, actor)

    def set_enabled(self, channel_id: str, actor: str, enabled: bool) -> dict:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE alert_channels SET enabled=?,updated_at=? WHERE id=? AND actor=?",
                (int(enabled), time.time(), channel_id, actor),
            )
            connection.commit()
        if not cursor.rowcount:
            raise AlertChannelError(CHANNEL_NOT_FOUND)
        return self.get(channel_id, actor)

    def mark_secret(self, channel_id: str, actor: str, present: bool) -> dict:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE alert_channels SET has_secret=?,updated_at=? WHERE id=? AND actor=?",
                (int(present), time.time(), channel_id, actor),
            )
            connection.commit()
        if not cursor.rowcount:
            raise AlertChannelError(CHANNEL_NOT_FOUND)
        return self.get(channel_id, actor)

    def get(self, channel_id: str, actor: Optional[str] = None) -> Optional[dict]:
        query = "SELECT * FROM alert_channels WHERE id=?"
        values = [channel_id]
        if actor is not None:
            query += " AND actor=?"
            values.append(actor)
        with closing(self._connect()) as connection:
            row = connection.execute(query, values).fetchone()
        return self._row(row) if row else None

    def list(self, actor: Optional[str] = None) -> list:
        query = "SELECT * FROM alert_channels"
        values = []
        if actor is not None:
            query += " WHERE actor=?"
            values.append(actor)
        query += " ORDER BY created_at DESC"
        with closing(self._connect()) as connection:
            return [self._row(row) for row in connection.execute(query, values)]

    def delete(self, channel_id: str, actor: str) -> dict:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM alert_channels WHERE id=? AND actor=?",
                (channel_id, actor),
            )
            connection.commit()
        if not cursor.rowcount:
            raise AlertChannelError(CHANNEL_NOT_FOUND)
        return {"deleted": True, "id": channel_id}

    def enabled_channels(self) -> list:
        """Every enabled channel, shaped for :func:`alert_delivery.deliver_alert`."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM alert_channels WHERE enabled=1 ORDER BY created_at"
            )
            return [self._row(row) for row in rows]

    def record_delivery(self, channel_id: str, status: str, error: str = "") -> None:
        """Persist one channel's last delivery outcome for the UI to show."""
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE alert_channels
                SET last_delivery_status=?,last_delivery_error=?,last_delivery_at=?
                WHERE id=?
                """,
                (str(status)[:40], str(error)[:480], time.time(), channel_id),
            )
            connection.commit()


def record_results(store: "AlertChannelStore", results) -> str:
    """Persist per-channel delivery outcomes and return a compact summary."""
    for result in results:
        store.record_delivery(
            result.channel_id,
            "delivered" if result.delivered else "failed",
            result.error,
        )
    return summarize_results(results)


def build_delivery_callback(store: "AlertChannelStore", resolve_secret, senders=None):
    """Build the ``deliver(alert)`` callback the trigger evaluator invokes.

    It fans a fired alert out to every enabled channel, records each channel's
    own outcome, and returns a one-line summary for the trigger row. The bound
    ``resolve_secret`` maps a channel's broker purpose to its secret.
    """
    def deliver(alert: dict) -> str:
        results = deliver_alert(store.enabled_channels(), alert, resolve_secret, senders)
        return record_results(store, results)

    return deliver
