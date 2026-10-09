"""Authentication, authorization, sessions, and audit storage for API v2."""

from __future__ import annotations

import hashlib
import hmac
import base64
import json
import os
import secrets
import sqlite3
import threading
import time
import struct
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional
from urllib.parse import quote

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .runtime_paths import env_value, state_path


ROLE_LEVELS = {
    "viewer": 10,
    "operator": 20,
    "administrator": 30,
}


class LastAdministratorError(RuntimeError):
    """Raised when an operation would remove the final enabled administrator."""


@dataclass(frozen=True)
class AuthSession:
    username: str
    role: str
    csrf_token: str
    expires_at: int


class SecurityStore:
    """Small SQLite-backed security store suitable for a single appliance."""

    def __init__(self, database_path: Optional[str] = None, session_hours: int = 12):
        self.database_path = database_path or env_value(
            "VAELOR_SECURITY_DB", "PM_DASHBOARD_SECURITY_DB",
            state_path("security.sqlite3"),
        )
        self.session_seconds = max(1, int(session_hours)) * 3600
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._session_touch_lock = threading.Lock()
        self._session_touched_at: Dict[str, int] = {}

    def _connect(self) -> sqlite3.Connection:
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                str(db_path),
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600,
            )
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        os.chmod(db_path, 0o600)
        connection = sqlite3.connect(str(db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")  # pairs-with: sqlite-foreign-keys-spaced
        for sqlite_sidecar in ("-wal", "-shm"):
            sidecar_path = Path("{}{}".format(db_path, sqlite_sidecar))
            if sidecar_path.exists():
                try:
                    os.chmod(sidecar_path, 0o600)
                except FileNotFoundError:
                    # SQLite may remove a transient sidecar between exists()
                    # and chmod() while parallel authenticated reads connect.
                    pass
                except PermissionError:
                    # Windows may lock an open SQLite sidecar against chmod;
                    # its ACL is inherited from the already-protected parent.
                    if os.name != "nt":
                        raise
        self._ensure_schema(connection)
        return connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            connection.execute("PRAGMA journal_mode = WAL")  # pairs-with: sqlite-journal-mode-spaced
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                    csrf_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    last_seen_at INTEGER NOT NULL,
                    remote_addr TEXT,
                    user_agent TEXT
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at INTEGER NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT,
                    result TEXT NOT NULL,
                    remote_addr TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_expires
                    ON sessions(expires_at);
                CREATE INDEX IF NOT EXISTS idx_audit_created
                    ON audit_events(created_at DESC);
                CREATE TABLE IF NOT EXISTS user_mfa (
                    username TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE,
                    nonce BLOB NOT NULL, ciphertext BLOB NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL
                );
                """
            )
            connection.commit()
            self._schema_ready = True

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _password_hash(password: str) -> str:
        salt = os.urandom(16)
        n, r, p = 16384, 8, 1
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=32,
        )
        return "$".join(
            (
                "scrypt",
                str(n),
                str(r),
                str(p),
                salt.hex(),
                digest.hex(),
            )
        )

    _UNKNOWN_ACCOUNT_HASH: Optional[str] = None

    @classmethod
    def _unknown_account_hash(cls) -> str:
        """A scrypt hash with the real parameters, made once, matching nothing."""
        if cls._UNKNOWN_ACCOUNT_HASH is None:
            cls._UNKNOWN_ACCOUNT_HASH = cls._password_hash(secrets.token_urlsafe(24))
        return cls._UNKNOWN_ACCOUNT_HASH

    @staticmethod
    def _password_matches(password: str, encoded: str) -> bool:
        try:
            algorithm, n, r, p, salt_hex, digest_hex = encoded.split("$", 5)
            if algorithm != "scrypt":
                return False
            actual = hashlib.scrypt(
                password.encode("utf-8"),
                salt=bytes.fromhex(salt_hex),
                n=int(n),
                r=int(r),
                p=int(p),
                dklen=len(bytes.fromhex(digest_hex)),
            )
            return hmac.compare_digest(actual.hex(), digest_hex)
        except (TypeError, ValueError):
            return False

    def has_users(self) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT 1 FROM users LIMIT 1").fetchone()
            return row is not None

    def bootstrap(self, username: str, password: str) -> bool:
        now = int(time.time())
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            if connection.execute("SELECT 1 FROM users LIMIT 1").fetchone():
                connection.rollback()
                return False
            connection.execute(
                """
                INSERT INTO users
                    (username, password_hash, role, enabled, created_at, updated_at)
                VALUES (?, ?, 'administrator', 1, ?, ?)
                """,
                (username, self._password_hash(password), now, now),
            )
            connection.commit()
        return True

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, str]]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT u.username, u.password_hash, u.role,
                       COALESCE(m.enabled, 0) AS mfa_enabled
                FROM users u
                LEFT JOIN user_mfa m ON m.username=u.username
                WHERE u.username = ? AND u.enabled = 1
                """,
                (username,),
            ).fetchone()
        if row is None:
            # R2-3 (pre-existing): an unknown account skipped scrypt, so the
            # answer's timing said whether the account exists. The same work
            # is done against a fixed hash and its answer thrown away.
            self._password_matches(password, self._unknown_account_hash())
            return None
        if not self._password_matches(password, row["password_hash"]):
            return None
        return {
            "username": row["username"], "role": row["role"],
            "mfa_enabled": bool(row["mfa_enabled"]),
        }

    def create_session(
        self,
        username: str,
        remote_addr: str = "",
        user_agent: str = "",
    ) -> Dict[str, Any]:
        token = secrets.token_urlsafe(32)
        csrf_token = self.csrf_for(token)
        now = int(time.time())
        expires_at = now + self.session_seconds
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            connection.execute(
                """
                INSERT INTO sessions
                    (token_hash, username, csrf_hash, created_at, expires_at,
                     last_seen_at, remote_addr, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._hash_token(token),
                    username,
                    self._hash_token(csrf_token),
                    now,
                    expires_at,
                    now,
                    remote_addr[:128],
                    user_agent[:512],
                ),
            )
            connection.commit()
        return {
            "token": token,
            "csrf_token": csrf_token,
            "expires_at": expires_at,
        }

    def get_session(self, token: str) -> Optional[AuthSession]:
        if not token:
            return None
        now = int(time.time())
        token_hash = self._hash_token(token)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT s.username, u.role, s.csrf_hash, s.expires_at,
                       s.last_seen_at
                FROM sessions s
                JOIN users u ON u.username = s.username
                WHERE s.token_hash = ? AND s.expires_at > ? AND u.enabled = 1
                """,
                (token_hash, now),
            ).fetchone()
            if row is None:
                return None
            # Authenticated pages load several endpoints in parallel. Writing
            # this same session row for every read creates avoidable SQLite
            # contention, so activity is coalesced into one update per minute.
            if int(row["last_seen_at"]) <= now - 60:
                with self._session_touch_lock:
                    last_touch = self._session_touched_at.get(
                        token_hash, int(row["last_seen_at"])
                    )
                    if last_touch <= now - 60:
                        connection.execute(
                            """
                            UPDATE sessions SET last_seen_at = ?
                            WHERE token_hash = ? AND last_seen_at <= ?
                            """,
                            (now, token_hash, now - 60),
                        )
                        connection.commit()
                        self._session_touched_at[token_hash] = now
        return AuthSession(
            username=row["username"],
            role=row["role"],
            csrf_token=row["csrf_hash"],
            expires_at=row["expires_at"],
        )

    @staticmethod
    def csrf_for(token: str) -> str:
        return hmac.new(
            token.encode("utf-8"),
            b"pironman-dashboard-csrf-v2",
            hashlib.sha256,
        ).hexdigest()

    def csrf_matches(
        self,
        session: AuthSession,
        csrf_token: str,
        token: str = "",
    ) -> bool:
        if not csrf_token:
            return False
        if token:
            return hmac.compare_digest(self.csrf_for(token), csrf_token)
        return hmac.compare_digest(
            session.csrf_token,
            self._hash_token(csrf_token),
        )

    def rotate_csrf(self, token: str) -> str:
        csrf_token = self.csrf_for(token)
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE sessions SET csrf_hash = ? WHERE token_hash = ?",
                (self._hash_token(csrf_token), self._hash_token(token)),
            )
            connection.commit()
        return csrf_token

    def revoke_session(self, token: str) -> None:
        if not token:
            return
        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?",
                (self._hash_token(token),),
            )
            connection.commit()

    def list_users(self) -> list[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT u.username, u.role, u.enabled, u.created_at, u.updated_at,
                       COALESCE(m.enabled, 0) AS mfa_enabled,
                       COUNT(s.token_hash) AS active_sessions
                FROM users u
                LEFT JOIN user_mfa m ON m.username=u.username
                LEFT JOIN sessions s ON s.username = u.username AND s.expires_at > ?
                GROUP BY u.username
                ORDER BY u.username
                """,
                (int(time.time()),),
            ).fetchall()
        return [
            {
                "username": row["username"],
                "role": row["role"],
                "enabled": bool(row["enabled"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "active_sessions": row["active_sessions"],
                "mfa_enabled": bool(row["mfa_enabled"]),
            }
            for row in rows
        ]

    def create_user(self, username: str, password: str, role: str) -> Dict[str, Any]:
        if role not in ROLE_LEVELS:
            raise ValueError("Choose viewer, operator, or administrator.")
        now = int(time.time())
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO users
                        (username, password_hash, role, enabled, created_at, updated_at)
                    VALUES (?, ?, ?, 1, ?, ?)
                    """,
                    (username, self._password_hash(password), role, now, now),
                )
                connection.commit()
        except sqlite3.IntegrityError as error:
            raise ValueError("That username already exists.") from error
        return next(item for item in self.list_users() if item["username"] == username)

    def update_user(
        self, username: str, *, role: Optional[str] = None,
        enabled: Optional[bool] = None, password: Optional[str] = None,
    ) -> Dict[str, Any]:
        if role is not None and role not in ROLE_LEVELS:
            raise ValueError("Choose viewer, operator, or administrator.")
        assignments = []
        values: list[Any] = []
        if role is not None:
            assignments.append("role = ?")
            values.append(role)
        if enabled is not None:
            assignments.append("enabled = ?")
            values.append(int(enabled))
        if password is not None:
            assignments.append("password_hash = ?")
            values.append(self._password_hash(password))
        if not assignments:
            raise ValueError("Choose an account setting to change.")
        now = int(time.time())
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            existing = connection.execute(
                "SELECT username, role, enabled FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if existing is None:
                connection.rollback()
                raise KeyError(username)
            next_role = role if role is not None else existing["role"]
            next_enabled = int(enabled) if enabled is not None else int(existing["enabled"])
            if (
                existing["role"] == "administrator"
                and existing["enabled"]
                and (next_role != "administrator" or not next_enabled)
            ):
                remaining = connection.execute(
                    """
                    SELECT COUNT(*) FROM users
                    WHERE role = 'administrator' AND enabled = 1
                      AND username != ?
                    """,
                    (username,),
                ).fetchone()[0]
                if int(remaining) < 1:
                    connection.rollback()
                    raise LastAdministratorError(
                        "Keep at least one enabled administrator account."
                    )
            assignments.append("updated_at = ?")
            values.append(now)
            values.append(username)
            cursor = connection.execute(
                "UPDATE users SET {} WHERE username = ?".format(", ".join(assignments)),
                values,
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise KeyError(username)
            if enabled is False or password is not None:
                connection.execute(
                    "DELETE FROM sessions WHERE username = ?",
                    (username,),
                )
            connection.commit()
        return next(item for item in self.list_users() if item["username"] == username)

    def delete_user(self, username: str) -> bool:
        """Permanently remove one account and its MFA/session records."""
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            existing = connection.execute(
                "SELECT role, enabled FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if existing is None:
                connection.rollback()
                return False
            if existing["role"] == "administrator" and existing["enabled"]:
                remaining = connection.execute(
                    """
                    SELECT COUNT(*) FROM users
                    WHERE role = 'administrator' AND enabled = 1
                      AND username != ?
                    """,
                    (username,),
                ).fetchone()[0]
                if int(remaining) < 1:
                    connection.rollback()
                    raise LastAdministratorError(
                        "Keep at least one enabled administrator account."
                    )
            cursor = connection.execute(
                "DELETE FROM users WHERE username = ?",
                (username,),
            )
            connection.commit()
        return cursor.rowcount == 1

    def administrator_count(self, *, enabled_only: bool = True) -> int:
        query = "SELECT COUNT(*) FROM users WHERE role = 'administrator'"
        if enabled_only:
            query += " AND enabled = 1"
        with closing(self._connect()) as connection:
            return int(connection.execute(query).fetchone()[0])

    def list_sessions(self) -> list[Dict[str, Any]]:
        now = int(time.time())
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            rows = connection.execute(
                """
                SELECT token_hash, username, created_at, expires_at, last_seen_at,
                       remote_addr, user_agent
                FROM sessions
                ORDER BY last_seen_at DESC
                LIMIT 200
                """
            ).fetchall()
            connection.commit()
        return [
            {
                "id": row["token_hash"],
                "username": row["username"],
                "created_at": row["created_at"],
                "expires_at": row["expires_at"],
                "last_seen_at": row["last_seen_at"],
                "remote_addr": row["remote_addr"],
                "user_agent": row["user_agent"],
            }
            for row in rows
        ]

    def revoke_session_id(self, session_id: str) -> bool:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (session_id,)
            )
            connection.commit()
            return cursor.rowcount == 1

    def _mfa_key(self):
        path = Path(self.database_path).with_name("totp.key")
        try:
            key = path.read_bytes()
        except FileNotFoundError:
            key = os.urandom(32)
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(key)
        if len(key) != 32:
            raise ValueError("The local two-factor key is invalid.")
        return key

    @staticmethod
    def _totp(secret: bytes, timestamp: Optional[int] = None):
        counter = int((timestamp if timestamp is not None else time.time()) // 30)
        digest = hmac.new(secret, struct.pack(">Q", counter), hashlib.sha1).digest()
        offset = digest[-1] & 0x0F
        value = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
        return "{:06d}".format(value % 1_000_000)

    def begin_totp(self, username: str):
        secret = os.urandom(20)
        encoded = base64.b32encode(secret).decode("ascii").rstrip("=")
        nonce = os.urandom(12)
        ciphertext = AESGCM(self._mfa_key()).encrypt(
            nonce, secret, username.encode("utf-8")
        )
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO user_mfa(username,nonce,ciphertext,enabled,updated_at)
                VALUES(?,?,?,0,?)
                ON CONFLICT(username) DO UPDATE SET
                    nonce=excluded.nonce,ciphertext=excluded.ciphertext,
                    enabled=0,updated_at=excluded.updated_at
                """,
                (username, nonce, ciphertext, int(time.time())),
            )
            connection.commit()
        label = quote("Vaelor:{}".format(username))
        return {
            "secret": encoded,
            "uri": "otpauth://totp/{}?secret={}&issuer=Vaelor&digits=6&period=30".format(label, encoded),
        }

    def _totp_secret(self, username: str):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT nonce,ciphertext,enabled FROM user_mfa WHERE username=?",
                (username,),
            ).fetchone()
        if row is None:
            return None, False
        try:
            secret = AESGCM(self._mfa_key()).decrypt(
                row["nonce"], row["ciphertext"], username.encode("utf-8")
            )
        except Exception as error:
            raise ValueError("Two-factor configuration could not be decrypted.") from error
        return secret, bool(row["enabled"])

    def verify_totp(self, username: str, code: str, require_enabled: bool = True):
        secret, enabled = self._totp_secret(username)
        if secret is None or (require_enabled and not enabled):
            return False
        code = str(code).strip()
        now = int(time.time())
        return len(code) == 6 and code.isdigit() and any(
            hmac.compare_digest(code, self._totp(secret, now + offset))
            for offset in (-30, 0, 30)
        )

    def confirm_totp(self, username: str, code: str):
        if not self.verify_totp(username, code, require_enabled=False):
            raise ValueError("The verification code is incorrect.")
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE user_mfa SET enabled=1,updated_at=? WHERE username=?",
                (int(time.time()), username),
            )
            connection.commit()
        return {"enabled": True}

    def disable_totp(self, username: str, code: str):
        if not self.verify_totp(username, code):
            raise ValueError("The verification code is incorrect.")
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM user_mfa WHERE username=?", (username,))
            connection.commit()
        return {"enabled": False}

    def role_allows(self, role: str, allowed_roles: Iterable[str]) -> bool:
        minimum = min(ROLE_LEVELS[item] for item in allowed_roles)
        return ROLE_LEVELS.get(role, 0) >= minimum

    def audit(
        self,
        actor: str,
        action: str,
        result: str,
        target: str = "",
        remote_addr: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        safe_details = json.dumps(details or {}, separators=(",", ":"), default=str)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO audit_events
                    (created_at, actor, action, target, result, remote_addr, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(time.time()),
                    # #205 coverage: cap actor like target/remote_addr. An
                    # unauthenticated login audits `username or "unknown"`; left
                    # uncapped, a 5000-char username writes a 5000-char row and
                    # each distinct long username gets its own rate-limit budget
                    # (keyed on ip:username) - an audit-table growth vector.
                    str(actor)[:256],
                    action,
                    target[:256],
                    result,
                    remote_addr[:128],
                    safe_details,
                ),
            )
            connection.commit()

    def audit_between(
        self, since: float, until: float, action: str, target: str = "", limit: int = 500,
    ) -> list[Dict[str, Any]]:
        """One action's audit rows in a time span, oldest first.

        For the Assistant's machine history (VD-205 item 3): the newest rows
        across the whole appliance say nothing about a window last night once
        a busy morning has pushed them out (adversarial review B-2).
        """
        query = ("SELECT created_at, actor, action, target, result FROM audit_events "
                 "WHERE action = ? AND created_at >= ? AND created_at <= ?")
        values: list = [action, int(since), int(until)]
        if target:
            query += " AND target = ?"
            values.append(target[:256])
        query += " ORDER BY created_at ASC LIMIT ?"
        values.append(max(1, min(int(limit), 2000)))
        with closing(self._connect()) as connection:
            rows = connection.execute(query, values).fetchall()
        return [{key: row[key] for key in ("created_at", "actor", "action", "target", "result")}
                for row in rows]

    def list_audit(self, limit: int = 50) -> list[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, actor, action, target, result,
                       remote_addr, details_json
                FROM audit_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "actor": row["actor"],
                "action": row["action"],
                "target": row["target"],
                "result": row["result"],
                "remote_addr": row["remote_addr"],
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        ]


class LoginLimiter:
    """Failed sign-ins, counted two ways, in memory for one dashboard process.

    **F11 (owner decision 2026-10-03, "Both limits").** W4d-D6 gave the
    limiter each browser's real address, so keyed on (address, username) it
    gave every address its own budget and N addresses had N times the guesses
    at one account (LESSONS 8: a limit is a limit over the population it
    counts). Now:

    - **per (address, username):** ``ADDRESS_ATTEMPTS`` failures in
      ``ADDRESS_WINDOW_SECONDS`` - 5 in 5 minutes - slows one noisy machine;
    - **per username, all addresses:** ``USERNAME_ATTEMPTS`` failures in
      ``USERNAME_WINDOW_SECONDS`` - 30 in 5 minutes - bounds what many
      addresses can try together, at most 8,640 guesses a day per account.

    **A stranger may slow the owner, never block them for long.** The account
    cap is a sliding window, so it lifts at most 5 minutes after the last
    counted failure, and refused attempts are not counted. An attack that keeps
    the cap full could still hold it, so an address that signed in as the
    account within ``KNOWN_ADDRESS_SECONDS`` (30 days) is not subject to the
    account cap - it keeps its own per-address limit - and the owner's usual
    machine is never locked out by strangers' failures.
    """

    ADDRESS_ATTEMPTS = 5
    ADDRESS_WINDOW_SECONDS = 300
    USERNAME_ATTEMPTS = 30
    USERNAME_WINDOW_SECONDS = 300
    KNOWN_ADDRESS_SECONDS = 30 * 24 * 3600
    #: A bound on the known-address memory per account, oldest dropped first.
    KNOWN_ADDRESSES_KEPT = 16

    ADDRESS_REFUSAL = "Too many attempts. Try again in a few minutes."
    USERNAME_REFUSAL = (
        "Too many failed sign-ins for this account from different addresses. "
        "Try again in a few minutes, or sign in from a machine that has signed "
        "in as this account before."
    )

    #: R2-3: the most keys any one table holds. A key used to be pruned only
    #: when the same key was asked about again, so 100,000 distinct usernames
    #: tried once each stayed in memory for ever - reachable without an
    #: account. Past this bound the least recently counted key is dropped
    #: first, after every expired key has been swept. The cost is stated, not
    #: hidden: a flood of more distinct names than this inside one window can
    #: push a real account's count out early; bounded memory is the trade.
    MAX_TRACKED_KEYS = 20_000
    #: Seconds between sweeps of keys whose window has passed.
    SWEEP_SECONDS = 60
    #: Usernames longer than this are keyed by their SHA-256, so a
    #: multi-megabyte name is not stored, twice, as a dictionary key.
    KEYED_USERNAME_CHARS = 64

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._by_address: "OrderedDict[tuple, list[float]]" = OrderedDict()
        self._by_username: "OrderedDict[str, list[float]]" = OrderedDict()
        self._known: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
        self._lock = threading.Lock()
        self._swept = clock()

    @classmethod
    def _key(cls, username: str) -> str:
        if len(username) <= cls.KEYED_USERNAME_CHARS:
            return username
        return "sha256:" + hashlib.sha256(username.encode("utf-8", "replace")).hexdigest()

    @staticmethod
    def _active(table: Dict[Any, list[float]], key: Any, cutoff: float) -> list[float]:
        active = [item for item in table.get(key, []) if item > cutoff]
        if active:
            table[key] = active
        else:
            table.pop(key, None)
        return active

    def _sweep(self, now: float) -> None:
        """Drop every expired key, at most once per ``SWEEP_SECONDS``."""
        if now - self._swept < self.SWEEP_SECONDS:
            return
        self._swept = now
        for table, window in ((self._by_address, self.ADDRESS_WINDOW_SECONDS),
                              (self._by_username, self.USERNAME_WINDOW_SECONDS)):
            for key in [key for key, stamps in table.items() if not stamps or stamps[-1] <= now - window]:
                del table[key]
        for username in list(self._known):
            addresses = self._known[username]
            for address in [a for a, seen in addresses.items() if now - seen > self.KNOWN_ADDRESS_SECONDS]:
                del addresses[address]
            if not addresses:
                del self._known[username]

    def _count(self, table: "OrderedDict[Any, list[float]]", key: Any, now: float, keep: int) -> None:
        stamps = table.pop(key, [])
        stamps.append(now)
        table[key] = stamps[-keep:]  # counting past the limit changes nothing
        while len(table) > self.MAX_TRACKED_KEYS:
            table.popitem(last=False)

    def _is_known(self, address: str, username: str, now: float) -> bool:
        seen = self._known.get(username, {}).get(address)
        return seen is not None and now - seen <= self.KNOWN_ADDRESS_SECONDS

    def refusal(self, address: str, username: str) -> Optional[tuple]:
        """``None`` when a sign-in may be tried, else ``(code, message)``."""
        now = self._clock()
        username = self._key(username)
        with self._lock:
            self._sweep(now)
            mine = self._active(
                self._by_address, (address, username), now - self.ADDRESS_WINDOW_SECONDS)
            if len(mine) >= self.ADDRESS_ATTEMPTS:
                return ("login_rate_limited", self.ADDRESS_REFUSAL)
            everyone = self._active(
                self._by_username, username, now - self.USERNAME_WINDOW_SECONDS)
            if (len(everyone) >= self.USERNAME_ATTEMPTS
                    and not self._is_known(address, username, now)):
                return ("login_rate_limited", self.USERNAME_REFUSAL)
            return None

    def failed(self, address: str, username: str) -> None:
        now = self._clock()
        username = self._key(username)
        with self._lock:
            self._sweep(now)
            self._count(self._by_address, (address, username), now, self.ADDRESS_ATTEMPTS)
            self._count(self._by_username, username, now, self.USERNAME_ATTEMPTS)

    def succeeded(self, address: str, username: str) -> None:
        """A sign-in worked: clear this address's own count, and remember it.

        R2-4: the account-wide count is NOT cleared. A success from a known
        address resetting it would let an attacker who also controls one
        known address (or simply waits for the owner to sign in) start a fresh
        30-guess budget; the cap still lifts on its own after its window.
        """
        now = self._clock()
        username = self._key(username)
        with self._lock:
            self._by_address.pop((address, username), None)
            known = self._known.pop(username, {})
            known[address] = now
            for stale in sorted(known, key=known.get)[:-self.KNOWN_ADDRESSES_KEPT]:
                del known[stale]
            self._known[username] = known
            while len(self._known) > self.MAX_TRACKED_KEYS:
                self._known.popitem(last=False)
