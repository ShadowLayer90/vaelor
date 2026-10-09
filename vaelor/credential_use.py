"""When a stored credential was last USED: one column, one verb, one recorder.

The question "when did Vaelor last use this credential?" has one owner: the
vault's ``last_used_at`` column, written only by :func:`record_use` (a
connection Test records its own ``last_tested_at`` and is not a use), which every credential listing already carries
(the Encrypted AI connections rows, the LLM Server key table). Before this
module nothing wrote it except a connection Test (ACC-107, ACC-043), so a
listing read "never used" for a key an external app had used all day, and for
an SSH credential Vaelor presented every ten seconds.

**Two halves, two processes.**

* :func:`record_use` runs INSIDE the credential broker (``vaelor-secrets``), as
  the ``record_use`` socket verb. It moves ``last_used_at`` forward only - an
  older timestamp never replaces a newer one - so a late or repeated report
  cannot make a key look less recently used than it was. It decrypts nothing
  and writes no audit row: a use is not a security event, and one audit row per
  use would bury the ones that are.
* :class:`CredentialUseRecorder` runs in the CALLERS (the control plane and the
  workload executor). A caller reports a use at the point it presented the
  secret and was answered - an AI Chat reply, an SSH login, a Hugging Face
  download, an application deploy - through :func:`note_credential_use`. A
  resolve is NOT a use: several loops resolve a lease every few seconds only to
  read its address, and recording there would make every credential read "used
  just now" (LESSONS pattern 5).

The recorder is best-effort by contract: a broker that is down, slow or older
than this verb costs the caller nothing, because reporting a use must never
fail or delay the use itself. It is throttled per credential, since an SSH
credential is presented on every telemetry scrape of a worker.

Inbound LLM Server keys reach the same verb from :mod:`vaelor.llm_gate_usage`,
which reads the gate's access log, so the key table reads one column for every
kind of key.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import closing
from typing import Any, Callable, Dict, Mapping, Optional, Union

LOGGER = logging.getLogger(__name__)

#: How often one credential's use is reported, at most. A "last used" reading is
#: read in minutes, and an SSH credential is presented on every ten-second scrape
#: of a worker; one broker round trip a minute per credential is enough.
USE_REPORT_INTERVAL_SECONDS = 60.0

#: How far ahead of the broker's clock a reported use may be stamped and still be
#: believed. Every caller is on this machine and shares its clock, so a few
#: seconds of scheduling is all; more is a wrong unit or a wrong clock, and it is
#: clamped rather than stored.
FUTURE_TOLERANCE_SECONDS = 5

#: How long the recorder waits on the broker socket. A use report is never worth
#: holding anything for, and it runs off the caller's thread anyway.
REPORT_TIMEOUT_SECONDS = 5


def record_use(vault: Any, credential_id: Any, used_at: Any = None) -> Dict[str, Any]:
    """Move one credential's ``last_used_at`` forward to ``used_at`` (broker side).

    ``used_at`` defaults to now and is clamped to at most
    :data:`FUTURE_TOLERANCE_SECONDS` ahead of it. A revoked key is not moved: its
    last use is history, and a revoke must not be followed by a fresh "used"
    reading from a report that was already in flight. Answers
    ``{"recorded": bool}`` - False for an unknown, revoked or blank id, and for a
    report no newer than the use already stored.
    """
    clean_id = str(credential_id or "").strip()
    now = int(time.time())
    try:
        stamp = int(float(used_at)) if used_at is not None else now
    except (TypeError, ValueError):
        stamp = now
    stamp = max(0, min(stamp, now + FUTURE_TOLERANCE_SECONDS))
    if not clean_id:
        return {"recorded": False}
    with closing(vault._connect()) as connection:
        cursor = connection.execute(
            """
            UPDATE credentials SET last_used_at = ?
            WHERE id = ? AND revoked_at IS NULL
              AND (last_used_at IS NULL OR last_used_at < ?)
            """,
            (stamp, clean_id, stamp),
        )
        connection.commit()
    return {"recorded": cursor.rowcount == 1}


def _credential_id(subject: Union[str, Mapping[str, Any], None]) -> str:
    """The credential id a lease or a bare id names, or ``""``."""
    if isinstance(subject, Mapping):
        return str(subject.get("credential_id") or "").strip()
    return str(subject or "").strip()


def _default_client() -> Any:
    from .credential_broker_client import CredentialBrokerClient

    return CredentialBrokerClient(timeout_seconds=REPORT_TIMEOUT_SECONDS)


class CredentialUseRecorder:
    """Report credential uses to the broker, throttled and off the caller's path.

    ``client_factory`` builds a broker client with a ``record_use`` method; the
    default is the shipping :class:`~vaelor.credential_broker_client.CredentialBrokerClient`
    with a short timeout. ``background`` sends from a daemon thread so a slow
    broker never delays an answer; a test passes False to send inline.
    """

    def __init__(
        self,
        *,
        client_factory: Optional[Callable[[], Any]] = None,
        clock: Callable[[], float] = time.time,
        interval_seconds: float = USE_REPORT_INTERVAL_SECONDS,
        background: bool = True,
    ):
        self._client_factory = client_factory or _default_client
        self._clock = clock
        self._interval = float(interval_seconds)
        self._background = background
        self._lock = threading.Lock()
        self._reported: Dict[str, float] = {}

    def note(
        self, subject: Union[str, Mapping[str, Any], None], *,
        broker: Any = None, used_at: Optional[float] = None,
    ) -> bool:
        """Report that the credential ``subject`` names was just used.

        ``subject`` is a broker lease (it carries ``credential_id``) or a bare id.
        ``broker`` is the caller's own broker client when it holds one; one
        without ``record_use`` (a double written for an older contract) falls
        back to the shipping client rather than dropping the report. True when a
        report was sent (or queued), False when it was throttled or there was
        nothing to report. Never raises.
        """
        credential_id = _credential_id(subject)
        if not credential_id:
            return False
        now = self._clock()
        with self._lock:
            last = self._reported.get(credential_id)
            if last is not None and now - last < self._interval:
                return False
            self._reported[credential_id] = now
        stamp = now if used_at is None else used_at
        client = broker if callable(getattr(broker, "record_use", None)) else None

        def send() -> None:
            try:
                target = client if client is not None else self._client_factory()
                target.record_use(credential_id, stamp)
            except Exception as error:  # noqa: BLE001 - a use report is best-effort
                LOGGER.debug("A credential use could not be recorded: %s", error)
                with self._lock:
                    # The next use tries again rather than waiting out the
                    # throttle for a report that never landed.
                    if self._reported.get(credential_id) == now:
                        self._reported.pop(credential_id, None)

        if self._background:
            threading.Thread(target=send, name="credential-use", daemon=True).start()
        else:
            send()
        return True


#: The process-wide recorder every use site reports through, so the throttle is
#: shared by every path that presents the same credential in this process.
RECORDER = CredentialUseRecorder()


def note_credential_use(
    subject: Union[str, Mapping[str, Any], None], *, broker: Any = None,
    used_at: Optional[float] = None,
) -> bool:
    """Report one use of the credential ``subject`` names, through :data:`RECORDER`."""
    return RECORDER.note(subject, broker=broker, used_at=used_at)
