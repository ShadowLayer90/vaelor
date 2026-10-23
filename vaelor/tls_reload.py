"""Serve a renewed console certificate without restarting the server (VD-212).

The household authority (``vaelor.tls_authority``, a separate root service)
re-issues the console's certificate - before it expires, when the machine's
addresses change, and once when an existing install migrates to the household
root - by replacing ``vaelor.crt`` and ``vaelor.key`` in place. Nothing restarts
the control plane or the noVNC terminator when it does, so both ask this holder
for the context at every handshake, and the holder notices the change.

**One rule decides when a reload happens: the files changed.** The signature is
the (device, inode, size, mtime) of both the certificate and the key, so an
``os.replace`` of either is seen, and a half-written pair - the certificate
replaced, the key not yet - is retried when the key lands instead of being
given up on (the signature moves again).

**Connections set up under the replaced certificate are closed** - each once
it is idle (no request in progress, no bytes either way for
:data:`IDLE_CLOSE_SECONDS`), and every one by :data:`CLOSE_CAP_SECONDS` after
the reload (owner ruling, round 2 F4: an upload, a restore or an AI Chat
stream in progress finishes first) - once the new one loads (VD-133 1.5.5 #1, adversarial review F1). TLS checks a certificate
only at the handshake, so a browser keep-alive connection opened under the old
certificate went on working after a re-issue: the open page read "no re-issue
planned" through it, forgot the announcement, and failed later with nothing
said. Closing those connections makes the page's next request open a new one,
which meets the new certificate while the page still remembers why. A holder
whose server only asks at a handshake also polls its files
(:meth:`ReloadingContext.watch`), so the close does not wait for some other
client to connect.

**A connection is tracked under the generation its handshake used**, read
together with the context (:meth:`ReloadingContext.current_with_generation`),
so a handshake that straddles a reload is closed as soon as it is tracked
rather than living on under the old certificate (review round 2, F1). A reload
whose certificate is byte-for-byte the one already served (a rewrite, a
``touch``) is not a new generation and closes nothing (F7).

**A reload that fails keeps the previous context** and logs once per
failure, not once per look (F5). A pair that fails to load is tried again on
the next look before it is called broken: the authority writes the key, then
the certificate, and a look between the two sees a pair that does not match
(F6) - that is logged at debug level, and only a pair that still fails at the
next look is a warning. It never logs the key or anything read from it:
the key is only ever handed to OpenSSL by path (LESSONS 24 / VD-124 - key
material stays out of logs). The FIRST load is different: it raises, because a
server with no certificate at all must fail loudly rather than serve nothing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import ssl
import threading
import time
from typing import Callable, Dict, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

#: (st_dev, st_ino, st_size, st_mtime_ns) for the certificate, then the key.
#: How often a watching holder looks at its files between handshakes.
WATCH_SECONDS = 5.0
#: A connection set up under a replaced certificate is closed once it has been
#: quiet this long in both directions with no request in progress (owner, F4).
IDLE_CLOSE_SECONDS = 2.0
#: ...and at the latest this long after the reload, whatever it is doing.
CLOSE_CAP_SECONDS = 15 * 60
Signature = Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]


def server_context(cert: str, key: str) -> ssl.SSLContext:
    """A TLS 1.2+ server context holding ``cert`` and ``key``.

    The one place both console terminators build their context, so the two
    cannot drift on the protocol floor (they once each carried a copy).
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert, key)
    return context


def _file_signature(path: str) -> Tuple[int, int, int, int]:
    stat = os.stat(path)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _certificate_digest(path: str) -> bytes:
    """A digest of the certificate file's bytes (raises OSError).

    The authority writes a certificate the same way every time, so the same
    bytes are the same certificate; anything else is a new one.
    """
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).digest()


def _failure_kind(error: BaseException) -> str:
    """Name a load failure without quoting anything read from the files."""
    reason = getattr(error, "reason", None) or getattr(error, "strerror", None)
    name = type(error).__name__
    return "{}: {}".format(name, reason) if reason else name


class ReloadingContext:
    """The console's current ``SSLContext``, re-read when its files change.

    ``current()`` is called once per accepted connection, before the handshake.
    It costs two ``stat`` calls when nothing changed.
    """

    def __init__(
        self,
        cert: str,
        key: str,
        *,
        build: Callable[[str, str], ssl.SSLContext] = server_context,
        log: Optional[logging.Logger] = None,
    ):
        self._cert = cert
        self._key = key
        self._build = build
        self._log = log or _LOGGER
        self._lock = threading.Lock()
        # The first load raises: no certificate at all is a startup failure.
        self._signature = self._read_signature()
        self._context = build(cert, key)
        self._cert_digest = _certificate_digest(cert)
        #: The signature that failed to load once, awaiting a second look.
        self._failed_once: Optional[Signature] = None
        #: The read failure last logged, so a missing file is said once.
        self._read_failure_logged: Optional[str] = None
        self._listening: Optional[ssl.SSLContext] = None
        #: Bumped by every reload that serves a different certificate; a tracked
        #: connection records the generation its handshake used.
        self.generation = 0
        #: When (monotonic) the current generation began serving.
        self._reloaded_at = time.monotonic()
        self._tracked: Dict[int, Tuple[int, Callable[[], None], Callable[[], bool]]] = {}
        self._next_ticket = 0

    def current_with_generation(self) -> Tuple[ssl.SSLContext, int]:
        """The context and its generation, read together (round 2, F1)."""
        self.current()
        with self._lock:
            return self._context, self.generation

    def track(self, close: Callable[[], None], generation: Optional[int] = None,
              quiet: Optional[Callable[[], bool]] = None) -> Callable[[], None]:
        """Register a connection set up under ``generation`` (default: current).

        Once a newer certificate serves, ``close`` is called - once - as soon as
        ``quiet()`` says the connection is idle, or :data:`CLOSE_CAP_SECONDS`
        after the reload, whichever is first (owner, F4: close when idle, so an
        upload, a restore or an AI Chat stream in progress finishes first).
        With no ``quiet`` the connection counts as idle. A connection whose
        handshake already used an older certificate is judged at once.
        Returns the call that forgets the connection when it ends on its own.
        """
        with self._lock:
            ticket = self._next_ticket
            self._next_ticket += 1
            used = self.generation if generation is None else generation
            self._tracked[ticket] = (used, close, quiet or (lambda: True))
            older = used < self.generation
        if older:
            self._close_older()

        def untrack() -> None:
            with self._lock:
                self._tracked.pop(ticket, None)
        return untrack

    def watch(self, stopping: threading.Event, interval: float = WATCH_SECONDS) -> threading.Thread:
        """Every ``interval`` seconds until ``stopping``: look at the files, and
        close the old-certificate connections that have gone quiet since."""
        def loop() -> None:
            while not stopping.wait(interval):
                self.current()
                self._close_older()
        thread = threading.Thread(target=loop, name="vaelor-tls-watch", daemon=True)
        thread.start()
        return thread

    def _close_older(self) -> None:
        """Close each old-certificate connection that is quiet, or past the cap."""
        with self._lock:
            capped = time.monotonic() - self._reloaded_at >= CLOSE_CAP_SECONDS
            older = [(ticket, close, quiet) for ticket, (generation, close, quiet)
                     in self._tracked.items() if generation < self.generation]
        due = []
        for ticket, close, quiet in older:
            try:
                ready = capped or quiet()
            except (OSError, ValueError):
                ready = True
            if ready:
                due.append((ticket, close))
        with self._lock:
            due = [(ticket, close) for ticket, close in due
                   if self._tracked.pop(ticket, None) is not None]
        for _ticket, close in due:
            try:
                close()
            except (OSError, ValueError):
                pass
        if due:
            self._log.info("Closed %d connection(s) set up under the replaced "
                           "console certificate.", len(due))

    def _read_signature(self) -> Signature:
        return (_file_signature(self._cert), _file_signature(self._key))

    def current(self) -> ssl.SSLContext:
        """The context to hand this connection; reloads first if the files moved."""
        try:
            signature = self._read_signature()
        except OSError as error:
            # Mid-replace, or the files were removed: keep serving what we have,
            # and say so once per kind of failure, not at every look (F5).
            kind = _failure_kind(error)
            if self._read_failure_logged != kind:
                self._read_failure_logged = kind
                self._log.warning(
                    "The console certificate at %s could not be read (%s); still "
                    "serving the previous certificate.", self._cert, kind)
            return self._context
        self._read_failure_logged = None
        if signature == self._signature:
            return self._context
        reloaded = False
        with self._lock:
            if signature == self._signature:
                return self._context
            try:
                context = self._build(self._cert, self._key)
                der = _certificate_digest(self._cert)
            except (ssl.SSLError, OSError, ValueError) as error:
                if self._failed_once == signature:
                    # Still failing at a second look: really broken. Recorded,
                    # so it is said once and not tried at every connection.
                    self._signature = signature
                    self._failed_once = None
                    self._log.warning(
                        "The console certificate at %s changed but could not be "
                        "loaded (%s); still serving the previous certificate.",
                        self._cert, _failure_kind(error))
                else:
                    # Likely caught between the key and certificate writes (F6).
                    self._failed_once = signature
                    self._log.debug(
                        "The console certificate at %s could not be loaded yet "
                        "(%s); looking again.", self._cert, _failure_kind(error))
                return self._context
            self._signature = signature
            self._failed_once = None
            self._context = context
            if der != self._cert_digest:
                self._cert_digest = der
                self.generation += 1
                self._reloaded_at = time.monotonic()
                reloaded = True
                self._log.info(
                    "Serving the renewed console certificate from %s.", self._cert)
        if reloaded:
            self._close_older()
        return context

    def listening_context(self) -> ssl.SSLContext:
        """A context for a server that takes ONE context up front (asyncio).

        ``asyncio.start_server(ssl=...)`` wraps every connection with the
        context it was given, so the reload happens inside the handshake: the
        context's server-name callback, which OpenSSL runs for every
        ClientHello (with or without a server name), points the connection at
        :meth:`current`.
        """
        if self._listening is None:
            listening = self._build(self._cert, self._key)

            def choose(ssl_object, _server_name, _initial):
                chosen = self.current()
                if ssl_object.context is not chosen:
                    ssl_object.context = chosen
                return None

            listening.sni_callback = choose
            self._listening = listening
        return self._listening


class StaticContext:
    """A caller-built ``SSLContext`` that is never reloaded (it names no files)."""

    def __init__(self, context: ssl.SSLContext):
        self._context = context

    def current(self) -> ssl.SSLContext:
        return self._context
