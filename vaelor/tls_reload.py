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

**A reload that fails keeps the previous context** and logs one line naming the
file and the kind of failure. It never logs the key or anything read from it:
the key is only ever handed to OpenSSL by path (LESSONS 24 / VD-124 - key
material stays out of logs). The FIRST load is different: it raises, because a
server with no certificate at all must fail loudly rather than serve nothing.
"""

from __future__ import annotations

import logging
import os
import ssl
import threading
from typing import Callable, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

#: (st_dev, st_ino, st_size, st_mtime_ns) for the certificate, then the key.
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
        self._listening: Optional[ssl.SSLContext] = None

    def _read_signature(self) -> Signature:
        return (_file_signature(self._cert), _file_signature(self._key))

    def current(self) -> ssl.SSLContext:
        """The context to hand this connection; reloads first if the files moved."""
        try:
            signature = self._read_signature()
        except OSError as error:
            # Mid-replace, or the files were removed: keep serving what we have.
            self._log.warning(
                "The console certificate at %s could not be read (%s); still "
                "serving the previous certificate.",
                self._cert, _failure_kind(error),
            )
            return self._context
        if signature == self._signature:
            return self._context
        with self._lock:
            if signature == self._signature:
                return self._context
            # Recorded whether or not the load works: a broken pair is tried
            # once per change, not once per connection.
            self._signature = signature
            try:
                context = self._build(self._cert, self._key)
            except (ssl.SSLError, OSError, ValueError) as error:
                self._log.warning(
                    "The console certificate at %s changed but could not be "
                    "loaded (%s); still serving the previous certificate.",
                    self._cert, _failure_kind(error),
                )
                return self._context
            self._context = context
            self._log.info(
                "Serving the renewed console certificate from %s.", self._cert
            )
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
