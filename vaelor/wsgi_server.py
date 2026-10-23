"""Production WSGI serving for the Vaelor control plane.

This replaces Werkzeug's *development* WSGI server
(``werkzeug.serving.make_server``) on the appliance's production port 34001.
That server announces itself as a development server and, worse for a hardening
pass, emits a ``Server: Werkzeug/<ver> Python/<ver>`` response header that
discloses the exact framework and interpreter versions (#205 MED, confirmed on
the live Pi at Vaelor 2.1.0a54).

Two measured facts shaped this design; neither is theoretical.

**waitress is a production WSGI server but does not terminate TLS, and it
cannot be made to.** Serving waitress directly over an ``ssl``-wrapped listening
socket was tried and *measured* to fail: waitress drives its sockets
non-blocking through ``wasyncore``, and a 60-connection keep-alive/large-body
probe raised ``ssl.SSLWantWriteError`` on every request. A single happy-path
request had passed first - a plausible mechanism is not proof (LESSONS). So TLS
is terminated *in front of* waitress, never inside it.

**The appliance already terminates TLS exactly this way for noVNC**
(``tls_proxy.py``): a thin terminator ahead of a plaintext loopback backend.
This mirrors that proven shape for the control plane. The terminator uses
*blocking* ``ssl`` sockets - which resolve the WantRead/WantWrite cases the
non-blocking path could not - and hands decrypted bytes to a loopback-only
waitress.

``request.is_secure`` still reports ``True`` (so HSTS is still emitted) because
the loopback waitress is told ``url_scheme="https"``: it only ever receives
terminator-forwarded traffic, which is always TLS, so the scheme is not a guess.

The version-disclosing ``Server`` header is removed *at the source*: waitress
omits the header entirely when its ``ident`` is falsy, so there is nothing to
strip after the fact and nothing that can leak the version.

Residuals (documented, not hidden):

* waitress serves one worker thread per in-flight request, and a streaming
  response holds its worker for the response's lifetime - the telemetry SSE
  (``/stream/telemetry``) is an infinite loop, and AI-chat streams the same way.
  ``WAITRESS_THREADS`` is therefore sized from an explicit
  viewers x streams-per-viewer + request-headroom budget (see the constant), not
  a flat guess: a flat 16 starved the control plane once >16 long-lived streams
  were open (#205 M4). Beyond that budget's worth of *genuinely concurrent*
  streams a single fixed pool is still the wrong tool - that load wants a reverse
  proxy or an async server - but for one Pi a generously sized pool is simpler
  and safe, and the live test on the box settles the true ceiling.
* ``shutdown()`` stops the terminator cleanly (its threads are owned here) and
  makes a best-effort ``close()`` on waitress; the daemon worker threads exit
  with the process. On the appliance systemd sends ``SIGTERM`` and the process
  is replaced, so graceful in-process shutdown is a convenience for the legacy
  embedded caller, not the appliance path.
"""

from __future__ import annotations

import logging
import socket
import ssl
import threading
import time

import waitress

from .tls_reload import IDLE_CLOSE_SECONDS, ReloadingContext, StaticContext

_LOGGER = logging.getLogger(__name__)

#: The terminator forwards to waitress here; waitress binds this and nothing
#: else when TLS is terminated in front of it.
BACKEND_HOST = "127.0.0.1"

#: Concurrent TLS connections the terminator will service before refusing new
#: ones. A bound is deliberate: the dev server it replaces spawned an unbounded
#: thread per request.
MAX_TLS_CONNECTIONS = 128

#: Peak dashboard viewers a single appliance serves at once - browser tabs
#: across the household's or small team's phones, tablets and laptops. Generous
#: for one Pi; the true ceiling is a live-test question, not a unit-test one
#: (#205 M4).
MAX_CONCURRENT_STREAM_VIEWERS = 16

#: Long-lived SSE streams one viewer can hold at the same time, each pinning one
#: waitress worker for the connection's whole life: the telemetry stream
#: (`/stream/telemetry`, an infinite `while True: ... sleep(1)`) AND an AI-chat
#: stream. This is why the pool cannot be sized as if a worker frees per request.
STREAMS_PER_VIEWER = 2

#: Workers held back for ordinary request/response calls so a dashboard API call
#: never queues behind the streams. Requests are sub-second, so this covers a
#: burst rather than scaling with viewers.
REQUEST_HEADROOM_THREADS = 16

#: waitress worker threads. Sized so every viewer's streams AND normal requests
#: get a worker instead of queueing, because a streaming response holds its
#: worker for its whole lifetime (see module docstring residuals). The dev server
#: replaced here used `threaded=True` - effectively unbounded - so a flat 16 was
#: a starvation regression (#205 M4): >16 concurrent long-lived streams exhausted
#: the pool and the NEXT request, dashboard API calls included, hung behind them.
#: Idle SSE workers are I/O-bound-cheap (a thread blocked on sleep/recv), so
#: sizing generously costs little. `test_wsgi_server.py` pins this at or above
#: its documented floor so it cannot silently drop back to a starving value.
WAITRESS_THREADS = (
    MAX_CONCURRENT_STREAM_VIEWERS * STREAMS_PER_VIEWER + REQUEST_HEADROOM_THREADS
)

#: Seconds allowed for the TLS handshake before the connection is dropped, so a
#: stalled handshake cannot hold a connection slot indefinitely.
TLS_HANDSHAKE_TIMEOUT = 10.0

#: Bytes moved per relay read in the terminator.
PIPE_BUFFER = 64 * 1024

#: waitress omits the ``Server`` header when ident is falsy. This empty string is
#: how the version disclosure (#205) is removed at the source rather than
#: stripped after the response is built.
SERVER_IDENT = ""


def _close(sock):
    try:
        sock.close()
    except OSError:
        pass


def _shutdown_both(sock):
    """End a relayed TLS connection from another thread.

    The OS socket is shut down directly (``socket.socket.shutdown``), not
    through ``SSLSocket.shutdown``, which would drop the TLS object under the
    pump threads still reading it; their blocked reads then return and the
    normal teardown runs.
    """
    try:
        socket.socket.shutdown(sock, socket.SHUT_RDWR)
    except OSError:
        pass


def _shutdown_write(sock):
    try:
        sock.shutdown(socket.SHUT_WR)
    except OSError:
        pass


class _PeerTable:
    """Which real client each terminator-to-waitress connection carries.

    W4d-D6: every request reached waitress from the terminator over loopback,
    so ``request.remote_addr`` was ``127.0.0.1`` for every browser - the
    Sessions list showed it for a LAN laptop, the login limiter keyed every
    client together, and a "loopback-only" route was loopback-only to nobody.
    The terminator knows the client it accepted and the loopback port it
    connected from; waitress reports that port as ``REMOTE_PORT``. Keying on it
    trusts no header a client could send (an ``X-Forwarded-For`` would be one).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._peers = {}
        #: Requests the app is still answering, per backend port (F4: a
        #: connection with one in progress is not idle, however quiet).
        self._busy = {}

    def begin(self, backend_port):
        with self._lock:
            port = int(backend_port)
            self._busy[port] = self._busy.get(port, 0) + 1

    def end(self, backend_port):
        with self._lock:
            port = int(backend_port)
            left = self._busy.get(port, 0) - 1
            if left > 0:
                self._busy[port] = left
            else:
                self._busy.pop(port, None)

    def busy(self, backend_port):
        with self._lock:
            return self._busy.get(int(backend_port), 0) > 0

    def register(self, backend_port, client_address):
        with self._lock:
            self._peers[int(backend_port)] = str(client_address)

    def forget(self, backend_port):
        with self._lock:
            self._peers.pop(int(backend_port), None)

    def lookup(self, backend_port):
        with self._lock:
            return self._peers.get(int(backend_port))


class _RealPeerMiddleware:
    """Put the terminator's client address into ``REMOTE_ADDR``.

    Only a request that arrived on the loopback backend from a port the
    terminator registered is rewritten; anything else keeps what waitress saw.
    The terminator's own address is kept as ``vaelor.tls_terminator_addr``.
    """

    def __init__(self, app, peers):
        self._app = app
        self._peers = peers

    def __call__(self, environ, start_response):
        if environ.get("REMOTE_ADDR") == BACKEND_HOST:
            try:
                client = self._peers.lookup(environ.get("REMOTE_PORT", ""))
            except (TypeError, ValueError):
                client = None
            if client:
                environ["vaelor.tls_terminator_addr"] = environ["REMOTE_ADDR"]
                environ["REMOTE_ADDR"] = client
                # In progress until the server closes the answer (a stream
                # included), so a certificate swap waits for it (F4).
                port = environ.get("REMOTE_PORT", "")
                self._peers.begin(port)
                try:
                    answer = self._app(environ, start_response)
                except BaseException:
                    self._peers.end(port)
                    raise
                if isinstance(answer, (list, tuple)):
                    # Already whole: nothing is left in progress, and wrapping
                    # it would hide its length from waitress (no keep-alive).
                    self._peers.end(port)
                    return answer
                return _EndOnClose(answer, lambda: self._peers.end(port))
        return self._app(environ, start_response)


class _EndOnClose:
    """A WSGI answer that calls ``done`` once when the server closes it."""

    def __init__(self, answer, done):
        self._answer = answer
        self._done = done

    def __iter__(self):
        return iter(self._answer)

    def close(self):
        try:
            close = getattr(self._answer, "close", None)
            if close is not None:
                close()
        finally:
            done, self._done = self._done, None
            if done is not None:
                done()


class _TlsTerminator:
    """Blocking TLS listener that relays decrypted bytes to a loopback backend.

    Blocking sockets are the point: they handle the ``SSLWant*`` cases that broke
    waitress-over-TLS, so the terminator is a plain byte pump and waitress never
    sees a TLS socket.
    """

    def __init__(self, host, port, ssl_context, backend_port, log, peers=None):
        self._peers = peers if peers is not None else _PeerTable()
        self._host = host
        self._port = port
        # A holder, not a context: ``current()`` is asked at every accept, so a
        # certificate the household authority renews is served from the next
        # handshake on with no restart (VD-212; see ``tls_reload``).
        self._contexts = (
            ssl_context if hasattr(ssl_context, "current")
            else StaticContext(ssl_context)
        )
        self._backend_port = backend_port
        self._log = log
        self._slots = threading.Semaphore(MAX_TLS_CONNECTIONS)
        self._listener = None
        self._stopping = threading.Event()

    def serve_forever(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self._host, self._port))
        listener.listen(64)
        self._listener = listener
        # A renewed certificate closes the connections set up under the old one
        # (VD-133 1.5.5 #1), so the files are watched, not only read at accept.
        if hasattr(self._contexts, "watch"):
            self._contexts.watch(self._stopping)
        try:
            while not self._stopping.is_set():
                try:
                    client, address = listener.accept()
                except OSError:
                    if self._stopping.is_set():
                        break
                    raise
                if not self._slots.acquire(blocking=False):
                    # Over the connection cap: refuse rather than queue, so a
                    # flood cannot grow memory without bound.
                    _close(client)
                    continue
                threading.Thread(
                    target=self._service,
                    args=(client, address),
                    name="vaelor-tls-conn",
                    daemon=True,
                ).start()
        finally:
            _close(listener)

    def shutdown(self):
        self._stopping.set()
        if self._listener is not None:
            # Closing the listening socket unblocks the accept() above.
            _close(self._listener)

    def _service(self, client, address=None):
        try:
            client.settimeout(TLS_HANDSHAKE_TIMEOUT)
            try:
                # Context and generation read together, so a handshake that
                # straddles a reload is tracked under the certificate it used.
                if hasattr(self._contexts, "current_with_generation"):
                    context, generation = self._contexts.current_with_generation()
                else:
                    context, generation = self._contexts.current(), None
                tls = context.wrap_socket(client, server_side=True)
            except (ssl.SSLError, OSError):
                _close(client)
                return
            tls.settimeout(None)
            try:
                backend = socket.create_connection(
                    (BACKEND_HOST, self._backend_port), timeout=5
                )
            except OSError:
                _close(tls)
                return
            # ``create_connection(timeout=5)`` caps only how long *establishing*
            # the loopback connection may take - fast and correct, since a slow
            # connect to our own waitress means it is wedged. But the returned
            # socket keeps that 5s as its *read* timeout, and ``_pipe`` then loops
            # on ``backend.recv()``. A handler that takes >5s to produce its full
            # response (e.g. the model-backed Custom Applications intake under
            # load) would make that recv raise ``socket.timeout``; the pump's
            # ``except OSError`` would end the loop and reset the client with no
            # HTTP response at ~5s - the "intake resets at ~5s" defect, and fatal
            # to any slow-but-alive non-streaming response. (Streaming responses
            # survive only because bytes keep arriving before the deadline.) The
            # front must not impose a read deadline shorter than a legitimate
            # request; the app + waitress own request lifetime. So clear it,
            # mirroring the tls side above. None (not a finite value) is chosen
            # deliberately: a finite read timeout would have to comfortably exceed
            # the longest legitimate request - chat runs up to MODEL_ANSWER_TIMEOUT
            # (~900s) - or it would just reintroduce this bug for long chats, and
            # any number picked "to be safe" invites exactly that drift. The only
            # cost of None is that a backend which accepts but never sends would
            # pin this pump thread (and its connection slot) indefinitely; that is
            # acceptable here because the peer is always the LOOPBACK waitress
            # owned by this same process (not a hostile remote), the pump threads
            # are daemon, and MAX_TLS_CONNECTIONS already bounds the slot count.
            backend.settimeout(None)
            # Registered before a byte is relayed, so waitress cannot read a
            # request from this port before its client is known (W4d-D6).
            backend_port = backend.getsockname()[1]
            if address:
                self._peers.register(backend_port, address[0])
            # Last bytes either way, stamped by the pumps: with no request in
            # progress, the measure of "idle" a certificate swap waits for (F4).
            activity = [time.monotonic()]

            def quiet():
                return (not self._peers.busy(backend_port)
                        and time.monotonic() - activity[0] >= IDLE_CLOSE_SECONDS)
            untrack = (self._contexts.track(lambda: _shutdown_both(tls), generation, quiet)
                       if hasattr(self._contexts, "track") else (lambda: None))
            try:
                self._pipe(tls, backend, activity)
            finally:
                untrack()
                self._peers.forget(backend_port)
                _close(tls)
                _close(backend)
        finally:
            self._slots.release()

    @staticmethod
    def _pipe(tls, backend, activity=None):
        stamp = activity if activity is not None else [0.0]

        def forward(src, dst):
            try:
                while True:
                    data = src.recv(PIPE_BUFFER)
                    if not data:
                        break
                    stamp[0] = time.monotonic()
                    dst.sendall(data)
                    stamp[0] = time.monotonic()
            except OSError:
                pass
            finally:
                # Signal EOF to the peer so its recv() returns and its forward
                # loop ends, instead of both threads blocking forever.
                _shutdown_write(dst)

        pump = threading.Thread(
            target=forward,
            args=(backend, tls),
            name="vaelor-tls-pump",
            daemon=True,
        )
        pump.start()
        forward(tls, backend)
        pump.join()


class ControlPlaneServer:
    """Serve the control-plane WSGI app on ``host:port`` through a production
    WSGI server, with TLS terminated in front of it rather than by a development
    server.

    API-compatible with the ``werkzeug.serving.make_server`` handle it replaces -
    ``serve_forever()``, ``shutdown()``, ``server_close()`` - so both call sites,
    ``vaelor.server.main()`` and the legacy ``VaelorControlPlane``, drive it the
    way they drove the dev server.
    """

    def __init__(self, host, port, app, ssl_context=None, log=None):
        self._host = host
        self._port = port
        self._log = log or _LOGGER
        context = self._build_ssl_context(ssl_context)
        self._secure = context is not None
        self._peers = _PeerTable()
        if self._secure:
            app = _RealPeerMiddleware(app, self._peers)
        # With TLS terminated in front, waitress binds loopback on an ephemeral
        # port; with no certificate configured it binds the public host directly
        # (unchanged from the dev server's no-TLS behaviour - not a downgrade).
        # url_scheme is pinned to https for the fronted case so is_secure/HSTS
        # still hold: the loopback backend only ever receives TLS-terminated
        # traffic, so the scheme is a fact, not a guess.
        self._waitress = waitress.create_server(
            app,
            host=BACKEND_HOST if self._secure else host,
            port=0 if self._secure else port,
            threads=WAITRESS_THREADS,
            ident=SERVER_IDENT,
            url_scheme="https" if self._secure else "http",
        )
        self._backend_port = self._waitress.effective_port
        self._waitress_thread = None
        self._terminator = None
        if self._secure:
            self._terminator = _TlsTerminator(
                host, port, context, self._backend_port, self._log,
                peers=self._peers,
            )

    @property
    def secure(self):
        return self._secure

    @staticmethod
    def _build_ssl_context(ssl_context):
        """A context holder: fixed for a caller's own ``SSLContext``, and
        reloading for the appliance's ``(cert, key)`` paths, so a renewed
        certificate is served without a restart (VD-212). The first load of the
        paths still raises - no certificate is a startup failure."""
        if ssl_context is None:
            return None
        if isinstance(ssl_context, ssl.SSLContext):
            return StaticContext(ssl_context)
        cert, key = ssl_context
        return ReloadingContext(cert, key)

    def serve_forever(self):
        if self._terminator is None:
            # No certificate configured: waitress owns the public socket and this
            # call blocks in it.
            self._waitress.run()
            return
        # TLS terminated in front: waitress serves the loopback backend on its
        # own thread, and this call blocks in the terminator on the public port.
        self._waitress_thread = threading.Thread(
            target=self._waitress.run,
            name="vaelor-wsgi-backend",
            daemon=True,
        )
        self._waitress_thread.start()
        self._terminator.serve_forever()

    def shutdown(self):
        if self._terminator is not None:
            self._terminator.shutdown()
        try:
            self._waitress.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass

    def server_close(self):
        self.shutdown()


def serve(app, host, port, ssl_context=None, log=None):
    """Build a :class:`ControlPlaneServer` and block serving it.

    This is the appliance entrypoint's serving call - the production replacement
    for ``make_server(...).serve_forever()``. ``KeyboardInterrupt`` returns
    cleanly, as it did for the dev server.
    """
    server = ControlPlaneServer(host, port, app, ssl_context=ssl_context, log=log)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server
