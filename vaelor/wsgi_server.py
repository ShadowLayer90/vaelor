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

import waitress

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


def _shutdown_write(sock):
    try:
        sock.shutdown(socket.SHUT_WR)
    except OSError:
        pass


class _TlsTerminator:
    """Blocking TLS listener that relays decrypted bytes to a loopback backend.

    Blocking sockets are the point: they handle the ``SSLWant*`` cases that broke
    waitress-over-TLS, so the terminator is a plain byte pump and waitress never
    sees a TLS socket.
    """

    def __init__(self, host, port, ssl_context, backend_port, log):
        self._host = host
        self._port = port
        self._ssl_context = ssl_context
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
        try:
            while not self._stopping.is_set():
                try:
                    client, _addr = listener.accept()
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
                    args=(client,),
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

    def _service(self, client):
        try:
            client.settimeout(TLS_HANDSHAKE_TIMEOUT)
            try:
                tls = self._ssl_context.wrap_socket(client, server_side=True)
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
            try:
                self._pipe(tls, backend)
            finally:
                _close(tls)
                _close(backend)
        finally:
            self._slots.release()

    @staticmethod
    def _pipe(tls, backend):
        def forward(src, dst):
            try:
                while True:
                    data = src.recv(PIPE_BUFFER)
                    if not data:
                        break
                    dst.sendall(data)
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
                host, port, context, self._backend_port, self._log
            )

    @property
    def secure(self):
        return self._secure

    @staticmethod
    def _build_ssl_context(ssl_context):
        if ssl_context is None:
            return None
        if isinstance(ssl_context, ssl.SSLContext):
            return ssl_context
        cert, key = ssl_context
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert, key)
        return context

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
