"""Where Vaelor's TLS material lives: the one home for every path and route (VD-212).

The household authority (a root unique to this install), the console's serving
certificate it issues, the bundle workers trust, and the public route the root
is downloaded from. Every reader and writer imports from here, so no two
modules can disagree about where the controller's certificate is (LESSONS 6:
three readers once derived it three ways).

Paths only - no logic that touches the files. The authority itself is
:mod:`vaelor.tls_authority`.
"""

from __future__ import annotations

from .runtime_paths import app_path, env_value, state_path

#: The authority's home. Under ``/etc/vaelor`` because a keep-data uninstall
#: removes ``/opt/vaelor`` but keeps ``/etc/vaelor``, and VD-212 keeps the root
#: across a keep-data reinstall so the owner's devices stay trusted.
AUTHORITY_DIR = "/etc/vaelor/authority"
#: The root's public certificate (0644).
ROOT_CERT = AUTHORITY_DIR + "/root.crt"
#: The root's private key, systemd-creds encrypted to this host (0600). It is
#: decrypted only into the authority's memory; never written in the clear.
ROOT_KEY_CRED = AUTHORITY_DIR + "/root-key.cred"

#: The public copy of the root the console serves and the installer prints the
#: fingerprint of (0644; recreated on every install).
PUBLIC_ROOT = app_path("tls/household-root.crt")
#: What workers trust (Telegraf ``tls_ca``, custom agents): the root, plus the
#: certificate still being served while an existing install migrates.
WORKER_TRUST_BUNDLE = app_path("tls/worker-trust.pem")
#: A new console certificate issued but not served yet (migration phase 0).
PENDING_DIR = app_path("tls/pending")


def leaf_cert() -> str:
    """The console's serving certificate: ``VAELOR_TLS_CERT``, else the installed file."""
    return env_value("VAELOR_TLS_CERT", "PM_TLS_CERT", app_path("tls/vaelor.crt"))


def leaf_key() -> str:
    """The console's serving key: ``VAELOR_TLS_KEY``, else the installed file."""
    return env_value("VAELOR_TLS_KEY", "PM_TLS_KEY", app_path("tls/vaelor.key"))


#: The control plane's advisory record of which workers hold the current trust
#: bundle; the authority reads it only to decide WHEN to promote, never what to
#: sign (it is writable by the control plane's account).
FLEET_TRUST_STATE = state_path("tls/fleet-trust.json")

#: The root in the controller's own OS trust store; ``<id>`` is the root's id.
CONTROLLER_OS_TRUST_TEMPLATE = "/usr/local/share/ca-certificates/vaelor-household-{}.crt"
#: The root in a worker's OS trust store (one root per worker: its controller's).
WORKER_OS_TRUST_FILE = "/usr/local/share/ca-certificates/vaelor-household-root.crt"

#: The one route that serves the root's public certificate WITHOUT sign-in, so a
#: first-time owner can trust the console before typing a password (owner,
#: VD-212 build answers). It serves nothing else.
ROOT_ROUTE = "/api/v2/security/trust/root.crt"
