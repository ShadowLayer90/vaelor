"""Curated application templates with appliance-safe Compose defaults."""

from __future__ import annotations

import secrets
from typing import Any

from .runtime_paths import data_path

#: Bytes of entropy behind each auto-generated ``secret_env`` value. 24 bytes of
#: urlsafe base64 is ~32 characters and well past guessable for a browser IDE
#: sign-in password.
_SECRET_TOKEN_BYTES = 24


APP_TEMPLATES: dict[str, dict[str, Any]] = {
    "grafana": {
        "id": "grafana",
        "name": "Grafana",
        "category": "Dashboards",
        "description": "Build dashboards for metrics, sensors, and services.",
        "image": "grafana/grafana:latest",
        "default_port": 3000,
        "container_port": 3000,
        "memory": "512 MB",
        "memory_mib": 512,
        "storage": "Persistent dashboard database",
        "source": "Grafana Labs",
        "volume": ("grafana-data", "/var/lib/grafana"),
        "setup": {
            "login": "admin / admin",
            "note": "Grafana asks you to set a new password at first sign-in.",
            "steps": [
                "Open Grafana and sign in with the username and password admin / admin.",
                "Set a new password when Grafana asks you to.",
                "Add a data source, then build your first dashboard.",
            ],
        },
    },
    "uptime-kuma": {
        "id": "uptime-kuma",
        "name": "Uptime Kuma",
        "category": "Monitoring",
        "description": "Monitor websites and services with friendly status pages.",
        "image": "louislam/uptime-kuma:2",
        "default_port": 3001,
        "container_port": 3001,
        "memory": "512 MB",
        "memory_mib": 512,
        "storage": "Persistent monitor history",
        "source": "Uptime Kuma",
        "volume": ("uptime-data", "/app/data"),
        "setup": {
            "first_run_wizard": True,
            "steps": [
                "Open the app to start its first-run wizard.",
                "Create your Uptime Kuma admin account (a username and password).",
                "Add your first monitor to begin watching a site or service.",
            ],
        },
    },
    "nginx-welcome": {
        "id": "nginx-welcome",
        "name": "NGINX Welcome Site",
        "category": "Web",
        "description": "A lightweight web server for testing the app platform.",
        "image": "nginx:stable-alpine",
        "default_port": 8080,
        "container_port": 80,
        "memory": "128 MB",
        "memory_mib": 128,
        "storage": "No persistent data required",
        "source": "NGINX",
        "volume": None,
    },
    "vaultwarden": {
        "id": "vaultwarden",
        "name": "Vaultwarden",
        "category": "Security",
        "description": "Self-hosted password manager compatible with Bitwarden apps.",
        "image": "vaultwarden/server:latest",
        "default_port": 8222,
        "container_port": 80,
        "memory": "256 MB",
        "memory_mib": 256,
        "storage": "Encrypted vault database",
        "source": "Vaultwarden",
        "volume": ("vaultwarden-data", "/data"),
        "setup": {
            "https_required": True,
            "note": "The web vault needs HTTPS in front of it before you can create an account.",
            "steps": [
                "Put a TLS reverse proxy in front so Vaultwarden is reached over HTTPS - the plain http address cannot create accounts.",
                "Open it over HTTPS and create your admin account.",
                "Install a Bitwarden client app and point it at your server's HTTPS address.",
            ],
        },
    },
    "filebrowser": {
        "id": "filebrowser",
        "name": "File Browser",
        "category": "Files",
        "description": "Browse, upload, and share files through a tidy web interface.",
        "image": "filebrowser/filebrowser:latest",
        "default_port": 8082,
        "container_port": 80,
        "memory": "128 MB",
        "memory_mib": 128,
        "storage": "Indexed file database",
        "source": "FileBrowser",
        "volume": ("filebrowser-data", "/database"),
        "setup": {
            # The image sets a RANDOM admin password on first run and prints it
            # to the container log as "randomly generated password: ...". The
            # old "admin / admin" story was wrong and locked owners out; the
            # Logs tab does not redact that line, so point them at it.
            "login": "admin (password is generated on first run)",
            "steps": [
                "Open the Logs tab and copy the value after 'randomly generated password:'.",
                "Open File Browser and sign in as admin with that password.",
                "Change the password from the Settings page, then set the root folder.",
            ],
        },
    },
    "jellyfin": {
        "id": "jellyfin",
        "name": "Jellyfin",
        "category": "Media",
        "description": "Stream your own movies, shows, and music to any device.",
        "image": "jellyfin/jellyfin:latest",
        "default_port": 8096,
        "container_port": 8096,
        "memory": "1 GB",
        "memory_mib": 1024,
        "storage": "Library metadata and cache",
        "source": "Jellyfin",
        "volume": ("jellyfin-config", "/config"),
        # Media apps expose a second, browsable data root: the in-container path
        # where Vaelor bind-mounts a managed host directory for the library.
        # The file manager browses it and the compose renders exactly this one
        # controlled bind (see ``media_host_dir``/``render_compose``). Any other
        # Media-category app can opt in the same way by declaring this key.
        "media_mount": "/media",
        "setup": {
            "first_run_wizard": True,
            "steps": [
                "Open Jellyfin to start the setup wizard and create your administrator user.",
                "Add your media to the /media library: drop large files straight into its host folder on the appliance (<workloads>/jellyfin/media), or upload smaller files from the Files tab in this manager.",
                "In the wizard, add a library folder pointing at /media, then finish.",
            ],
        },
    },
    "gitea": {
        "id": "gitea",
        "name": "Gitea",
        "category": "Development",
        "description": "Lightweight self-hosted Git service with issues and reviews.",
        "image": "gitea/gitea:latest",
        "default_port": 3030,
        "container_port": 3000,
        "memory": "512 MB",
        "memory_mib": 512,
        "storage": "Git repositories and database",
        "source": "Gitea",
        "volume": ("gitea-data", "/data"),
        "setup": {
            "first_run_wizard": True,
            "steps": [
                "Open Gitea to reach its initial configuration page.",
                "Keep the SQLite defaults and register the first account - it becomes the administrator.",
                "Create your first repository.",
            ],
        },
    },
    "code-server": {
        "id": "code-server",
        "name": "code-server",
        "category": "Development",
        "description": "Run Visual Studio Code in the browser from any machine.",
        "image": "lscr.io/linuxserver/code-server:latest",
        "default_port": 8083,
        "container_port": 8443,
        "memory": "512 MB",
        "memory_mib": 512,
        "storage": "Editor settings and extensions",
        "source": "LinuxServer",
        "volume": ("code-server-config", "/config"),
        # The LinuxServer image comes up passwordless unless PASSWORD is set,
        # which would ship an unauthenticated browser IDE with a shell. Vaelor
        # auto-generates a unique PASSWORD per install (see secret_env below).
        "secret_env": ["PASSWORD"],
        "setup": {
            # The Configuration tab REDACTS the generated PASSWORD, so the old
            # "copy it from Configuration" story could not be followed. The
            # reveal endpoint (/managed/apps/<id>/credentials) hands the real
            # value to this panel instead.
            "note": (
                "Vaelor generated a unique PASSWORD for this app. It is shown "
                "in this panel so you can copy it to sign in."
            ),
            "steps": [
                "Copy the generated password shown in this panel.",
                "Click Open and configure, then sign in with that password.",
                "You can change it later from the app's own settings.",
            ],
        },
    },
    "heimdall": {
        "id": "heimdall",
        "name": "Heimdall",
        "category": "Dashboards",
        "description": "Organize links to all your self-hosted apps on one page.",
        "image": "lscr.io/linuxserver/heimdall:latest",
        "default_port": 8084,
        "container_port": 80,
        "memory": "128 MB",
        "memory_mib": 128,
        "storage": "Dashboard layout database",
        "source": "LinuxServer",
        "volume": ("heimdall-config", "/config"),
        "setup": {
            "steps": [
                "Open Heimdall - the dashboard starts empty.",
                "Use the add button to pin your first application tile.",
                "Arrange your tiles; Heimdall sets up no login by default.",
            ],
        },
    },
    "adguard-home": {
        "id": "adguard-home",
        "name": "AdGuard Home",
        "category": "Network",
        "description": (
            "Network-wide ad and tracker blocking DNS server. This port opens "
            "the first-run setup wizard; finish it and keep the admin UI on the "
            "same port."
        ),
        "image": "adguard/adguardhome:latest",
        "default_port": 3053,
        "container_port": 3000,
        "memory": "256 MB",
        "memory_mib": 256,
        "storage": "Filter configuration store",
        "source": "AdGuard",
        "volume": ("adguard-conf", "/opt/adguardhome/conf"),
        "extra_ports": [("dns-tcp", "tcp", 53), ("dns-udp", "udp", 53)],
        "setup": {
            "first_run_wizard": True,
            "note": "Needs host port 53 free (stop the system DNS resolver first).",
            "steps": [
                "Free host port 53 first (stop the system DNS resolver), or the deploy stays blocked.",
                "Open AdGuard Home to run its setup wizard and create an admin account.",
                "Point a device's DNS at this server to start filtering ads.",
            ],
        },
    },
    "syncthing": {
        "id": "syncthing",
        "name": "Syncthing",
        "category": "Files",
        "description": "Continuously sync files between your own devices, no cloud.",
        "image": "syncthing/syncthing:latest",
        "default_port": 8384,
        "container_port": 8384,
        "memory": "256 MB",
        "memory_mib": 256,
        "storage": "Folder sync configuration",
        "source": "Syncthing",
        "volume": ("syncthing-config", "/var/syncthing"),
        "extra_ports": [("sync-tcp", "tcp", 22000), ("sync-udp", "udp", 22000)],
        "setup": {
            "steps": [
                "Open Syncthing - its web GUI has no password yet.",
                "Set a GUI username and password under Actions then Settings.",
                "Add a folder and a remote device to begin syncing.",
            ],
        },
    },
    "it-tools": {
        "id": "it-tools",
        "name": "IT-Tools",
        "category": "Utilities",
        "description": "A handy collection of developer and sysadmin tools.",
        "image": "corentinth/it-tools:latest",
        "default_port": 8085,
        "container_port": 80,
        "memory": "64 MB",
        "memory_mib": 64,
        "storage": "Stateless web utility",
        "source": "IT-Tools",
        "volume": None,
    },
    "n8n": {
        "id": "n8n",
        "name": "n8n",
        "category": "Automation",
        "description": "Build workflow automations by wiring apps together visually.",
        "image": "docker.n8n.io/n8nio/n8n:latest",
        "default_port": 5678,
        "container_port": 5678,
        "memory": "1 GB",
        "memory_mib": 1024,
        "storage": "Workflow database and credentials",
        "source": "n8n",
        "volume": ("n8n-data", "/home/node/.n8n"),
        "setup": {
            "first_run_wizard": True,
            "steps": [
                "Open n8n to create the owner account with your email and a password.",
                "Complete the short onboarding screens.",
                "Build your first workflow on the canvas.",
            ],
        },
    },
    "homepage": {
        "id": "homepage",
        "name": "Homepage",
        "category": "Dashboards",
        "description": "A modern, fully static start page for your services.",
        "image": "ghcr.io/gethomepage/homepage:latest",
        "default_port": 3010,
        "container_port": 3000,
        "memory": "128 MB",
        "memory_mib": 128,
        "storage": "Service configuration files",
        "source": "Homepage",
        "volume": ("homepage-config", "/app/config"),
        # Homepage is configured by editing YAML files in its data volume, not
        # through a web UI. These are the files the app-files editor exposes
        # for read/write, relative to the volume mount (/app/config).
        "config_files": [
            "services.yaml", "settings.yaml", "bookmarks.yaml",
            "widgets.yaml", "docker.yaml",
        ],
        # Disable homepage's host-header allowlist. Homepage >=0.9 refuses to
        # answer unless the request Host matches HOMEPAGE_ALLOWED_HOSTS; the
        # appliance is reached by an arbitrary IP/hostname, so "*" lets it
        # respond on whatever host the operator points at it.
        "env": {"HOMEPAGE_ALLOWED_HOSTS": "*"},
        "setup": {
            "steps": [
                "Open the Files tab in this manager.",
                "Edit services.yaml and bookmarks.yaml to add your services and links.",
                "Reload Homepage to see your changes.",
            ],
        },
    },
}


def public_catalog() -> list[dict[str, Any]]:
    return [
        {key: value for key, value in template.items() if key != "volume"}
        for template in APP_TEMPLATES.values()
    ]


def _yaml_double_quoted(value: str) -> str:
    """Return ``value`` as a YAML double-quoted scalar.

    Only ``\\`` and ``"`` carry meaning inside a double-quoted YAML scalar, so
    escaping those two is enough to round-trip any value - including ones with
    ``:``, ``#``, leading ``*``/``{``/``[``, quotes, or spaces - back through a
    YAML parser unchanged.
    """
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _parse_double_quoted(token: str) -> str:
    """Inverse of :func:`_yaml_double_quoted` for a ``"..."`` scalar token."""
    inner = token[1:-1]
    out = []
    index = 0
    while index < len(inner):
        char = inner[index]
        if char == "\\" and index + 1 < len(inner):
            out.append(inner[index + 1])
            index += 2
        else:
            out.append(char)
            index += 1
    return "".join(out)


def install_env_from_compose(template_id: str, compose_text: str) -> dict[str, str]:
    """Recover a template's ``secret_env`` values from an existing compose file.

    Re-installing an app renders the compose fresh and refuses to proceed if it
    differs from the one on disk, so a re-install must reproduce the *same*
    generated secret rather than mint a new one. This reads the persisted value
    straight back out of the compose Vaelor itself wrote.
    """
    template = APP_TEMPLATES.get(template_id) or {}
    wanted = set(template.get("secret_env", []))
    recovered: dict[str, str] = {}
    if not wanted:
        return recovered
    for line in compose_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        payload = stripped[2:].strip()
        if len(payload) >= 2 and payload.startswith('"') and payload.endswith('"'):
            payload = _parse_double_quoted(payload)
        key, separator, value = payload.partition("=")
        if separator and key in wanted:
            recovered[key] = value
    return recovered


def declared_config_targets(template_id: str) -> list[str]:
    """In-container paths of a template's editable configuration files.

    The workload broker uses this to bound ``docker cp`` to exactly the files a
    template declares editable (see ``config_files``), independently of any
    caller-supplied path - the broker validates its own inputs rather than
    trusting the control plane. Returns an empty list for a template that
    declares no editable files or has no data volume, so the broker's
    exact-membership check fails closed.
    """
    template = APP_TEMPLATES.get(template_id) or {}
    files = template.get("config_files") or []
    volume = template.get("volume")
    if not files or not volume:
        return []
    _name, target = volume
    base = str(target).rstrip("/")
    return [f"{base}/{relative}" for relative in files]


def declared_data_roots(template_id: str) -> list[str]:
    """In-container directory roots the file manager may browse for a template.

    This is the template's data volume mount (``volume``'s 2nd element) plus,
    for a Media-category app that opts in, its ``media_mount`` container path.
    Both the workload broker and :mod:`vaelor.workload_files` confine every
    file-manager path to being equal to or under one of these roots, so a
    template with no volume and no media mount (``nginx-welcome``, ``it-tools``)
    returns an empty list and confinement fails closed. Paths are normalized
    with no trailing slash.
    """
    template = APP_TEMPLATES.get(template_id) or {}
    roots: list[str] = []
    volume = template.get("volume")
    if volume:
        _name, mount = volume
        roots.append(str(mount).rstrip("/"))
    media_mount = template.get("media_mount")
    if media_mount:
        roots.append(str(media_mount).rstrip("/"))
    return roots


def media_host_dir(template_id: str) -> str | None:
    """Absolute host path bind-mounted at the template's ``media_mount``.

    ``None`` when the template declares no ``media_mount``. It is placed under
    the managed data tree (``<data>/workloads/<template_id>/media``) so it stays
    inside the tree Vaelor already owns and confines to, and so it is never an
    arbitrary operator-chosen host path. This is the ONE controlled bind mount
    the compose renderer is allowed to emit.
    """
    template = APP_TEMPLATES.get(template_id) or {}
    if not template.get("media_mount"):
        return None
    # The bind is rendered into a Compose file consumed by Docker on the Linux
    # appliance, so the host path is always POSIX even when this renders on a
    # non-POSIX developer machine.
    return data_path(f"workloads/{template_id}/media").replace("\\", "/")


def data_path_is_within_roots(template_id: str, container_path: Any) -> bool:
    """True when ``container_path`` is equal to or under a browsable data root.

    Shared by the workload broker's ``_managed_data_path`` and
    :mod:`vaelor.workload_files` so the confinement rule has exactly one
    encoding (LESSONS 6): an absolute, ``..``-free in-container path whose
    normalized form equals or sits under one of :func:`declared_data_roots`.
    The check is on path segments, not a raw string prefix, so ``/config`` does
    not admit ``/configX``. A template with no data roots admits nothing.
    """
    if not isinstance(container_path, str) or not container_path.startswith("/"):
        return False
    segments = container_path.split("/")
    if ".." in segments:
        return False
    normalized = "/" + "/".join(
        segment for segment in segments if segment not in ("", ".")
    )
    for root in declared_data_roots(template_id):
        if normalized == root or normalized.startswith(root + "/"):
            return True
    return False


def build_install_env(template_id: str, existing_compose: str | None = None) -> dict[str, str] | None:
    """Resolve the install-time env for a template's ``secret_env`` keys.

    On a first install (``existing_compose is None``) each key gets a fresh,
    unique secret. On a re-install the already-persisted values are reused so
    the rendered compose matches the one on disk byte-for-byte and the password
    never rotates.
    """
    keys = APP_TEMPLATES.get(template_id, {}).get("secret_env", [])
    if not keys:
        return None
    if existing_compose is not None:
        return install_env_from_compose(template_id, existing_compose)
    return {key: secrets.token_urlsafe(_SECRET_TOKEN_BYTES) for key in keys}


def render_compose(template_id: str, host_port: int, install_env: dict[str, str] | None = None) -> str:
    template = APP_TEMPLATES.get(template_id)
    if template is None:
        raise ValueError("Choose a supported application.")
    if isinstance(host_port, bool) or not isinstance(host_port, int):
        raise ValueError("Choose a whole-number application port.")
    port = host_port

    if not 1024 <= port <= 65535 or port in {34001, 34002}:
        raise ValueError("Choose an available port between 1024 and 65535.")
    extra_ports = template.get("extra_ports", [])
    if port in {int(extra_port) for _name, _proto, extra_port in extra_ports}:
        raise ValueError("Choose a different port; this app already reserves it.")
    memory = f"{int(template['memory_mib'])}m"
    lines = [
        "services:",
        "  app:",
        f"    image: {template['image']}",
        f"    container_name: vaelor-{template_id}",
        "    restart: unless-stopped",
        "    labels:",
        '      io.vaelor.managed: "true"',
        f'      io.vaelor.template: "{template_id}"',
        "    ports:",
        f'      - "{port}:{template["container_port"]}"',
    ]
    for _name, proto, extra_port in extra_ports:
        lines.append(f'      - "{extra_port}:{extra_port}/{proto}"')
    lines.extend([
        f"    mem_limit: {memory}",
        '    cpus: "2.0"',
    ])
    env = dict(template.get("env") or {})
    if install_env:
        env.update(install_env)
    if env:
        lines.append("    environment:")
        for key, value in env.items():
            # Double-quote every entry so a value with a YAML metacharacter
            # (``a: b``, ``foo #bar``, a leading ``*``/``{``/``[``, quotes,
            # spaces) survives round-trip through a YAML parser instead of
            # silently misparsing.
            lines.append(f"      - {_yaml_double_quoted(f'{key}={value}')}")
    volume = template.get("volume")
    media_dir = media_host_dir(template_id)
    service_volumes: list[str] = []
    volume_name = None
    if volume:
        volume_name, target = volume
        service_volumes.append(f"      - {volume_name}:{target}")
    # The ONE permitted bind: the template's own managed media host directory at
    # its declared media_mount. No arbitrary host mounts, no docker.sock, no
    # privileged access - this is the media library root the file manager edits.
    if media_dir:
        service_volumes.append(f"      - {media_dir}:{template['media_mount']}")
    if service_volumes:
        lines.append("    volumes:")
        lines.extend(service_volumes)
    if volume_name:
        # Only the named volume needs a top-level declaration; the bind does not.
        lines.extend(["", "volumes:", f"  {volume_name}: {{}}"])
    return "\n".join(lines) + "\n"
