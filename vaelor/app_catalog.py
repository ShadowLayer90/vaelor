"""Curated application templates with appliance-safe Compose defaults."""

from __future__ import annotations

from typing import Any


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
}


def public_catalog() -> list[dict[str, Any]]:
    return [
        {key: value for key, value in template.items() if key != "volume"}
        for template in APP_TEMPLATES.values()
    ]


def render_compose(template_id: str, host_port: int) -> str:
    template = APP_TEMPLATES.get(template_id)
    if template is None:
        raise ValueError("Choose a supported application.")
    if isinstance(host_port, bool) or not isinstance(host_port, int):
        raise ValueError("Choose a whole-number application port.")
    port = host_port

    if not 1024 <= port <= 65535 or port in {34001, 34002}:
        raise ValueError("Choose an available port between 1024 and 65535.")
    memory = "128m" if template_id == "nginx-welcome" else "512m"
    lines = [
        "services:",
        "  app:",
        f"    image: {template['image']}",
        f"    container_name: vaelor-{template_id}",
        "    restart: unless-stopped",
        "    labels:",
        '      io.pironman.managed: "true"',
        f'      io.pironman.template: "{template_id}"',
        "    ports:",
        f'      - "{port}:{template["container_port"]}"',
        f"    mem_limit: {memory}",
        '    cpus: "2.0"',
    ]
    volume = template.get("volume")
    if volume:
        volume_name, target = volume
        lines.extend(["    volumes:", f"      - {volume_name}:{target}", "", "volumes:", f"  {volume_name}: {{}}"])
    return "\n".join(lines) + "\n"
