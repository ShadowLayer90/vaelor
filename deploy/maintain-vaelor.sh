#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
units=(
  vaelor-control-plane.service
  vaelor-credential-broker.service
  vaelor-workload-executor.service
  vaelor-workload-broker.service
  vaelor-system-update.service
  vaelor-host-desktop.service
  vaelor-hardware-bridge.service
  vaelor-appliance-recovery.service
  vaelor-vnc-gateway.service
  vaelor-vnc-tls-proxy.service
)

usage() {
  cat <<'EOF'
Usage:
  maintain-vaelor.sh status
  maintain-vaelor.sh repair --wheel FILE [--wheelhouse DIR]
  maintain-vaelor.sh uninstall --confirm uninstall-vaelor-keep-data
  maintain-vaelor.sh uninstall --purge-data \
    --confirm uninstall-vaelor-and-delete-all-data

Repair is an idempotent reinstall of the selected version. Uninstall keeps
users, credentials, models, workloads, chats, and settings unless --purge-data
is paired with the exact destructive confirmation phrase.
EOF
}

[[ "${EUID}" -eq 0 ]] || {
  echo "Run Vaelor maintenance as root." >&2
  exit 1
}
operation="${1:-}"
[[ -n "${operation}" ]] || { usage >&2; exit 2; }
shift

case "${operation}" in
  status)
    ((${#})) && { usage >&2; exit 2; }
    printf '%-44s %-12s %-12s\n' "SERVICE" "ACTIVE" "ENABLED"
    for unit in "${units[@]}"; do
      printf '%-44s %-12s %-12s\n' \
        "${unit}" \
        "$(systemctl is-active "${unit}" 2>/dev/null || true)" \
        "$(systemctl is-enabled "${unit}" 2>/dev/null || true)"
    done
    if [[ -d /var/lib/vaelor ]]; then
      echo
      du -sh /var/lib/vaelor
    fi
    ;;
  repair)
    wheel=""
    wheelhouse=""
    while (($#)); do
      case "$1" in
        --wheel) wheel="${2:-}"; shift 2 ;;
        --wheelhouse) wheelhouse="${2:-}"; shift 2 ;;
        *) usage >&2; exit 2 ;;
      esac
    done
    [[ -f "${wheel}" ]] || {
      echo "Repair requires a readable versioned wheel." >&2
      exit 1
    }
    arguments=(
      --wheel "${wheel}"
      --unattended
      --without-docker
    )
    [[ -z "${wheelhouse}" ]] || arguments+=(--wheelhouse "${wheelhouse}")
    exec "${script_dir}/install-vaelor.sh" "${arguments[@]}"
    ;;
  uninstall)
    purge=0
    confirmation=""
    while (($#)); do
      case "$1" in
        --purge-data) purge=1; shift ;;
        --confirm) confirmation="${2:-}"; shift 2 ;;
        *) usage >&2; exit 2 ;;
      esac
    done
    expected="uninstall-vaelor-keep-data"
    ((purge)) && expected="uninstall-vaelor-and-delete-all-data"
    [[ "${confirmation}" == "${expected}" ]] || {
      echo "Nothing changed. Use the exact confirmation shown in --help." >&2
      exit 1
    }
    if ((purge)) && command -v docker >/dev/null &&
      [[ -x /opt/vaelor/venv/bin/python ]]; then
      /opt/vaelor/venv/bin/python - <<'PY'
import json
import pathlib
import subprocess

root = pathlib.Path("/var/lib/vaelor/workloads").resolve()
listed = subprocess.run(
    ["docker", "ps", "-aq", "--no-trunc"],
    capture_output=True, text=True, check=False, timeout=15,
).stdout.splitlines()
managed = []
if listed:
    inspected = subprocess.run(
        ["docker", "inspect", *listed],
        capture_output=True, text=True, check=False, timeout=30,
    )
    if inspected.returncode == 0:
        for item in json.loads(inspected.stdout):
            labels = (item.get("Config") or {}).get("Labels") or {}
            working = labels.get("com.docker.compose.project.working_dir", "")
            try:
                candidate = pathlib.Path(working).resolve()
                owned = root in candidate.parents
            except (OSError, ValueError):
                owned = False
            if owned:
                managed.append(str(item.get("Id", "")))
if managed:
    subprocess.run(
        ["docker", "rm", "-f", *managed],
        check=True, timeout=120,
    )
PY
    fi
    if command -v docker >/dev/null &&
      [[ -f /var/lib/vaelor/workloads/system-web-research/compose.yaml ]]; then
      docker compose --project-name system-web-research \
        -f /var/lib/vaelor/workloads/system-web-research/compose.yaml \
        down --remove-orphans || true
    fi
    systemctl disable --now "${units[@]}" >/dev/null 2>&1 || true
    for unit in "${units[@]}"; do
      rm -f "/etc/systemd/system/${unit}"
    done
    rm -rf \
      /etc/systemd/system/vaelor-control-plane.service.d \
      /etc/systemd/system/vaelor-workload-executor.service.d \
      /etc/systemd/system/tigervncserver@:1.service.d \
      /opt/vaelor
    rm -f \
      /etc/tmpfiles.d/vaelor.conf \
      /etc/udev/rules.d/99-vaelor-rpi-vcio.rules \
      /etc/udev/rules.d/99-vaelor-rpi-cpu-fan.rules
    for type_path in /sys/class/thermal/cooling_device*/type; do
      [[ -e "${type_path}" ]] || continue
      [[ "$(cat "${type_path}" 2>/dev/null || true)" == "pwm-fan" ]] || continue
      state_path="${type_path%/type}/cur_state"
      chown root:root "${state_path}" || true
      chmod 0644 "${state_path}" || true
    done
    command -v udevadm >/dev/null && udevadm control --reload-rules || true
    systemctl daemon-reload
    systemctl reset-failed
    if ((purge)); then
      rm -rf /var/lib/vaelor /var/log/vaelor /run/vaelor /etc/vaelor
      for user in vaelor-research vaelor-vnc vaelor-secrets vaelor-workloads vaelor; do
        id -u "${user}" >/dev/null 2>&1 && userdel "${user}" || true
      done
      for group in vaelor-vnc vaelor-credentials vaelor-jobs vaelor; do
        getent group "${group}" >/dev/null && groupdel "${group}" || true
      done
      echo "Vaelor services, software, configuration, credentials, and data were removed."
    else
      echo "Vaelor software was removed. Data remains under /var/lib/vaelor."
    fi
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
