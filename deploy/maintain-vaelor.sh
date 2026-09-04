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
  vaelor-appliance-upgrade.service
  vaelor-application-research.service
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
  maintain-vaelor.sh uninstall --purge-data --bare-os \
    --confirm uninstall-vaelor-and-delete-all-data

Repair is an idempotent reinstall of the selected version. Uninstall keeps
users, credentials, models, workloads, chats, and settings unless --purge-data
is paired with the exact destructive confirmation phrase.

--bare-os additionally removes the OS stack the installer added - Docker,
InfluxDB, the ROCm gfx1151 packages and /opt/rocm, amd-smi, novnc, the AMD apt
source and keyring - and unmasks SunFounder's pironman5.service, returning the
machine to its pre-Vaelor state. It requires --purge-data and the destructive
confirmation. Shared OS tools (python3, curl, git, openssl) are left in place.
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
    bare_os=0
    confirmation=""
    while (($#)); do
      case "$1" in
        --purge-data) purge=1; shift ;;
        --bare-os) bare_os=1; shift ;;
        --confirm) confirmation="${2:-}"; shift 2 ;;
        *) usage >&2; exit 2 ;;
      esac
    done
    # A bare-OS teardown is a superset of a data purge - it removes shared OS
    # packages other software could depend on - so it is only accepted with the
    # destructive purge confirmation, never on its own.
    ((bare_os)) && ! ((purge)) && {
      echo "--bare-os requires --purge-data and the destructive confirmation." >&2
      exit 1
    }
    expected="uninstall-vaelor-keep-data"
    ((purge)) && expected="uninstall-vaelor-and-delete-all-data"
    [[ "${confirmation}" == "${expected}" ]] || {
      echo "Nothing changed. Use the exact confirmation shown in --help." >&2
      exit 1
    }
    # Tear down each managed workload's Compose project before anything stops
    # or removes Docker or the workloads. Removing Docker while these projects
    # stand orphans their bridge networks in the kernel, so a box that has
    # deployed and removed apps accumulates dead br-* interfaces. `compose
    # down` removes each project's network with its containers, and the final
    # prune sweeps any that outlived their project. Best-effort throughout: a
    # missing project, or a Docker that is already gone, must never block the
    # uninstall the operator confirmed.
    workloads_root="/var/lib/vaelor/workloads"
    if command -v docker >/dev/null && [[ -d "${workloads_root}" ]]; then
      for compose_file in "${workloads_root}"/*/compose.yaml; do
        [[ -f "${compose_file}" ]] || continue
        project_dir="$(dirname "${compose_file}")"
        docker compose --project-name "$(basename "${project_dir}")" \
          --project-directory "${project_dir}" -f "${compose_file}" \
          down --remove-orphans || true
      done
      docker network prune -f || true
    fi
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
      /etc/systemd/system/vaelor-workload-broker.service.d \
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
      if ((bare_os)); then
        # Return the machine to its pre-Vaelor state. Every step is best-effort:
        # a package already gone, or one the base image shipped that apt refuses
        # to remove, must never block the teardown the operator confirmed.
        export DEBIAN_FRONTEND=noninteractive
        # Lift Vaelor's mask on SunFounder's control plane (the installer masked
        # it so Vaelor could own the only web control plane). The installer also
        # removed SunFounder's own unit file, so unmask clears Vaelor's mark but
        # the SunFounder service returns only once its package is reinstalled.
        # Also drop the boot-load config for the HP sensors module (the module
        # itself ships with the distribution).
        systemctl unmask pironman5.service >/dev/null 2>&1 || true
        rm -f /etc/modules-load.d/vaelor-hp-wmi-sensors.conf
        # ROCm gfx1151: the installer holds the whole amdrocm-*/rocm-* closure so
        # an apt upgrade cannot break the pinned build - unhold before purging.
        # /opt/rocm is package-owned and goes with the purge; remove it too in
        # case a file was placed outside the manifest. The AMD apt source and its
        # keyring were added by the installer, so they go as well.
        mapfile -t rocm_held < <(dpkg-query -W -f='${Package}\n' 'amdrocm-*' 'rocm-*' 2>/dev/null || true)
        ((${#rocm_held[@]})) && apt-mark unhold "${rocm_held[@]}" >/dev/null 2>&1 || true
        apt-get purge -y 'amdrocm-*' 'rocm-*' >/dev/null 2>&1 || true
        rm -rf /opt/rocm
        rm -f /etc/apt/sources.list.d/amdrocm.list /etc/apt/keyrings/amdrocm.gpg
        # Docker, InfluxDB, amd-smi and novnc are the rest of the stack the
        # installer added. Shared OS tools (python3, curl, git, openssl) are
        # deliberately left in place - they are not Vaelor's to remove.
        apt-get purge -y docker.io docker-compose-v2 influxdb amd-smi novnc \
          >/dev/null 2>&1 || true
        apt-get autoremove --purge -y >/dev/null 2>&1 || true
        apt-get update >/dev/null 2>&1 || true
        # apt purge leaves the daemons' own data trees behind: /var/lib/docker
        # (images, volumes, container and network state) and /var/lib/influxdb
        # (the metrics TSM store) both survive a package purge, so a bare-OS
        # teardown removes them explicitly - otherwise multi-GB of Vaelor-era
        # container images and the metrics database outlive the wipe.
        rm -rf /var/lib/docker /var/lib/influxdb /etc/docker
        # The installer leaves a release snapshot (this script included) under
        # /usr/lib/vaelor, outside every tree removed above so it survives to run
        # the teardown. Remove it last. If THIS invocation is running from it,
        # hand the delete to a detached process so bash does not lose its own
        # file mid-read; a copy run from elsewhere (e.g. the recovery daemon's
        # /run copy) removes it directly.
        if [[ "${script_dir}" == /usr/lib/vaelor/* ]]; then
          setsid bash -c 'sleep 2; rm -rf /usr/lib/vaelor' >/dev/null 2>&1 &
        else
          rm -rf /usr/lib/vaelor
        fi
        echo "Vaelor and the OS stack it added (Docker, InfluxDB, ROCm, amd-smi) were removed; the machine is back to its pre-Vaelor state."
      else
        echo "Vaelor services, software, configuration, credentials, and data were removed."
      fi
    else
      echo "Vaelor software was removed. Data remains under /var/lib/vaelor."
    fi
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
