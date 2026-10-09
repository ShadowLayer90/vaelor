#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The interpreter that answers "which containers and container images are
# Vaelor's": the appliance venv's python, importing the constants the serving
# modules hold, so this script never retypes a container name or an image
# digest and a re-pin in the package is a re-pin here. A variable for ONE
# reason: so tests/test_installer_provisioning.py can execute the uninstall path
# with the repository's own interpreter and prove the argv it produces matches
# those constants. Nothing on the appliance ever sets it.
vaelor_python="${VAELOR_MAINTAIN_PYTHON:-/opt/vaelor/venv/bin/python}"
# Where the systemd units live and where the installer records what it added.
# Overridable for ONE reason, like vaelor_python above: so the uninstall path
# can be executed by tests/test_installer_provisioning.py against injected unit
# files and marker without touching the host. Nothing on the appliance sets
# either - both default to the real appliance path.
systemd_dir="${VAELOR_SYSTEMD_DIR:-/etc/systemd/system}"
# One line per OS package the installer (or Vaelor, on the operator's request)
# actually installed, written the moment it ran apt-get for it. --bare-os purges
# only what this names, so a Docker, InfluxDB or amd-smi the operator had before
# Vaelor is never removed. Its absence means an appliance installed before Vaelor
# recorded this, and the teardown falls back with a warning.
installed_marker="${VAELOR_INSTALLED_MARKER:-/usr/lib/vaelor/release/installed-by-vaelor}"
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

Repair is an idempotent reinstall of the selected version. It runs the installer
with --without-docker, so it never touches the container engine the appliance is
already running on - and therefore does not pre-pull the serving container images
either. A fresh install does that; after a repair each image is pulled by the
first deploy that needs it.

Every uninstall stops the containers Vaelor runs - the managed application
projects, the GPU AI-Chat model and the LLM Server proxy. It also stops and
removes any GPU-cluster (Mode B) serving this node runs - the vLLM servers, Ray
workers and gates, the model-pull oneshots, and the balancer container - so no
LLM is left answering on the LAN after Vaelor is gone. A worker node's own
replicas and gates are removed by that worker's uninstall, not this one. Without
--purge-data it keeps users, credentials, models, workloads, chats, settings,
and every container image, so a reinstall is fast.

Every uninstall also removes the GPU memory pool setting Vaelor wrote
(/etc/modprobe.d/vaelor-gpu-memory.conf) and rebuilds the boot image. The
pool returns to the kernel's own size at the next restart, which this script
does not perform.

--purge-data deletes all of that, including the container images Vaelor pulled:
the two gfx1151 serving images, the LLM Server's nginx and the multi-node vLLM
image, each removed by the exact reference pinned in the installed Vaelor
package (by digest, the form a pinned pull is stored under). It also removes the
browser-desktop account (vaelor-desktop) with its home and the VNC display it
enabled. Images Vaelor did not pull, and Docker itself, are left alone. It
requires the exact destructive confirmation phrase.

--bare-os additionally removes the OS stack the installer added - Docker and
containerd with their whole image stores (/var/lib/docker AND
/var/lib/containerd, where Docker 29 keeps images: so EVERY container image on
the box goes, not only Vaelor's), the docker group and Docker's leftover
bridges, InfluxDB with its database and /etc/influxdb, the ROCm gfx1151
packages and /opt/rocm, amd-smi, novnc, the TigerVNC and GNOME Flashback
packages the browser desktop installed, the AMD apt source and keyring - and
unmasks SunFounder's pironman5.service, returning the machine to its
pre-Vaelor state. It requires --purge-data and the destructive confirmation.
It purges only the OS packages Vaelor recorded installing (the marker at
/usr/lib/vaelor/release/installed-by-vaelor), so a Docker, InfluxDB or amd-smi
that was already present when Vaelor was installed is left alone; an appliance
installed before Vaelor kept that record falls back to purging the full stack
its installer can add and says so.
KEPT, deliberately: the shared OS tools the installer merely made sure of
(python3, curl, git, openssl), and anything the installer did not add - a
FastFlowLM package installed by hand under /opt/fastflowlm, model copies
preserved under /opt/flm-models-preserved, and the operator's own accounts,
files and packages.
EOF
}

# Docker names this script never retypes. $1 selects the list: `containers` is
# the serving containers the root hardware bridge launches (the GPU AI-Chat
# model and the LLM Server proxy) and `images` is every container image Vaelor
# pulls (the two gfx1151 serving images, the proxy's nginx and every pinned
# multi-node vLLM image - the default and any other a deployment may choose),
# one per line, read from the installed package's constants.
# Prints nothing when the venv cannot answer - a broken runtime must not block
# the uninstall the operator confirmed - and the caller says so.
vaelor_docker_names() {
  [[ -x "${vaelor_python}" ]] || return 0
  "${vaelor_python}" - "$1" <<'PY' 2>/dev/null || true
import sys
from vaelor.gpu_pool_runtime import VLLM_IMAGES
from vaelor.gpu_rocmfpx_service import GPU_CONTAINER_IMAGE, GPU_CONTAINER_NAME, GPU_FORK_IMAGE
from vaelor.llm_server_proxy import PROXY_CONTAINER_NAME, PROXY_IMAGE
names = {
    "containers": (GPU_CONTAINER_NAME, PROXY_CONTAINER_NAME),
    "images": (PROXY_IMAGE, GPU_CONTAINER_IMAGE, GPU_FORK_IMAGE, *VLLM_IMAGES),
}
print("\n".join(names[sys.argv[1]]))
PY
}

# apt purge ONE name at a time, then read back dpkg's verdict for each instead
# of trusting apt's exit status through the `|| true` every best-effort step here
# needs. One name per invocation because `apt-get purge foo bar unknown` takes a
# single unlocatable name as fatal to the WHOLE list - so one package apt cannot
# find (a typo, or one the base image never had) would veto the purge of every
# real package beside it; purged singly, a bad name loses only its own line.
#
# The first cold teardown found InfluxDB left at dpkg state `ic` (removed, its
# configuration kept) with the failure invisible: its postrm runs `systemctl
# disable influxdb` a second time on the purge step, after the remove step has
# already deleted the unit, and exits non-zero. A package that is removed but
# not purged is finished by neutralising that postrm and purging again, and
# anything still installed afterwards is named on stderr with the command that
# finishes it, so a leftover is a message rather than a silent survivor. Any
# dpkg state whose current-status field is `n` (`un`, `pn`, ...) means "not
# installed" and counts as gone, so a planned-but-absent package raises no false
# "still at" NOTE.
purge_os_packages() {
  local package state
  for package in "$@"; do
    apt-get purge -y "${package}" >/dev/null 2>&1 || true
  done
  for package in "$@"; do
    state="$(dpkg-query -W -f='${db:Status-Abbrev}' "${package}" 2>/dev/null || true)"
    case "${state}" in
      "" | ?n*) continue ;;
      rc* | ic*)
        printf '#!/bin/sh\nexit 0\n' |
          tee "/var/lib/dpkg/info/${package}.postrm" >/dev/null || true
        chmod 0755 "/var/lib/dpkg/info/${package}.postrm" 2>/dev/null || true
        dpkg --purge "${package}" >/dev/null 2>&1 || true
        state="$(dpkg-query -W -f='${db:Status-Abbrev}' "${package}" 2>/dev/null || true)"
        ;;
    esac
    case "${state}" in
      "" | ?n*) ;;
      *)
        echo "Package ${package} is still at dpkg state '${state% }';" \
          "finish with: apt-get purge ${package}" >&2
        ;;
    esac
  done
}

# `id -u` rather than bash's own EUID for one reason: EUID is set by the shell
# before any script runs, so an execution test cannot stand in for root, while
# `id` - the command this script already asks about every account below - can
# be answered from PATH. On the appliance the two are the same number.
[[ "$(id -u)" -eq 0 ]] || {
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
    # Stop Vaelor's own services FIRST. The hardware bridge and the executor
    # both watch the containers they launched and relaunch one that disappears,
    # so a container removed while they still run comes straight back and then
    # holds the image the purge below is meant to remove. The teardown itself
    # runs as a transient unit owned by pid 1 when the console launched it
    # (appliance_recovery.perform_uninstall), so stopping every Vaelor unit here
    # cannot stop this script.
    systemctl disable --now "${units[@]}" >/dev/null 2>&1 || true
    # Stop and remove any GPU-cluster (Mode B) serving THIS node runs, in EVERY
    # mode and before the serving containers below: a cluster left running keeps
    # a vLLM server up, and that server holds the vLLM image the purge is meant
    # to remove. The servers, Ray workers and gates are systemd units the mode
    # switch wrote, and the model pulls are oneshot units; their names are
    # derived in vaelor/gpu_pool_units.py, so this globs the unit FILES rather
    # than respelling the names here. The balancer is a container, not a unit
    # (vaelor-vllm-<name>-balancer), removed by name prefix. A worker node's own
    # replicas and gates are that worker's uninstall to remove, not this one.
    cluster_units=()
    for unit_file in "${systemd_dir}"/vaelor-vllm-*.service \
      "${systemd_dir}"/vaelor-model-pull-*.service; do
      [[ -e "${unit_file}" ]] || continue
      cluster_units+=("$(basename "${unit_file}")")
    done
    if ((${#cluster_units[@]})); then
      systemctl disable --now "${cluster_units[@]}" >/dev/null 2>&1 || true
      for unit in "${cluster_units[@]}"; do
        rm -f "${systemd_dir}/${unit}"
      done
      rm -f /etc/vaelor/vllm/*.conf
      systemctl daemon-reload
    fi
    if command -v docker >/dev/null; then
      mapfile -t cluster_balancers < <(
        docker ps -aq --no-trunc \
          --filter 'name=^vaelor-vllm-.*-balancer$' 2>/dev/null || true
      )
      if ((${#cluster_balancers[@]})); then
        docker rm -f "${cluster_balancers[@]}" >/dev/null 2>&1 || true
      fi
    fi
    # Tear down each managed workload's Compose project before anything removes
    # Docker or the workloads. Removing Docker while these projects stand
    # orphans their bridge networks in the kernel, so a box that has deployed
    # and removed apps accumulates dead br-* interfaces. `compose down` removes
    # each project's network with its containers; a scoped prune sweeps only a
    # network THAT project left behind. A bare `docker network prune` would take
    # every unused network on the box, an operator's own included, so the prune
    # is filtered to the project's Compose label. Best-effort throughout: a
    # missing project, or a Docker that is already gone, must never block the
    # uninstall the operator confirmed.
    workloads_root="${VAELOR_WORKLOADS_ROOT:-/var/lib/vaelor/workloads}"
    if command -v docker >/dev/null && [[ -d "${workloads_root}" ]]; then
      for compose_file in "${workloads_root}"/*/compose.yaml; do
        [[ -f "${compose_file}" ]] || continue
        project_dir="$(dirname "${compose_file}")"
        project="$(basename "${project_dir}")"
        docker compose --project-name "${project}" \
          --project-directory "${project_dir}" -f "${compose_file}" \
          down --remove-orphans || true
        docker network prune -f \
          --filter "label=com.docker.compose.project=${project}" \
          >/dev/null 2>&1 || true
      done
    fi
    if command -v docker >/dev/null; then
      # The serving containers the root bridge launched with `docker run`, not
      # Compose, so no project above covers them. Removed on EVERY uninstall:
      # they are Vaelor services, not data, and an LLM left answering on the
      # LAN after Vaelor is gone is nobody's.
      mapfile -t serving_containers < <(vaelor_docker_names containers)
      if ((${#serving_containers[@]})); then
        docker rm -f "${serving_containers[@]}" >/dev/null 2>&1 || true
      else
        echo "The Vaelor runtime at ${vaelor_python} could not name its serving" \
          "containers; any still running are left for docker ps to show." >&2
      fi
    fi
    if ((purge)) && command -v docker >/dev/null && [[ -x "${vaelor_python}" ]]; then
      # Any container a managed Compose project left behind, found by the
      # project directory Compose labelled it with rather than by name, so a
      # renamed or half-removed project is still Vaelor's to remove. Python
      # only reads the inspect JSON on its stdin (the program is passed with
      # -c for exactly that reason - a heredoc would take stdin from the
      # pipe); docker is called from here, and a container that cannot be
      # removed stops the purge the way it always has.
      mapfile -t listed < <(docker ps -aq --no-trunc 2>/dev/null || true)
      managed=()
      if ((${#listed[@]})); then
        mapfile -t managed < <(docker inspect "${listed[@]}" 2>/dev/null |
          "${vaelor_python}" -c '
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
try:
    items = json.load(sys.stdin)
except ValueError:
    items = []
for item in items:
    labels = (item.get("Config") or {}).get("Labels") or {}
    working = labels.get("com.docker.compose.project.working_dir", "")
    try:
        owned = root in pathlib.Path(working).resolve().parents
    except (OSError, ValueError):
        owned = False
    if owned:
        print(item.get("Id", ""))
' "${workloads_root}" || true)
      fi
      if ((${#managed[@]})); then
        docker rm -f "${managed[@]}"
      fi
    fi
    if command -v docker >/dev/null &&
      [[ -f /var/lib/vaelor/workloads/system-web-research/compose.yaml ]]; then
      docker compose --project-name system-web-research \
        -f /var/lib/vaelor/workloads/system-web-research/compose.yaml \
        down --remove-orphans || true
    fi
    if ((purge)) && command -v docker >/dev/null; then
      # The container images Vaelor pulled are Vaelor's data, so a data purge
      # removes them - by the pinned reference the package holds, which is the
      # digest form a pinned pull is stored under (such an image shows as
      # `<none>` in `docker images`, and only the digest reference finds it).
      # Removing by reference drops only that reference: an image the operator
      # also tagged, or another container still uses, stays. An image that is
      # not present is skipped; one that cannot be removed is named on stderr.
      mapfile -t serving_images < <(vaelor_docker_names images)
      if ((${#serving_images[@]})); then
        for image in "${serving_images[@]}"; do
          docker image inspect "${image}" >/dev/null 2>&1 || continue
          if docker image rm "${image}" >/dev/null 2>&1; then
            echo "Removed container image ${image}"
          else
            echo "Could not remove container image ${image}; it is still in" \
              "use or already gone. Finish with: docker image rm ${image}" >&2
          fi
        done
      else
        echo "The Vaelor runtime at ${vaelor_python} could not name the" \
          "container images it pulled; they are left for docker images to show." >&2
      fi
    fi
    for unit in "${units[@]}"; do
      rm -f "/etc/systemd/system/${unit}"
    done
    rm -rf \
      /etc/systemd/system/vaelor-control-plane.service.d \
      /etc/systemd/system/vaelor-workload-executor.service.d \
      /etc/systemd/system/vaelor-workload-broker.service.d \
      /etc/systemd/system/tigervncserver@:1.service.d \
      /opt/vaelor
    # The GPU memory pool setting (VD-161) is Vaelor's own file, so every
    # uninstall removes it - with or without --purge-data: it is a kernel
    # setting, not the owner's data, and nothing would be left to manage it.
    # The boot image is rebuilt so the next restart really does return the
    # pool to the kernel's own size, and the owner is told a restart is what
    # brings that back; this script restarts nothing. The path is
    # `gpu_memory_pool.CONFIG_PATH`, retyped because bash cannot import it;
    # tests/test_gpu_memory_pool_installer.py compares the two.
    gpu_memory_setting="/etc/modprobe.d/vaelor-gpu-memory.conf"
    if [[ -e "${gpu_memory_setting}" ]]; then
      rm -f "${gpu_memory_setting}"
      # Said truthfully either way: the pool returns to the kernel's own size
      # at the next restart only if the boot image was rebuilt without the
      # setting. When it was not, the restart may still apply the old size.
      if command -v update-initramfs >/dev/null 2>&1 \
        && update-initramfs -u >/dev/null 2>&1; then
        echo "Removed the GPU memory pool setting and rebuilt the boot image." \
          "Restart this machine to return the GPU memory pool to the kernel's" \
          "own size."
      else
        echo "Removed the GPU memory pool setting, but the boot image" \
          "could not be rebuilt, so a restart may still apply the old size." \
          "Finish with: update-initramfs -u - then restart this machine." >&2
      fi
    fi
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
    # --- BEGIN w3-perf (VD-147): the worker GPU sampler -----------------------
    # A machine that was a Vaelor worker may still run the non-root GPU sampler
    # the controller installed; the names are worker_telemetry_config's
    # SAMPLER_UNIT_NAME, SAMPLER_PATH and SAMPLER_STAGING_PATH, and
    # tests/test_uninstall_gpu_sampler.py holds this block to them. systemd is
    # asked only when the unit file exists, and the files go with rm -f, so a
    # machine that never ran one is a no-op. Before the daemon-reload below.
    gpu_sampler_unit="vaelor-gpu-sampler.service"
    if [[ -e "${systemd_dir}/${gpu_sampler_unit}" ]]; then
      systemctl disable --now "${gpu_sampler_unit}" >/dev/null 2>&1 || true
    fi
    rm -f "${systemd_dir}/${gpu_sampler_unit}" \
      /usr/local/lib/vaelor/vaelor-gpu-sampler.pyz \
      /usr/local/lib/vaelor/.vaelor-gpu-sampler.pyz.new
    # --- END w3-perf (VD-147): the worker GPU sampler -------------------------
    systemctl daemon-reload
    systemctl reset-failed
    if ((purge)); then
      # The browser desktop Vaelor commissions on request
      # (host_desktop.commission_vnc): the VNC display instance it enabled, the
      # managed account it created and that account's home. The account name is
      # `host_desktop_vnc.MANAGED_DESKTOP_USER`, retyped here because bash
      # cannot import it; tests/test_installer_provisioning.py compares the two.
      systemctl disable --now 'tigervncserver@:1.service' >/dev/null 2>&1 || true
      loginctl terminate-user vaelor-desktop >/dev/null 2>&1 || true
      id -u vaelor-desktop >/dev/null 2>&1 && userdel -r vaelor-desktop || true
      rm -rf /var/lib/vaelor /var/log/vaelor /run/vaelor /etc/vaelor
      for user in vaelor-research vaelor-vnc vaelor-secrets vaelor-workloads vaelor; do
        id -u "${user}" >/dev/null 2>&1 && userdel "${user}" || true
      done
      for group in vaelor-bridge vaelor-vnc vaelor-credentials vaelor-jobs vaelor; do
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
        # Stop the container engine before its packages and stores go: containerd
        # holds overlay mounts under /var/lib/containerd that cannot be removed
        # from under a running daemon, and a stopped engine cannot re-create the
        # bridges removed below.
        systemctl disable --now docker.socket docker.service containerd.service \
          >/dev/null 2>&1 || true
        # Docker, containerd, InfluxDB, amd-smi, novnc and the browser desktop's
        # packages are the rest of the stack the installer (or Vaelor, on the
        # operator's request) added. But "the installer did not add it" is a
        # promise the teardown can only keep if the installer said what it added:
        # its three "already installed; leaving it alone" branches (Docker,
        # InfluxDB, amd-smi) mean a package present here may be the operator's,
        # not Vaelor's. So --bare-os purges only what the marker records the
        # installer actually apt-installed. containerd is on that list when
        # Docker was: autoremove would remove it with docker.io but not purge it.
        # Shared OS tools (python3, curl, git, openssl) are never recorded, so
        # they are never purged. Without the marker (an appliance installed
        # before Vaelor recorded this) the full stack the installer can add is
        # purged and a warning says pre-existing packages cannot be told apart.
        if [[ -s "${installed_marker}" ]]; then
          mapfile -t os_packages < <(grep -vE '^[[:space:]]*$' "${installed_marker}" || true)
        else
          echo "No installed-by-vaelor marker at ${installed_marker}: this" \
            "appliance predates Vaelor recording what it added, so a" \
            "pre-existing Docker, InfluxDB or amd-smi cannot be told from one" \
            "Vaelor installed. Purging the full stack the installer can add." >&2
          os_packages=(
            docker.io docker-compose-v2 containerd influxdb amd-smi
            novnc tigervnc-standalone-server gnome-session-flashback
          )
        fi
        ((${#os_packages[@]})) && purge_os_packages "${os_packages[@]}"
        apt-get autoremove --purge -y >/dev/null 2>&1 || true
        apt-get update >/dev/null 2>&1 || true
        # apt purge leaves the daemons' own data trees behind, so a bare-OS
        # teardown removes them explicitly. Docker 29 keeps its images in
        # containerd's store, so /var/lib/containerd is what actually holds the
        # multi-GB serving images and any image a deploy or the operator pulled
        # - the first cold teardown removed /var/lib/docker alone and left 44 GB
        # of images behind, and the "cold" reinstall found every image present.
        # /var/lib/influxdb is the metrics TSM store and /etc/influxdb the
        # configuration a postrm that failed would have left.
        for netns in /run/docker/netns/*; do
          [[ -e "${netns}" ]] || continue
          umount "${netns}" >/dev/null 2>&1 || true
        done
        rm -rf \
          /var/lib/docker /var/lib/containerd /etc/docker \
          /run/docker /run/containerd /opt/containerd \
          /var/lib/influxdb /etc/influxdb
        # Docker's kernel bridges and its group outlive the package purge too
        # (docker0, docker_gwbridge and one br-* per Compose network were still
        # up after the first cold teardown, cleared only by a reboot). Only
        # Docker's own link names are touched: docker0, docker_gwbridge, and a
        # Compose bridge, which docker names `br-` followed by exactly twelve hex
        # digits (the network id's short form). An operator's own bridge named
        # br-lan, br-ex or br-int does not match that shape and is not Vaelor's
        # to delete; a veth belonging to anything else is not either.
        while read -r link; do
          if [[ "${link}" == docker0 || "${link}" == docker_gwbridge ||
            "${link}" =~ ^br-[0-9a-f]{12}$ ]]; then
            ip link delete "${link}" >/dev/null 2>&1 || true
          fi
        done < <(ip -o link show 2>/dev/null | awk -F': ' '{print $2}' | cut -d@ -f1 || true)
        getent group docker >/dev/null && groupdel docker || true
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
        echo "Vaelor and the OS stack it added (Docker and containerd with every container image, InfluxDB, ROCm, amd-smi, novnc, the browser-desktop packages) were removed; the machine is back to its pre-Vaelor state."
      else
        echo "Vaelor services, software, configuration, credentials, data, and the container images it pulled were removed."
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
