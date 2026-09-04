#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wheel=""
wheelhouse=""
unattended=0
migrate=0
# "we handle the rest": Docker is provisioned by default, unattended included.
# Only an explicit --without-docker declines it (#207).
with_docker="yes"
skip_system_packages=0
with_hat=0
accept_sunfounder_terms=0
# The physical Pironman 5 board: base/max/mini/pro-max are distinct boards that
# SunFounder's installer cannot auto-detect (which is why it *takes* --variant),
# and Vaelor reads the model it writes. Never silently assume "base" (#210).
hat_variant=""
# Set to 1 only when the SunFounder HAT runtime is installed *this run*, which is
# the sole thing that authorises the single reboot at the very end (#207).
hat_installed=0
migration_applied=0
migration_complete=0
legacy_active=()
legacy_units=(
  pironman-appliance-recovery.service
  pironman-credential-broker.service
  pironman-host-desktop-broker.service
  pironman-system-update.service
  pironman-vnc-gateway.service
  pironman-vnc-tls-proxy.service
  pironman-workload-executor.service
  pironman5.service
)
vaelor_units=(
  vaelor-credential-broker.service
  vaelor-system-update.service
  vaelor-appliance-upgrade.service
  vaelor-host-desktop.service
  vaelor-hardware-bridge.service
  vaelor-appliance-recovery.service
  vaelor-application-research.service
  vaelor-workload-broker.service
  vaelor-workload-executor.service
  vaelor-control-plane.service
  vaelor-vnc-gateway.service
  vaelor-vnc-tls-proxy.service
)

restore_legacy_after_failed_migration() {
  local status=$?
  if [[ "${migrate}" -eq 1 && "${migration_complete}" -eq 0 ]]; then
    systemctl disable --now "${vaelor_units[@]}" >/dev/null 2>&1 || true
    if [[ "${migration_applied}" -eq 1 ]]; then
      /opt/vaelor/venv/bin/vaelor-migrate --rollback \
        --confirm rollback-vaelor-migration >/dev/null 2>&1 || true
    fi
    if ((${#legacy_active[@]})); then
      systemctl start "${legacy_active[@]}" >/dev/null 2>&1 || true
    fi
  fi
  exit "${status}"
}
trap restore_legacy_after_failed_migration ERR

usage() {
  cat <<'EOF'
Usage: install-vaelor.sh --wheel FILE [options]
  --wheel FILE         Versioned Vaelor wheel to install
  --wheelhouse DIR     Offline dependency wheel directory
  --unattended         Do not prompt; provision the full stack non-interactively
  --with-docker        Install Docker (the default; kept for compatibility)
  --without-docker     Explicitly decline Docker; container workloads stay off
  --skip-system-packages
                       Native-package integration only; required OS packages
                       (including InfluxDB 1.x and Docker) must already be
                       installed
  --migrate            Apply the reviewed Pironman-to-Vaelor state migration
  --with-hat           On a Raspberry Pi, offer the SunFounder Pironman HAT
                       runtime (fan/OLED/RGB). Unattended, this flag is what
                       lets the offer proceed without a prompt.
  --hat-variant V      Which Pironman 5 board: base | max | mini | pro-max.
                       Required to install the HAT (the boards are distinct and
                       cannot be auto-detected); the installer refuses to guess.
  --accept-sunfounder-terms
                       Accept SunFounder's terms (GPL-2.0, and what its
                       install.sh does - see the offer text) up front, so the
                       HAT offer needs no interactive consent.

"User installs stock Ubuntu, we handle the rest": this installer provisions
InfluxDB 1.x for telemetry retention and Docker for container workloads by
default. The one thing it does not provision for you is the SunFounder Pironman
HAT runtime under /opt/pironman5 - on a Raspberry Pi it offers it, with consent
and terms; see the offer text and RELEASE.md.
EOF
}

while (($#)); do
  case "$1" in
    --wheel) wheel="${2:-}"; shift 2 ;;
    --wheelhouse) wheelhouse="${2:-}"; shift 2 ;;
    --unattended) unattended=1; shift ;;
    --migrate) migrate=1; shift ;;
    --with-docker) with_docker="yes"; shift ;;
    --without-docker) with_docker="no"; shift ;;
    --skip-system-packages) skip_system_packages=1; shift ;;
    --with-hat) with_hat=1; shift ;;
    --hat-variant) hat_variant="${2:-}"; shift 2 ;;
    --accept-sunfounder-terms) accept_sunfounder_terms=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${EUID}" -eq 0 ]] || {
  echo "Run this installer as root." >&2
  exit 1
}
[[ -f "${wheel}" ]] || {
  echo "Provide a readable Vaelor wheel with --wheel." >&2
  exit 1
}
[[ -r /etc/os-release ]] || {
  echo "Vaelor requires a Linux distribution with /etc/os-release." >&2
  exit 1
}

# shellcheck disable=SC1091
source /etc/os-release
case "${ID:-}" in
  ubuntu|debian|raspbian|kali) ;;
  *)
    echo "This installer currently supports Debian-family hosts only." >&2
    exit 1
    ;;
esac
case "$(uname -m)" in
  aarch64|arm64|x86_64|amd64) ;;
  *)
    echo "Vaelor packages currently support arm64 and amd64 only." >&2
    exit 1
    ;;
esac

# The SunFounder Pironman HAT runtime: the one thing "we handle the rest" hands
# back to the owner, as an offer rather than a silent assumption (#207).
#
# Vaelor's hardware bridge drives the fan, OLED and RGB by importing `pironman5`
# from /opt/pironman5/venv and reading /opt/pironman5/config.json (see
# vaelor/hardware_bridge.py). Only SunFounder's own installer provisions that
# runtime, and a *clean pinned non-interactive* install through it is not
# achievable, which is why this offers rather than runs it:
#   * Its install.sh hardcodes pm_auto@v2, pm_dashboard@v2 and sf_rpi_status@main
#     - floating branches, with no flag to pin them - and sources its installer
#     framework cache-busted (`?$(date +%s)`) from a second repo's main branch.
#     Pinning the pironman5 repo itself leaves those three floating (#130).
#   * It reboots the Pi at the end. Running it *inside* this installer could
#     reboot mid-provision, before Vaelor finishes.
#   * It enables a competing pironman5.service + pm_dashboard web control plane
#     that Vaelor stubs out (hardware_bridge sets PMDashboard=None so the legacy
#     package cannot compose a second control plane). We neutralise that below
#     whenever the runtime is present, however it got there.
# So the offer prints the command pinned as far as the installer allows (a
# specific pironman5 tag) with SunFounder's terms stated, and the owner runs it
# as its own step. The bridge is optional at runtime: it serves host power on
# every machine and reports the enclosure absent when this runtime is missing,
# so declining still succeeds.
#
# Pinned to a specific SunFounder release tag rather than a branch (#130). The
# residual floating sub-packages above are SunFounder's, inside their installer,
# and are stated in the offer rather than hidden.
SUNFOUNDER_PIRONMAN_TAG="v1.3.17"

host_is_raspberry_pi() {
  [[ -e /boot/firmware/config.txt || -e /boot/config.txt ]] && return 0
  grep -qi 'raspberry pi' /proc/device-tree/model 2>/dev/null
}

# Whenever the SunFounder runtime is present - whether the owner accepted the
# offer, or it was already on the appliance - keep only the hardware daemon our
# bridge imports and shut down the web control plane that would fight Vaelor's.
neutralize_sunfounder_control_plane() {
  [[ -d /opt/pironman5 ]] || return 0
  if systemctl list-unit-files pironman5.service >/dev/null 2>&1 ||
    [[ -e /etc/systemd/system/pironman5.service ]]; then
    # Disable *and* mask: their pironman5.service runs pm_dashboard's own web
    # listener. Vaelor imports the pironman5 library directly in
    # vaelor-hardware-bridge.service, so removing their service removes the
    # competing control plane without touching the daemon the bridge needs.
    systemctl disable --now pironman5.service >/dev/null 2>&1 || true
    # `systemctl mask` refuses to mask a unit whose real file already lives in
    # /etc/systemd/system - it will not overwrite an admin unit - and SunFounder
    # installs pironman5.service exactly there. On the first live HAT run the
    # mask was swallowed by `|| true` and the unit was left merely *disabled*,
    # which a later action can re-enable; mask is the stronger guarantee VD-098
    # intends (#210). So remove their unit file first, reload, THEN mask (a
    # symlink to /dev/null), and verify the mask actually stuck.
    rm -f /etc/systemd/system/pironman5.service
    systemctl daemon-reload
    systemctl mask pironman5.service >/dev/null 2>&1 || true
    if [[ "$(systemctl is-enabled pironman5.service 2>/dev/null || true)" == "masked" ]]; then
      echo "SunFounder's pironman5.service is masked; Vaelor owns the only control plane."
    else
      echo "Warning: SunFounder's pironman5.service could not be masked; it is at" >&2
      echo "least disabled, but re-check it by hand (#210)." >&2
    fi
  fi
  if [[ -f /opt/pironman5/config.json ]]; then
    echo "Pironman HAT runtime present; the hardware bridge reads /opt/pironman5/config.json."
  fi
}

hat_variant_is_valid() {
  case "$1" in
    base | max | mini | pro-max) return 0 ;;
    *) return 1 ;;
  esac
}

# Install SunFounder's HAT runtime, pinned, without letting it reboot mid-run.
#
# Their install.sh ends with `installer_prompt_reboot`, which reads `< /dev/tty`
# (not stdin), so closing stdin cannot answer it and would leave the read
# blocking or spinning. We give the child its own pseudo-terminal via `script`
# and feed it "n", so the enclosure overlays are written but the Pi is NOT
# rebooted here - Vaelor's own single reboot at the very end is the only one.
install_pironman_hat_pinned() {
  if (
    set -e
    apt-get update
    command -v git >/dev/null 2>&1 || apt-get install -y git
    command -v script >/dev/null 2>&1 || apt-get install -y bsdutils
    rm -rf /tmp/pironman5-hat
    git clone https://github.com/sunfounder/pironman5 /tmp/pironman5-hat
    git -C /tmp/pironman5-hat checkout "${SUNFOUNDER_PIRONMAN_TAG}"
    # Pin the *installed* code, not just this script. install.sh re-clones
    # pironman5 into $HOME and, left alone, tracks the floating 1.3.x branch - it
    # self-reported 1.3.18 on the first Pi run while we "pinned" v1.3.17, because
    # the checkout above only pins which install.sh runs, not what it installs.
    # install.sh's PIRONMAN5_BRANCH override makes that re-clone
    # `git clone -b ${SUNFOUNDER_PIRONMAN_TAG}`, so the code that runs is the tag.
    # (pm_auto/pm_dashboard/sf_rpi_status stay floating - hardcoded with no
    # override - which the accepted terms state.)
    #
    # Answer SunFounder's mid-install reboot prompt with 'n' through the pty,
    # BOUNDED. An unbounded feed of 'n' buffered ~5.7 GB into `script` over the
    # minutes install.sh runs without reading it, and OOM-killed a 7.7 GB Pi
    # (#209): an unbounded producer with an absent consumer. A few hundred lines
    # queue in the pty until the one reboot prompt reads a single 'n', then the
    # feed ends. pipefail off so the feed dying by SIGPIPE when `script` exits
    # first is not read as a failed install; `script`'s own status is what counts.
    set +o pipefail
    { for _ in $(seq 200); do printf 'n\n'; done; } | script -qec \
      "PIRONMAN5_BRANCH=${SUNFOUNDER_PIRONMAN_TAG} bash /tmp/pironman5-hat/install.sh --variant ${hat_variant} </dev/null" \
      /dev/null
  ); then
    hat_installed=1
    echo "SunFounder Pironman HAT runtime installed (${hat_variant}, pinned ${SUNFOUNDER_PIRONMAN_TAG})."
    echo "No reboot yet - Vaelor finishes first, then reboots once at the end."
  else
    echo "SunFounder HAT install did not complete; the enclosure will report as" >&2
    echo "absent and every other Vaelor feature works. Continuing." >&2
  fi
}

offer_pironman_hat() {
  host_is_raspberry_pi || return 0  # x86/Z2 have no HAT: offer nothing.
  if [[ -f /opt/pironman5/config.json && -d /opt/pironman5/venv ]]; then
    echo "Pironman HAT runtime found under /opt/pironman5; fan/OLED/RGB will use it."
    return 0
  fi
  local proceed=0
  if ((with_hat && accept_sunfounder_terms)); then
    proceed=1
  elif ((unattended)); then
    echo "NOTE: the SunFounder Pironman HAT runtime is not installed; fan/OLED/RGB"
    echo "control needs it. Re-run with --with-hat --accept-sunfounder-terms"
    echo "--hat-variant <base|max|mini|pro-max> to install it, or install it"
    echo "yourself. Vaelor handles everything else and reports the enclosure as"
    echo "absent without it."
    return 0
  fi
  # Resolve the board model *before* showing the terms, so the terms name it and
  # so we never silently assume 'base' - SunFounder's boards are physically
  # distinct and its installer cannot auto-detect them, so a wrong guess mislabels
  # a Max as a base board (#210). Interactive: ask. Unattended: --hat-variant is
  # required, and an empty/invalid value fails clearly rather than defaulting.
  if [[ -z "${hat_variant}" && "${unattended}" -eq 0 ]]; then
    printf "Which Pironman 5 board? [base/max/mini/pro-max] "
    read -r hat_variant </dev/tty 2>/dev/null || hat_variant=""
  fi
  if ! hat_variant_is_valid "${hat_variant}"; then
    echo "The Pironman HAT needs its board model. Pass --hat-variant with one of" >&2
    echo "base, max, mini, pro-max. Refusing to guess 'base' - that would" >&2
    echo "mislabel a different board (#210). Skipping the HAT runtime." >&2
    return 0
  fi
  # The terms have to be honest, so state what accepting will actually do.
  echo "----------------------------------------------------------------------"
  echo "Optional: SunFounder Pironman HAT runtime (fan, OLED, RGB control)"
  echo "This is SunFounder's software, not Vaelor's. Accepting will:"
  echo "  * Install it for your Pironman 5 board: ${hat_variant}."
  echo "  * Install SunFounder's code, licensed GPL-2.0, from github.com/sunfounder,"
  echo "    pinned to tag ${SUNFOUNDER_PIRONMAN_TAG}. Its installer also pulls"
  echo "    several SunFounder packages (pm_auto, pm_dashboard, sf_rpi_status)"
  echo "    from floating branches, which are SunFounder's and not pinnable here."
  echo "  * Create a 'pironman5' system user with passwordless sudo for"
  echo "    shutdown/reboot/systemctl, and write Raspberry Pi device-tree overlays."
  echo "  * Have Vaelor mask SunFounder's own web dashboard so only Vaelor's"
  echo "    control plane runs; the hardware daemon stays for Vaelor's bridge."
  echo "  * Require one reboot at the very end (after Vaelor finishes) to load the"
  echo "    overlays. Declining is fine - the enclosure just reports as absent."
  if ((proceed == 0)); then
    printf "Install the SunFounder HAT runtime now, accepting these terms? [y/N] "
    local answer=""
    read -r answer </dev/tty 2>/dev/null || answer=""
    # Accept "y" and the full word "yes" (any case). A naive user typing "yes"
    # to an opt-in must not be read as a decline (review UX #207). Anything else
    # still declines - default-no is correct for this opt-in.
    [[ "${answer}" =~ ^[Yy]([Ee][Ss])?$ ]] && proceed=1
  fi
  if ((proceed)); then
    install_pironman_hat_pinned
  else
    echo "Skipping the HAT runtime. Re-run with --with-hat to be offered it again."
  fi
}

export DEBIAN_FRONTEND=noninteractive
wait_for_packages() {
  local deadline=$((SECONDS + 900))
  local locks=(
    /var/lib/dpkg/lock-frontend
    /var/lib/dpkg/lock
    /var/lib/apt/lists/lock
    /var/cache/apt/archives/lock
  )
  while command -v fuser >/dev/null 2>&1 &&
    fuser "${locks[@]}" >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
      echo "Ubuntu's package service stayed busy for 15 minutes. Try again later." >&2
      exit 1
    fi
    echo "Waiting for Ubuntu's background package service to finish..."
    sleep 5
  done
  dpkg --configure -a
}
if ((skip_system_packages)); then
  for command in python3 openssl systemctl systemd-creds curl; do
    command -v "${command}" >/dev/null || {
      echo "Required native-package dependency is missing: ${command}" >&2
      exit 1
    }
  done
  [[ -d /usr/share/novnc ]] || {
    echo "The native package requires the novnc OS package." >&2
    exit 1
  }
else
  wait_for_packages
  apt-get update
  apt-get install -y python3 python3-venv python3-pip openssl novnc curl
fi

# The HAT offer runs here, after base packages: git and a pseudo-terminal are
# available and the dpkg lock has been waited out. On accept it installs the
# runtime and sets hat_installed, which the single final reboot is gated on. It
# creates /opt/pironman5, which neutralize_sunfounder_control_plane (later, in
# the same run, with no exit between) then takes over from SunFounder's dashboard.
offer_pironman_hat

# A freshly installed docker.io can report the daemon `active` while containerd
# has not finished initialising its content store, so the FIRST `docker pull` -
# the one a Custom Application deploy makes - fails with
# `mkdir /var/lib/containerd/.../ingest/...: no such file or directory`, and the
# box only recovers after a manual `systemctl restart containerd docker`. A
# turnkey appliance must never need that hand repair, so once the services are
# enabled we actively PROVE the runtime is ready before the install finishes:
# both daemons are active, `docker info` reports a real storage driver, and a
# throwaway pull of a tiny image actually forces the content store to
# initialise. If it is not ready we perform, once, the very restart the manual
# fix performs, then keep probing with a short backoff to a bounded deadline.
# Idempotent: on a re-run the pull is cached/instant and the removal is a no-op.
ensure_docker_ready() {
  local probe_image="hello-world"
  local deadline=$((SECONDS + 180))
  local restarted=0
  while ((SECONDS < deadline)); do
    if systemctl is-active --quiet docker.service &&
      { ! systemctl list-unit-files containerd.service >/dev/null 2>&1 ||
        systemctl is-active --quiet containerd.service; } &&
      [[ -n "$(docker info --format '{{.Driver}}' 2>/dev/null)" ]] &&
      docker pull "${probe_image}" >/dev/null 2>&1; then
      # The probe image is a throwaway - remove it so the box is left clean.
      docker rmi "${probe_image}" >/dev/null 2>&1 || true
      echo "Docker and containerd are ready: storage driver active and the" \
        "content store initialises a test pull."
      return 0
    fi
    # The uninitialised-content-store failure is exactly what a containerd+docker
    # restart repairs by hand; do that once, then keep probing until the deadline.
    if ((restarted == 0)); then
      echo "Container runtime not ready yet; restarting containerd and docker to" \
        "initialise the content store..."
      systemctl restart containerd.service docker.service >/dev/null 2>&1 || true
      restarted=1
    fi
    sleep 3
  done
  echo "Docker/containerd did not become ready within 180s: the container" >&2
  echo "runtime could not initialise its content store, so the first Custom" >&2
  echo "Application 'docker pull' would fail with a missing /var/lib/containerd" >&2
  echo "ingest directory - the exact defect this gate exists to prevent." >&2
  if ((skip_system_packages)); then
    # Native-package/offline image: the image builder owns the runtime, and this
    # box may have no registry network. Degrade honestly like the other
    # skip_system_packages deferrals rather than failing a pre-built image.
    echo "Native-package install: leaving the container runtime to the image" >&2
    echo "and continuing." >&2
    return 0
  fi
  exit 1
}

# Docker is provisioned unconditionally unless the owner passes --without-docker.
# Container workloads (the Apps and AI surfaces, and system-web-research) do not
# run without it, so leaving it out silently is the "we handle the rest" gap
# #207 closes; the opt-out stays for an owner who runs workloads elsewhere.
if [[ "${with_docker}" != "no" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    if ((skip_system_packages)); then
      echo "Native-package install expects Docker to be present already." >&2
      echo "Install docker.io and docker-compose-v2, or re-run without" >&2
      echo "--skip-system-packages." >&2
      exit 1
    fi
    apt-get install -y docker.io docker-compose-v2
  fi
  # containerd is the content/runtime store docker layers on, so it must be
  # enabled and started BEFORE docker. docker.io pulls it in as a dependency but
  # does not guarantee its content store is initialised; enabling an
  # already-running unit is a no-op, so this stays idempotent on a re-run.
  if systemctl list-unit-files containerd.service >/dev/null 2>&1; then
    systemctl enable --now containerd.service
  fi
  systemctl enable --now docker.service
  # Prove the runtime is fully initialised before anything pulls an image.
  ensure_docker_ready
fi

# --- InfluxDB, pinned to the 1.x line -----------------------------------------
#
# `vaelor/database.py` speaks the InfluxDB **1.x** API: get_list_database(),
# create_database(), switch_database(), and InfluxQL ("SHOW DATABASES",
# "SELECT ... FROM history"). InfluxDB 2.x/3.x removed databases entirely
# (buckets + Flux), so an accidental 2.x here would silently reinstate the exact
# "telemetry is not retained" defect VD-095 just fixed - the database would never
# be created and every row would be dropped, with a 200 from the health endpoint
# (LESSONS 4). So this must be 1.x, and it is pinned so an install is
# reproducible rather than "whatever the tag points at today" (#130).
#
# Source and pin: one specific .deb for 1.12.4-1 - the current maintained 1.x -
# fetched from InfluxData's package pool, whose URL names the version and whose
# bytes are verified against a recorded sha256 (#130). Two deliberate choices,
# both for the reasons the amd-smi block states:
#   * A pinned .deb, not `apt install influxdb`. Ubuntu 24.04+ main repos do not
#     carry the 1.x line at a known version, and the EOL 1.8.10 is no longer in
#     the release archive. `apt install influxdb` resolves to whatever a repo
#     happens to publish, which is neither pinned nor guaranteed 1.x. The .deb
#     URL plus the sha256 *is* the pin.
#   * We do NOT add an APT source. Fetching one sha-verified .deb from
#     repos.influxdata.com's pool is a single `curl` + `apt-get install
#     ./file.deb`; nothing is written to /etc/apt/sources.list*, so the
#     appliance's standing trust on every future `apt upgrade` is unchanged.
#     Adding a sources.list entry would be an owner's decision; this is not that.
INFLUXDB_VERSION="1.12.4-1"
influxdb_verify() {
  # Prove the *database* API answers, not merely that a server is up - the same
  # distinction database.py draws, and the one VD-095 turned on (LESSONS 4).
  local deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    if command -v influx >/dev/null 2>&1 &&
      influx -host 127.0.0.1 -port 8086 -execute 'SHOW DATABASES' \
        >/dev/null 2>&1; then
      echo "InfluxDB ${INFLUXDB_VERSION} answers SHOW DATABASES on localhost:8086."
      return 0
    fi
    sleep 2
  done
  echo "InfluxDB did not answer 'SHOW DATABASES' on localhost:8086 within 90s." >&2
  echo "Telemetry retention (VD-095) depends on the 1.x database API; refusing" >&2
  echo "to finish an install that cannot store history." >&2
  exit 1
}
install_influxdb() {
  if command -v influxd >/dev/null 2>&1; then
    # A pre-existing influxd could be 2.x, which also provides the `influxd`
    # binary but has no databases (buckets + Flux) - accepting it silently is
    # how VD-095 comes back (LESSONS 4). Confirm the 1.x line before trusting it.
    # `|| true` inside the substitution: under `set -o pipefail`, `head` closing
    # the pipe early makes grep exit non-zero even on a match, and an empty match
    # would fail too - either would trip `set -e` on the assignment. A missing or
    # unparseable version then falls through to the not-1.x refusal below, which
    # is the safe direction.
    local existing_version
    existing_version="$(influxd version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true)"
    if [[ "${existing_version}" != 1.* ]]; then
      echo "An InfluxDB daemon is already installed but is not 1.x" >&2
      echo "(reported: '${existing_version:-unknown}'). Vaelor's database.py needs" >&2
      echo "the 1.x database API; refusing rather than silently accepting 2.x." >&2
      exit 1
    fi
    echo "influxd ${existing_version} (1.x) is already installed; leaving it in place."
  elif ((skip_system_packages)); then
    echo "Native-package install expects InfluxDB 1.x to be present already." >&2
    echo "Install influxdb ${INFLUXDB_VERSION} (the 1.x line, not the 2.x" >&2
    echo "package), or re-run without --skip-system-packages." >&2
    exit 1
  else
    local arch sha url tmp
    case "$(uname -m)" in
      aarch64 | arm64) arch="arm64" ;;
      x86_64 | amd64) arch="amd64" ;;
      *)
        echo "No pinned InfluxDB 1.x .deb for this architecture." >&2
        exit 1
        ;;
    esac
    # sha256 of influxdb_1.12.4-1_<arch>.deb from InfluxData's package pool,
    # checked against the signed apt index, recorded so a swapped or truncated
    # download fails loudly instead of installing something else.
    case "${arch}" in
      amd64) sha="39362452cfd9584603e96c185d88b561fa45bc7078e5dbef3a3311b81c45cb5b" ;;
      arm64) sha="f64f50fdfb9c36e63f8628186ec273b55e2f8e065c46433ac59e35982d4ec99e" ;;
    esac
    url="https://repos.influxdata.com/debian/packages/influxdb_${INFLUXDB_VERSION}_${arch}.deb"
    tmp="$(mktemp --suffix=.deb)"
    echo "Fetching pinned InfluxDB ${INFLUXDB_VERSION} (${arch}) from InfluxData's pool."
    curl --fail --location --silent --show-error --max-time 300 -o "${tmp}" "${url}"
    echo "${sha}  ${tmp}" | sha256sum --check --status || {
      echo "InfluxDB ${INFLUXDB_VERSION} .deb failed its sha256 pin (#130); refusing." >&2
      rm -f "${tmp}"
      exit 1
    }
    apt-get install -y "${tmp}"
    rm -f "${tmp}"
  fi
  systemctl enable --now influxdb.service
  influxdb_verify
}
install_influxdb

for group in vaelor vaelor-jobs vaelor-credentials vaelor-vnc; do
  getent group "${group}" >/dev/null || groupadd --system "${group}"
done
create_user() {
  local name="$1" group="$2"
  id -u "${name}" >/dev/null 2>&1 || useradd \
    --system --gid "${group}" --home-dir /nonexistent \
    --shell /usr/sbin/nologin "${name}"
}
create_user vaelor vaelor
create_user vaelor-workloads vaelor-jobs
create_user vaelor-secrets vaelor-credentials
create_user vaelor-vnc vaelor-vnc
create_user vaelor-research vaelor
usermod -aG vaelor-jobs,vaelor-credentials,vaelor-vnc vaelor
usermod -aG vaelor vaelor-workloads
usermod -aG vaelor vaelor-secrets
usermod -aG vaelor vaelor-vnc
usermod -aG vaelor vaelor-research
getent group docker >/dev/null && usermod -aG docker vaelor-workloads

install -d -m 0755 -o root -g root /opt/vaelor
install -d -m 0755 -o root -g root /etc/vaelor
# The staged application workflow used to be gated by VAELOR_APPLICATION_*_ENABLED
# flags written here. Those flags are gone: the workflow now auto-enables on the
# presence of a working Assistant model (VD-109), so no application.env is
# written. The services still reference it as an optional EnvironmentFile
# (EnvironmentFile=-...), which stays non-fatal when the file is absent.
install -d -m 0750 -o vaelor -g vaelor /var/lib/vaelor /var/log/vaelor
install -d -m 2770 -o vaelor-workloads -g vaelor-jobs \
  /var/lib/vaelor/workloads /var/lib/vaelor/models \
  /var/lib/vaelor/backups /var/lib/vaelor/jobs
# Guarded web research (SearXNG) is brought up by ENABLING web research, not at
# install time (#212, auto-start-on-enable). The
# install prepares only the owned mount point here; enabling the service
# (vaelor/web_research.py, /api/v2 web-research routes -> the workload executor)
# deploys AND starts the one owned compose project - loopback-only,
# digest-pinned, cap_drop, 512 MB cap - so a granted agent then fetches
# end-to-end with no separate manual container deploy. The compose carries
# restart:unless-stopped and the executor re-ensures it on start-up, so once
# enabled it stays up automatically. The endpoint env below (vaelor-control-
# plane / -workload-executor units) points at the loopback address the manager
# binds, so no restart-config change is needed. Until web research is enabled,
# agent_task_runner surfaces "web access is granted but the search service is
# unavailable" (LESSONS #4/#11) rather than an empty answer that reads as "no
# access".
install -d -m 2770 -o vaelor-workloads -g vaelor-jobs \
  /var/lib/vaelor/workloads/system-web-research
install -d -m 2770 -o vaelor -g vaelor-jobs /var/lib/vaelor/applications
for application_db in /var/lib/vaelor/applications/applications.sqlite3*; do
  [[ -e "${application_db}" ]] || continue
  chown vaelor:vaelor-jobs "${application_db}"
  chmod 0660 "${application_db}"
done
install -d -m 2770 -o vaelor -g vaelor-jobs /var/lib/vaelor/assistant
install -d -m 0770 -o vaelor -g vaelor \
  /var/lib/vaelor/cluster /var/lib/vaelor/kvm
install -d -m 0770 -o vaelor-vnc -g vaelor-vnc /var/lib/vaelor/vnc
install -d -m 0700 -o vaelor-secrets -g vaelor-credentials \
  /var/lib/vaelor/credentials
cat >/etc/tmpfiles.d/vaelor.conf <<'EOF'
d /run/vaelor 0770 root vaelor -
EOF
systemd-tmpfiles --create /etc/tmpfiles.d/vaelor.conf

python3 -m venv /opt/vaelor/venv
constraint_args=()
constraint_file="${script_dir}/requirements-release.txt"
if [[ ! -f "${constraint_file}" ]]; then
  constraint_file="${script_dir}/../requirements-release.txt"
fi
if [[ -f "${constraint_file}" ]]; then
  constraint_args=(--constraint "${constraint_file}")
fi
pip_args=(install "${constraint_args[@]}" --upgrade "${wheel}")
reinstall_args=(install --no-deps --force-reinstall "${wheel}")
if [[ -n "${wheelhouse}" ]]; then
  [[ -d "${wheelhouse}" ]] || {
    echo "The supplied wheelhouse directory is not readable." >&2
    exit 1
  }
  pip_args=(
    install --no-index --find-links "${wheelhouse}"
    "${constraint_args[@]}" --upgrade "${wheel}"
  )
  reinstall_args=(
    install --no-index --find-links "${wheelhouse}"
    --no-deps --force-reinstall "${wheel}"
  )
fi
/opt/vaelor/venv/bin/python -m pip "${pip_args[@]}"
/opt/vaelor/venv/bin/python -m pip "${reinstall_args[@]}"

# Retain the wheel that produced this install so the appliance upgrade broker
# (vaelor/appliance_upgrade.py) can reinstall it on a failed upgrade and prove
# the previous version came back. Non-fatal: a missing snapshot only means the
# broker reports rollback-unavailable, never that the install failed.
install -d -m 2770 -o vaelor-workloads -g vaelor-jobs /var/lib/vaelor/upgrade
if [[ -f "${wheel}" ]] &&
  current_version="$(/opt/vaelor/venv/bin/python -c 'import vaelor; print(vaelor.__version__)' 2>/dev/null)"; then
  install -d -m 2770 -o vaelor-workloads -g vaelor-jobs /var/lib/vaelor/upgrade/wheels
  retained="/var/lib/vaelor/upgrade/wheels/$(basename "${wheel}")"
  if install -m 0640 -o vaelor-workloads -g vaelor-jobs "${wheel}" "${retained}"; then
    printf '{"version":"%s","wheel":"%s"}\n' "${current_version}" "${retained}" \
      >/var/lib/vaelor/upgrade/current-wheel.json
    chown vaelor-workloads:vaelor-jobs /var/lib/vaelor/upgrade/current-wheel.json
    chmod 0640 /var/lib/vaelor/upgrade/current-wheel.json
  fi
fi

if [[ "${migrate}" -eq 1 ]]; then
  for unit in "${legacy_units[@]}"; do
    if systemctl is-active --quiet "${unit}"; then
      legacy_active+=("${unit}")
    fi
  done
  if ((${#legacy_active[@]})); then
    systemctl stop "${legacy_active[@]}"
  fi
  /opt/vaelor/venv/bin/vaelor-migrate --apply \
    --confirm migrate-control-plane-to-vaelor
  migration_applied=1
else
  migration="$(
    /opt/vaelor/venv/bin/vaelor-migrate
  )"
  if grep -q '"source_exists": true' <<<"${migration}"; then
    echo "Legacy Pironman control-plane state was detected." >&2
    echo "Re-run with --migrate after reviewing vaelor-migrate output." >&2
    exit 1
  fi
fi

chown -R vaelor-workloads:vaelor-jobs \
  /var/lib/vaelor/workloads /var/lib/vaelor/models \
  /var/lib/vaelor/backups /var/lib/vaelor/jobs
install -d -m 2770 -o vaelor-workloads -g vaelor-jobs \
  /var/lib/vaelor/backups/workloads \
  /var/lib/vaelor/backups/cluster
find /var/lib/vaelor/workloads -type d -exec chmod 2770 {} +
find /var/lib/vaelor/workloads -type f \
  \( -name 'compose.yaml' -o -name 'compose.yml' -o -name 'docker-compose.yml' \) \
  -exec chmod 0660 {} +
chown -R vaelor:vaelor-jobs /var/lib/vaelor/assistant
chown -R vaelor:vaelor /var/lib/vaelor/cluster /var/lib/vaelor/kvm
for shared_db in /var/lib/vaelor/assistant/custom-agents.sqlite3*; do
  [[ -e "${shared_db}" ]] || continue
  chown vaelor:vaelor-jobs "${shared_db}"
  chmod 0660 "${shared_db}"
done
chown -R vaelor-vnc:vaelor-vnc /var/lib/vaelor/vnc
chown -R vaelor-secrets:vaelor-credentials /var/lib/vaelor/credentials
for state_file in \
  /var/lib/vaelor/security.sqlite3 \
  /var/lib/vaelor/totp.key \
  /var/lib/vaelor/device-identity.json \
  /var/lib/vaelor/system-update-state.json; do
  [[ ! -e "${state_file}" ]] || chown vaelor:vaelor "${state_file}"
done

install -d -m 0700 -o root -g root /etc/vaelor/credentials
if [[ ! -f /etc/vaelor/credentials/master-key.cred ]]; then
  command -v systemd-creds >/dev/null || {
    echo "systemd-creds is required for the encrypted credential broker." >&2
    exit 1
  }
  head -c 32 /dev/urandom | systemd-creds encrypt \
    --with-key=host --name=master.key - \
    /etc/vaelor/credentials/master-key.cred
  chmod 0600 /etc/vaelor/credentials/master-key.cred
fi

install -d -m 0750 -o root -g vaelor /opt/vaelor/tls
if [[ ! -f /opt/vaelor/tls/vaelor.crt ]]; then
  openssl req -x509 -newkey rsa:3072 -sha256 -nodes -days 825 \
    -subj "/CN=$(hostname)" \
    -keyout /opt/vaelor/tls/vaelor.key \
    -out /opt/vaelor/tls/vaelor.crt
  chown root:vaelor /opt/vaelor/tls/vaelor.key /opt/vaelor/tls/vaelor.crt
  chmod 0640 /opt/vaelor/tls/vaelor.key
  chmod 0644 /opt/vaelor/tls/vaelor.crt
fi

for unit in "${script_dir}"/systemd/vaelor-*.service; do
  install -m 0644 "${unit}" "/etc/systemd/system/$(basename "${unit}")"
done
install -d -m 0755 /etc/systemd/system/tigervncserver@:1.service.d
cat >/etc/systemd/system/tigervncserver@:1.service.d/vaelor-session-cleanup.conf <<'EOF'
[Service]
# GNOME moves desktop processes into the account's user manager. Ensure they
# cannot survive a stopped VNC display and collide with the next session.
#
# **This list and `host_desktop.MANAGED_DESKTOP_USER` are the same rule in two
# languages, and bash cannot import the constant.** It named `vaelor-desktop`
# only while the appliance carried a second, pre-rename account - so the
# cleanup terminated an account that did not exist, the real session survived
# every stopped display, and the next browser desktop met the leftovers.
# Found on the Pi 2026-08-11 (#166). #168 removed that second account rather
# than adding to this list; `tests/test_host_desktop.py` compares the two
# sides so neither can gain or lose a name alone.
#
# The `-` prefix makes a miss non-fatal, so naming an account that has never
# logged in costs nothing and naming the present one is what matters.
ExecStopPost=-/usr/bin/loginctl terminate-user vaelor-desktop
EOF
install -d -m 0755 /etc/systemd/system/vaelor-workload-executor.service.d
install -d -m 0755 /etc/systemd/system/vaelor-workload-broker.service.d
if getent group docker >/dev/null; then
  cat >/etc/systemd/system/vaelor-workload-executor.service.d/docker.conf <<'EOF'
[Service]
SupplementaryGroups=vaelor-credentials docker
EOF
  cat >/etc/systemd/system/vaelor-workload-broker.service.d/docker.conf <<'EOF'
[Service]
SupplementaryGroups=vaelor docker
EOF
else
  rm -f /etc/systemd/system/vaelor-workload-executor.service.d/docker.conf
  rm -f /etc/systemd/system/vaelor-workload-broker.service.d/docker.conf
fi
install -d -m 0755 /etc/systemd/system/vaelor-control-plane.service.d
hardware_groups=()
for group in video render input gpio i2c spi; do
  getent group "${group}" >/dev/null && hardware_groups+=("${group}")
done
if ((${#hardware_groups[@]})); then
  printf '[Service]\nSupplementaryGroups=vaelor-jobs vaelor-credentials %s\n' \
    "${hardware_groups[*]}" \
    >/etc/systemd/system/vaelor-control-plane.service.d/hardware-groups.conf
fi
if [[ -e /dev/vcio ]] && getent group video >/dev/null; then
  install -d -m 0755 /etc/udev/rules.d
  cat >/etc/udev/rules.d/99-vaelor-rpi-vcio.rules <<'EOF'
# Ubuntu may create /dev/vcio as root:root 0600. Grant only the standard
# Raspberry Pi video group access for Vaelor's unprivileged PMIC adapter.
KERNEL=="vcio", GROUP="video", MODE="0660"
EOF
  chgrp video /dev/vcio
  chmod 0660 /dev/vcio
  udevadm control --reload-rules
fi
if getent group gpio >/dev/null &&
  compgen -G '/sys/class/thermal/cooling_device*/type' >/dev/null; then
  install -d -m 0755 /etc/udev/rules.d
  cat >/etc/udev/rules.d/99-vaelor-rpi-cpu-fan.rules <<'EOF'
# Permit only the gpio group to adjust the Raspberry Pi PWM cooling state.
SUBSYSTEM=="thermal", KERNEL=="cooling_device*", ATTR{type}=="pwm-fan", RUN+="/bin/chgrp gpio /sys%p/cur_state", RUN+="/bin/chmod 0664 /sys%p/cur_state"
EOF
  for type_path in /sys/class/thermal/cooling_device*/type; do
    [[ "$(cat "${type_path}" 2>/dev/null || true)" == "pwm-fan" ]] || continue
    state_path="${type_path%/type}/cur_state"
    chgrp gpio "${state_path}"
    chmod 0664 "${state_path}"
  done
  udevadm control --reload-rules
fi
cat >/etc/systemd/system/vaelor-control-plane.service.d/tls.conf <<'EOF'
[Service]
Environment=VAELOR_TLS_CERT=/opt/vaelor/tls/vaelor.crt
Environment=VAELOR_TLS_KEY=/opt/vaelor/tls/vaelor.key
Environment=VAELOR_SECURE_COOKIES=1
EOF

# hp-wmi-sensors ships with the distribution and is not loaded by default, and
# nothing prompts anyone to load it. Without it a machine with three readable
# fans and seven labelled board temperatures reports none of them, which is how
# Vaelor came to tell an HP workstation it had no fans.
#
# Gated on the firmware actually offering the sensors: the WMI GUID is what
# says so. Loading a module on a host whose firmware does not implement it
# would create an empty hwmon node and turn "we did not look" into "we looked
# and there is nothing", which is worse. Failure here is never fatal - the
# reader degrades with a stated reason, and an appliance must not refuse to
# install because an optional sensor driver would not load.
install_wmi_sensors() {
  local guid="5FB7F034-2C63-45e9-BE91-3D44E2C707E4"
  case "$(uname -m)" in
    x86_64 | amd64) ;;
    *) return 0 ;;
  esac
  if ! ls /sys/bus/wmi/devices 2>/dev/null | grep -qiF "${guid}"; then
    echo "This host's firmware does not publish the HP WMI sensor GUID; " \
      "skipping hp-wmi-sensors."
    return 0
  fi
  if ! modprobe hp-wmi-sensors 2>/dev/null; then
    echo "hp-wmi-sensors could not be loaded; fan and board temperatures " \
      "will report as unavailable with a reason." >&2
    return 0
  fi
  install -d -m 0755 /etc/modules-load.d
  printf '# Fans and board temperatures for HP hardware. See vaelor/wmi_sensors.py.\nhp-wmi-sensors\n' \
    > /etc/modules-load.d/vaelor-hp-wmi-sensors.conf
  echo "hp-wmi-sensors loaded and configured to load at boot."
}
install_wmi_sensors || true

# The neural processor publishes no utilisation or power through the kernel, so
# `amd-smi` is the only path to either. Gated on the device actually being
# bound, for the same reason as the WMI GUID check above: installing a vendor
# tool on a host with no such device turns "we did not look" into "we looked
# and there is nothing".
#
# Two things this deliberately does NOT do, both measured on the target
# hardware rather than assumed:
#
# * **It never displaces an amd-smi already on PATH.** Ubuntu 26.04's own
#   `amd-smi` (7.2.0-3, library 26.2.1) reports the entire `usage` block as the
#   string "N/A" on a Ryzen AI Max and publishes no `apu_*` field at all. AMD's
#   ROCm build (26.5.0) on the same machine in the same minute publishes every
#   one of them. Replacing the second with the first would remove a working
#   reading.
# * **It does not add AMD's repository.** That build comes from
#   repo.amd.com, and adding a third-party APT source changes what the
#   appliance trusts. That is an owner's decision, not an installer's. Where
#   only Ubuntu's build is available the NPU readings stay unavailable and the
#   control plane says which of the two it is.
install_amd_smi() {
  case "$(uname -m)" in
    x86_64 | amd64) ;;
    *) return 0 ;;
  esac
  if command -v amd-smi >/dev/null 2>&1; then
    echo "amd-smi is already installed; leaving it alone."
    return 0
  fi
  local bound=0
  for link in /sys/bus/pci/devices/*/driver; do
    [[ -e "${link}" ]] || continue
    case "$(basename "$(readlink -f "${link}")")" in
      amdgpu | amdxdna) bound=1 ;;
    esac
  done
  if ((bound == 0)); then
    echo "No amdgpu or amdxdna device is bound on this host; skipping amd-smi."
    return 0
  fi
  if ((skip_system_packages)); then
    echo "Native-package install: amd-smi is left to the package manifest."
    return 0
  fi
  if ! apt-get install -y amd-smi; then
    echo "amd-smi is not available from this host's repositories; " \
      "neural-processor activity and power will report as unavailable " \
      "with a reason." >&2
    return 0
  fi
  echo "amd-smi installed. If the neural-processor readings still report as" \
    "unavailable, this build publishes no APU metrics for this device and" \
    "AMD's ROCm build is the one that does."
}
install_amd_smi || true

# The NPU Assistant runs out of the lemonade-server snap: it carries the NPU
# `flm-real` binary at the fixed path `vaelor/flm_service.py` references.
#
# This snap was once assumed to ALSO ship "TheRock's gfx1151 GPU libraries"
# the GPU AI-Chat model needs, but the current snap (v11.7.0) ships none of
# them - so a bare box that relied on the snap for them silently fell back to
# CPU. The gfx1151 ROCm 7.x runtime the GPU model links against is provisioned
# separately by install_rocm_gfx1151_runtime, and the ROCmFPX engine itself by
# install_gpu_rocmfpx_engine (both below); the snap is the NPU half only.
#
# Auto-provisioned per the owner's decision so a fresh Strix Halo box is
# turnkey, and gated exactly like amd-smi above: only where an AMD accelerator
# is actually bound, because installing an AI stack on a host with no such
# device turns "we did not look" into "we looked and there is nothing".
#
# Source stated plainly: this is the `lemonade-server` snap from the snap store
# (publisher Ken VanDine), pinned to `latest/stable` and held so it does not
# auto-update underneath the appliance - the same `held` state the Z2 carries.
# Non-fatal: an appliance must never fail to install because an optional AI
# stack could not be fetched.
install_npu_stack() {
  case "$(uname -m)" in
    x86_64 | amd64) ;;
    *) return 0 ;;
  esac
  if command -v snap >/dev/null 2>&1 && snap list lemonade-server >/dev/null 2>&1; then
    echo "The lemonade-server snap is already installed; leaving it alone."
    return 0
  fi
  local bound=0
  for link in /sys/bus/pci/devices/*/driver; do
    [[ -e "${link}" ]] || continue
    case "$(basename "$(readlink -f "${link}")")" in
      amdgpu | amdxdna) bound=1 ;;
    esac
  done
  if ((bound == 0)); then
    echo "No amdgpu or amdxdna device is bound on this host; skipping the lemonade-server snap."
    return 0
  fi
  # A snap is a system package, so native-package mode defers it the way
  # amd-smi is deferred to the manifest above.
  if ((skip_system_packages)); then
    echo "Native-package install: the lemonade-server snap is left to the package manifest."
    return 0
  fi
  if ! command -v snap >/dev/null 2>&1; then
    echo "snap is unavailable on this host; the NPU Assistant and the GPU" \
      "model will report as unavailable until the lemonade-server snap is" \
      "installed." >&2
    return 0
  fi
  if ! snap install lemonade-server --channel=latest/stable; then
    echo "The lemonade-server snap could not be installed from the snap store;" \
      "the NPU Assistant and the GPU model will report as unavailable until" \
      "the lemonade-server snap is installed." >&2
    return 0
  fi
  # Hold it so the appliance's AI stack does not shift under a background
  # snap refresh. Non-fatal: a failed hold does not undo a good install.
  snap refresh --hold lemonade-server || true
  echo "lemonade-server installed from the snap store (publisher Ken VanDine)" \
    "and held against auto-refresh. It provides the NPU flm-real binary the" \
    "control plane serves the Assistant from; the GPU AI-Chat runtime and" \
    "engine are provisioned separately (gfx1151 ROCm + ROCmFPX)."
}
install_npu_stack || true

# The GPU AI-Chat model runs on the ROCmFPX llama.cpp fork, which the control
# plane's `vaelor/gpu_rocmfpx_service.py` launches by fixed path
# (`/var/lib/vaelor/engines/rocmfpx/bin/llama-server`). That engine is a
# personal prebuilt published as a third-party GitHub release, not a distro or
# vendor package, so this fetches it exactly one way: the release tarball is
# downloaded, its SHA-256 is verified against a pinned digest, and it is
# installed ONLY if the checksum matches - an unverified binary is never
# placed on the appliance.
#
# Auto-provisioned per the owner's decision so a fresh Strix Halo box is
# turnkey, and gated like amd-smi/the snap above: only where an AMD GPU is
# bound, because the ROCmFPX engine is useless on a host that cannot run it.
# (The control plane's own `discover_gpu_rocm_serving` runtime-gates on
# gfx1151, so amdgpu-bound is a sufficient install-time gate.) Non-fatal on
# every failure - no network, a 404, a checksum mismatch, or a missing
# curl/tar - so an appliance never fails to install because an optional AI
# engine could not be fetched.
install_gpu_rocmfpx_engine() {
  case "$(uname -m)" in
    x86_64 | amd64) ;;
    *) return 0 ;;
  esac
  local engine_bin="/var/lib/vaelor/engines/rocmfpx/bin/llama-server"
  if [[ -x "${engine_bin}" ]]; then
    echo "The ROCmFPX GPU engine is already provisioned; leaving it alone."
    return 0
  fi
  local bound=0
  for link in /sys/bus/pci/devices/*/driver; do
    [[ -e "${link}" ]] || continue
    case "$(basename "$(readlink -f "${link}")")" in
      amdgpu) bound=1 ;;
    esac
  done
  if ((bound == 0)); then
    echo "No amdgpu GPU is bound on this host; skipping the ROCmFPX GPU engine."
    return 0
  fi
  # An offline native-package install must not reach out to GitHub: a pre-built
  # image provisions the engine at build time (the idempotency check above then
  # skips), so defer the network fetch the way the snap and amd-smi do.
  if ((skip_system_packages)); then
    echo "Native-package install: the ROCmFPX GPU engine is left to the image."
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
    echo "curl or tar is unavailable; the GPU AI-Chat model will report as" \
      "unavailable until the ROCmFPX engine is provisioned." >&2
    return 0
  fi
  # Third-party GitHub release, SHA-256 pinned. Never install without a match.
  local release_tarball_url="https://github.com/julianmb/q38rocm/releases/download/v1.0.0/strix-halo-rocmfpx-engine-v1.0.0-linux-x86_64.tar.gz"
  local expected_tarball_sha="bbc7845db0c012b97f1c9b8a2733a7083c6f9a749a453866fbe1994151d3364f"
  local tmp_tarball tmp_extract
  tmp_tarball="$(mktemp)" || return 0
  tmp_extract="$(mktemp -d)" || { rm -f "${tmp_tarball}"; return 0; }
  if ! curl -fL "${release_tarball_url}" -o "${tmp_tarball}"; then
    echo "The ROCmFPX engine tarball could not be downloaded; the GPU AI-Chat" \
      "model will report as unavailable until the ROCmFPX engine is" \
      "provisioned." >&2
    rm -rf "${tmp_tarball}" "${tmp_extract}"
    return 0
  fi
  if ! echo "${expected_tarball_sha}  ${tmp_tarball}" | sha256sum -c - >/dev/null 2>&1; then
    echo "The ROCmFPX engine tarball failed SHA-256 verification and was NOT" \
      "installed; the GPU AI-Chat model will report as unavailable until the" \
      "ROCmFPX engine is provisioned." >&2
    rm -rf "${tmp_tarball}" "${tmp_extract}"
    return 0
  fi
  if ! tar -xzf "${tmp_tarball}" -C "${tmp_extract}"; then
    echo "The ROCmFPX engine tarball could not be extracted; the GPU AI-Chat" \
      "model will report as unavailable until the ROCmFPX engine is" \
      "provisioned." >&2
    rm -rf "${tmp_tarball}" "${tmp_extract}"
    return 0
  fi
  install -d -m 0755 -o root -g root /var/lib/vaelor/engines/rocmfpx/bin
  # The engine ships llama-server alongside its co-located .so libraries; both
  # land in the one bin/ the GPU service points LD at. World-readable, and the
  # executables world-executable, so the root hardware bridge that launches it
  # is not the only reader.
  #
  # Locate that bin/ wherever the tarball puts it rather than assuming it sits
  # at the extraction root. The v1.0.0 release wraps everything in a
  # `strix-halo-rocmfpx-engine/` top-level directory, so a hardcoded
  # `${tmp_extract}/bin` was `cp: cannot stat .../bin/.` and left every clean
  # install with no GPU engine (found on the Z2 clean-install test). Finding it
  # by the llama-server binary survives that wrapper and any future re-layout.
  local src_bin
  src_bin="$(dirname "$(find "${tmp_extract}" -type f -name llama-server -print -quit 2>/dev/null)")"
  if [[ -z "${src_bin}" || ! -d "${src_bin}" ]]; then
    echo "The ROCmFPX engine tarball did not contain llama-server; the GPU" \
      "AI-Chat model will report as unavailable until the ROCmFPX engine is" \
      "provisioned." >&2
    rm -rf "${tmp_tarball}" "${tmp_extract}"
    return 0
  fi
  if ! cp -a "${src_bin}/." /var/lib/vaelor/engines/rocmfpx/bin/; then
    echo "The ROCmFPX engine files could not be installed; the GPU AI-Chat" \
      "model will report as unavailable until the ROCmFPX engine is" \
      "provisioned." >&2
    rm -rf "${tmp_tarball}" "${tmp_extract}"
    return 0
  fi
  chmod -R a+rX /var/lib/vaelor/engines/rocmfpx/bin
  rm -rf "${tmp_tarball}" "${tmp_extract}"
  echo "ROCmFPX GPU engine provisioned from julianmb/q38rocm v1.0.0, a" \
    "third-party GitHub release verified against its pinned SHA-256. It is" \
    "the llama.cpp fork the GPU AI-Chat model serves on."
}

# The ROCmFPX GPU engine above links its `libggml-hip.so` against a ROCm 7.x
# gfx1151 runtime - libamdhip64.so.7, libhipblas.so.3, librocblas.so.5,
# libhipblaslt.so.1, libhsa-runtime64.so.1, libamd_comgr.so.3,
# librocsolver.so.0, plus the gfx1151 rocBLAS Tensile kernels - resolved from
# /opt/rocm. Without that runtime libggml-hip.so silently falls back to CPU and
# the GPU model never runs on the GPU. Nothing else on a bare Strix Halo box
# provisions it: the lemonade-server snap ships none of these libraries, and
# the working Z2 only serves on the GPU because a full ROCm was already at
# /opt/rocm before the appliance install - this installer did not put it there.
# This step closes that gap so a bare box is turnkey.
#
# --- A deliberate, documented APT-source deviation --------------------------
# The InfluxDB and amd-smi steps above both refuse to add an APT source on
# principle - InfluxDB: "We do NOT add an APT source ... nothing is written to
# /etc/apt/sources.list*, so the appliance's standing trust on every future
# `apt upgrade` is unchanged. Adding a sources.list entry would be an owner's
# decision; this is not that." - and each fetches one pinned, sha-verified
# .deb instead. ROCm cannot be provisioned that way: it is a ~5.4 GB tree of
# many interdependent packages with transitive ROCm deps, not a single .deb, so
# per-.deb sha pinning is not workable. AMD's official signed repo is the
# supported gfx1151 channel and is exactly how the reference Z2 was
# provisioned, so this step DOES add that one source - deliberately, and
# narrowly, without weakening the posture those steps protect:
#   * signed-by a keyring only, and the key is BUNDLED with the installer
#     (deploy/amdrocm-keyring.gpg) and PINNED by fingerprint
#     (${ROCM_GFX1151_KEY_FINGERPRINT}), verified before use, never added to the
#     global apt trust store. It is not fetched: repo.amd.com serves the package
#     repo but returns 403 for every signing-key path, so a key URL is not an
#     option; the bundled key is AMD's public package-signing key, and pinning
#     it by fingerprint is stronger than trusting a URL fetch anyway;
#   * the package versions are PINNED (=${ROCM_GFX1151_VERSION}) and
#     `apt-mark hold`-held, so a background `apt upgrade` cannot shift the AI
#     stack under the appliance - the same "held" discipline the snap uses;
#   * on ANY failure the source and keyring this step added are removed again,
#     so a failed provision leaves the appliance's standing apt trust exactly
#     as it was, and GPU AI-Chat degrades honestly. The NPU path is never
#     blocked (called with `|| true`, and every branch `return`s non-fatally).
# Gated on an amdgpu binding exactly like install_gpu_rocmfpx_engine: this
# runtime exists only to serve that engine, so where the engine is skipped this
# is too, and it is useless on a host with no AMD GPU.
ROCM_GFX1151_VERSION="7.14.0-3"
# The primary fingerprint of AMD's public package-signing key that signs this
# repo (the key that Release.gpg verifies against). The bundled keyring
# deploy/amdrocm-keyring.gpg must match this before it is trusted.
ROCM_GFX1151_KEY_FINGERPRINT="D0F004A0025A1145C7807FCD0701EAC4D5E02107"

# Remove the AMD source and keyring this step installs, so a failed or aborted
# provision never leaves a standing third-party APT source on the appliance
# (the InfluxDB principle). The refresh drops the now-orphaned cached index.
rocm_gfx1151_remove_source() {
  rm -f /etc/apt/sources.list.d/amdrocm.list /etc/apt/keyrings/amdrocm.gpg
  apt-get update >/dev/null 2>&1 || true
}

install_rocm_gfx1151_runtime() {
  case "$(uname -m)" in
    x86_64 | amd64) ;;
    *) return 0 ;;
  esac
  # Idempotent: a working gfx1151 ROCm runtime is a real libamdhip64.so.7 AND a
  # gfx1151 rocBLAS Tensile kernel directory (the two halves libggml-hip.so
  # needs). Probe both before spending ~5.4 GB re-installing.
  #
  # Mirror the GPU service's resolver exactly (gpu_rocmfpx_service.py
  # resolve_rocm_lib_dir), which accepts the runtime under EITHER /opt/rocm/lib
  # OR any /opt/rocm/core-*/lib (candidate #3, added because AMD's packages can
  # land libamdhip64.so.7 under core-7.14/lib with no top-level /opt/rocm/lib).
  # A probe that only checked /opt/rocm/lib would false-negative on that layout,
  # re-install 5.4 GB every run, and disagree with the resolver about whether
  # the runtime is present.
  local rocm_hip_lib rocm_gfx_dir
  rocm_hip_lib="$(find /opt/rocm \
    \( -path '/opt/rocm/lib/libamdhip64.so.7' \
    -o -path '/opt/rocm/core-*/lib/libamdhip64.so.7' \) \
    -print -quit 2>/dev/null)"
  rocm_gfx_dir="$(find /opt/rocm -type d -path '*rocblas/library/gfx1151' \
    -print -quit 2>/dev/null)"
  if [[ -n "${rocm_hip_lib}" && -n "${rocm_gfx_dir}" ]]; then
    echo "A gfx1151 ROCm 7.x runtime is already present at /opt/rocm" \
      "(${rocm_hip_lib}); leaving it alone."
    return 0
  fi
  local bound=0
  for link in /sys/bus/pci/devices/*/driver; do
    [[ -e "${link}" ]] || continue
    case "$(basename "$(readlink -f "${link}")")" in
      amdgpu) bound=1 ;;
    esac
  done
  if ((bound == 0)); then
    echo "No amdgpu GPU is bound on this host; skipping the gfx1151 ROCm runtime."
    return 0
  fi
  # An offline native-package install must not add a repo or reach the network:
  # a pre-built image carries /opt/rocm (the probe above then skips), so defer
  # exactly as the snap, amd-smi and the ROCmFPX engine do.
  if ((skip_system_packages)); then
    echo "Native-package install: the gfx1151 ROCm runtime is left to the image."
    return 0
  fi
  for tool in gpg apt-get apt-mark; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
      echo "${tool} is unavailable; GPU AI-Chat is unavailable until a gfx1151" \
        "ROCm runtime is installed at /opt/rocm." >&2
      return 0
    fi
  done
  # Select AMD's repo path for THIS host's Ubuntu release from /etc/os-release
  # rather than hardcoding ubuntu2604. AMD publishes packages-multi-arch per
  # release as ubuntuXXYY (Ubuntu 26.04 -> ubuntu2604, 24.04 -> ubuntu2404).
  # Only Ubuntu is published on this channel; a non-Ubuntu Debian-family host,
  # or an Ubuntu with no VERSION_ID, has no AMD match and degrades honestly.
  # (ID/VERSION_ID were sourced from /etc/os-release near the top of this run.)
  if [[ "${ID:-}" != "ubuntu" || -z "${VERSION_ID:-}" ]]; then
    echo "AMD's gfx1151 ROCm channel publishes for Ubuntu only; this host is" \
      "'${ID:-unknown} ${VERSION_ID:-}', which AMD does not match. GPU AI-Chat" \
      "is unavailable until a gfx1151 ROCm runtime is installed at /opt/rocm." >&2
    return 0
  fi
  local ubuntu_code="ubuntu${VERSION_ID//./}"
  local keyring="/etc/apt/keyrings/amdrocm.gpg"
  local list="/etc/apt/sources.list.d/amdrocm.list"
  local repo_line
  repo_line="deb [arch=amd64 signed-by=${keyring}] https://repo.amd.com/rocm/packages-multi-arch/${ubuntu_code} stable main"
  # The pinned gfx1151 runtime + BLAS packages that land /opt/rocm and its
  # gfx1151 rocBLAS Tensile kernels, plus their transitive ROCm deps. Held
  # below so a background upgrade cannot move them.
  local packages=(
    "amdrocm-runtime7.14=${ROCM_GFX1151_VERSION}"
    "amdrocm-base7.14=${ROCM_GFX1151_VERSION}"
    "amdrocm-core7.14-gfx1151=${ROCM_GFX1151_VERSION}"
    "amdrocm-blas7.14-gfx1151=${ROCM_GFX1151_VERSION}"
  )
  install -d -m 0755 /etc/apt/keyrings
  # signed-by keyring only, from the BUNDLED key beside this installer - never
  # a URL (repo.amd.com returns 403 for every signing-key path). Resolve it via
  # ${script_dir}, the same way the wheel constraints and systemd/ payload are
  # found. VERIFY its primary fingerprint against the pinned constant before
  # trusting it; a missing file or a mismatch honest-degrades rather than
  # placing an unverified key.
  local bundled_key="${script_dir}/amdrocm-keyring.gpg"
  if [[ ! -r "${bundled_key}" ]]; then
    echo "The bundled AMD ROCm key (${bundled_key}) is missing; GPU AI-Chat is" \
      "unavailable until a gfx1151 ROCm runtime is installed at /opt/rocm." >&2
    return 0
  fi
  # Defense in depth: a plain "does the pinned fp appear anywhere" grep would
  # pass a decoy keyring of [attacker-primary + genuine-AMD-as-a-subkey], or any
  # keyring with an extra key, and apt then trusts EVERY key in a signed-by
  # keyring. So assert two things from gpg's colon output: the keyring holds
  # EXACTLY ONE public key (one `pub` record), AND the pinned fingerprint is
  # that key's PRIMARY fpr (the `fpr` record immediately after the `pub`), not a
  # subkey's. Anything else honest-degrades without installing the key.
  local rocm_key_colons rocm_key_pub_count rocm_key_primary_fpr
  rocm_key_colons="$(gpg --show-keys --with-colons "${bundled_key}" 2>/dev/null)"
  rocm_key_pub_count="$(awk -F: '$1=="pub"{n++} END{print n+0}' \
    <<<"${rocm_key_colons}")"
  rocm_key_primary_fpr="$(awk -F: \
    '$1=="pub"{want=1; next} $1=="fpr" && want{print $10; want=0}' \
    <<<"${rocm_key_colons}")"
  if [[ "${rocm_key_pub_count}" != "1" ||
    "${rocm_key_primary_fpr}" != "${ROCM_GFX1151_KEY_FINGERPRINT}" ]]; then
    echo "The bundled AMD ROCm key is not a single key whose primary" \
      "fingerprint is the pinned ${ROCM_GFX1151_KEY_FINGERPRINT} (saw" \
      "${rocm_key_pub_count} public key(s)); refusing an unverified key. GPU" \
      "AI-Chat is unavailable until a gfx1151 ROCm runtime is installed at" \
      "/opt/rocm." >&2
    return 0
  fi
  # Already dearmored/binary - copy it as-is (no dearmor needed).
  if ! cp -f "${bundled_key}" "${keyring}"; then
    echo "The verified AMD ROCm key could not be installed; GPU AI-Chat is" \
      "unavailable until a gfx1151 ROCm runtime is installed at /opt/rocm." >&2
    rocm_gfx1151_remove_source
    return 0
  fi
  chmod 0644 "${keyring}"
  printf '%s\n' "${repo_line}" >"${list}"
  if ! apt-get update; then
    echo "AMD's ROCm repo for ${ubuntu_code} could not be reached; GPU AI-Chat" \
      "is unavailable until a gfx1151 ROCm runtime is installed at /opt/rocm." >&2
    rocm_gfx1151_remove_source
    return 0
  fi
  if ! apt-get install -y "${packages[@]}"; then
    echo "The pinned gfx1151 ROCm ${ROCM_GFX1151_VERSION} packages could not be" \
      "installed; GPU AI-Chat is unavailable until a gfx1151 ROCm runtime is" \
      "installed at /opt/rocm." >&2
    rocm_gfx1151_remove_source
    return 0
  fi
  # Hold the ROCm packages against a background `apt upgrade` - the snap's
  # `--hold` discipline, in apt. The signed, version-pinned source stays (apt
  # needs it to satisfy the held versions); the hold is what keeps the AI stack
  # from drifting. Non-fatal: a failed hold does not undo a good install.
  #
  # Hold the whole RESOLVED dependency closure, not just the four top-level
  # packages: the install pulled a tree of transitive ROCm deps, and leaving any
  # of them unheld lets a background upgrade float it and skew the stack. Derive
  # the closure from what is actually installed in the amdrocm-*/rocm-*
  # namespace (dpkg-query over the installed set), which is exactly the packages
  # the four above dragged in. Fall back to the four top-level names if the
  # query yields nothing.
  local -a rocm_installed=()
  local pkg
  while IFS= read -r pkg; do
    [[ -n "${pkg}" ]] && rocm_installed+=("${pkg}")
  done < <(dpkg-query -W -f='${Package}\n' 'amdrocm-*' 'rocm-*' 2>/dev/null |
    sort -u)
  if ((${#rocm_installed[@]})); then
    apt-mark hold "${rocm_installed[@]}" >/dev/null 2>&1 || true
  else
    apt-mark hold "${packages[@]}" >/dev/null 2>&1 || true
  fi
  echo "gfx1151 ROCm ${ROCM_GFX1151_VERSION} runtime installed at /opt/rocm from" \
    "AMD's official signed repo (${ubuntu_code}) and held against apt upgrades." \
    "It is the ROCm 7.x runtime the ROCmFPX GPU engine links against."
}

# Runtime before engine, so the ROCm libraries libggml-hip.so links against are
# in place first. Both non-fatal - GPU AI-Chat degrades honestly and the NPU
# path is never blocked.
install_rocm_gfx1151_runtime || true
install_gpu_rocmfpx_engine || true

# If SunFounder's HAT runtime is present (offer accepted, or already on the
# appliance), take its hardware daemon and mask its competing web dashboard so
# only Vaelor's control plane serves (#207).
neutralize_sunfounder_control_plane

systemctl daemon-reload
systemctl enable "${vaelor_units[@]}"
systemctl restart "${vaelor_units[@]}"
health_deadline=$((SECONDS + 300))
healthy_samples=0
while ((SECONDS < health_deadline)); do
  services_healthy=1
  for unit in "${vaelor_units[@]}"; do
    if systemctl is-active --quiet "${unit}"; then
      continue
    fi
    # Optional platform adapters use explicit systemd conditions. An unmet
    # condition is a truthful unsupported capability, not a failed service.
    [[ "$(systemctl show --property=ConditionResult --value "${unit}")" == "no" ]] ||
      services_healthy=0
  done
  if [[ "${services_healthy}" -eq 1 ]] &&
    [[ -S /run/vaelor/credentiald.sock ]] &&
    [[ -S /run/vaelor/workloadd.sock ]] &&
    curl --fail --silent --show-error --insecure --max-time 5 \
      https://127.0.0.1:34001/api/v2/auth/status >/dev/null; then
    healthy_samples=$((healthy_samples + 1))
    if ((healthy_samples >= 3)); then
      break
    fi
  else
    healthy_samples=0
  fi
  sleep 2
done
if ((healthy_samples < 3)); then
  echo "Vaelor did not remain healthy for three consecutive checks." >&2
  systemctl status --no-pager "${vaelor_units[@]}" >&2 || true
  exit 1
fi
# Re-apply this release's serving settings to models already deployed.
#
# The upgrade replaced the code that decides the engine digest, the prompt-cache
# bound, the KV window and the container limit, and left every *deployed*
# compose at the previous release's rendering. Without this an install lands and
# the appliance goes on serving the old configuration, which is how a Pi ran an
# alpha-29 compose while everything reported alpha 35 (VD-081).
#
# **After the health gate, never before.** The executor performs the deploy, and
# an executor still on the previous release re-renders the previous release's
# compose while reporting success. Ordering is the correctness argument.
#
# Non-fatal: a refresh that cannot run must not fail an install that otherwise
# succeeded. It reports, and the next redeploy picks it up.
if /opt/vaelor/venv/bin/vaelor-refresh-models --apply; then
  :
else
  echo "Deployed models could not be refreshed; redeploy them from the UI to pick up this release's serving settings." >&2
fi

if [[ "${migrate}" -eq 1 ]]; then
  for unit in "${legacy_units[@]}"; do
    systemctl disable "${unit}" >/dev/null 2>&1 || true
  done
fi
migration_complete=1
trap - ERR

echo "Vaelor is installed at https://$(hostname -I | awk '{print $1}'):34001/v2/"

# The one and only reboot, dead last: after every step above has succeeded (the
# health gate would have exited non-zero otherwise) and after
# neutralize_sunfounder_control_plane has masked SunFounder's dashboard. It fires
# only when the HAT runtime was installed *this run* and so wrote device-tree
# overlays that need a reboot to load. No HAT install (declined, or not a Pi)
# means no reboot at all (#207). This is why SunFounder's own mid-install reboot
# prompt is suppressed above - a reboot here, never there.
if ((hat_installed)); then
  echo "The Pironman HAT overlays were written and need one reboot to load."
  if ((unattended)); then
    echo "Vaelor install complete; rebooting to activate the HAT overlays."
    reboot
  else
    printf "Reboot now to activate the Pironman HAT? [Y/n] "
    reboot_answer="y"
    read -r reboot_answer </dev/tty 2>/dev/null || reboot_answer="y"
    # [Y/n] default-yes: empty (just Enter) reboots, and so does "y"/"yes" (any
    # case). A naive user typing "yes" must not fall through to no-reboot and
    # leave the overlays unloaded (review UX #207).
    if [[ "${reboot_answer:-y}" =~ ^([Yy]([Ee][Ss])?)?$ ]]; then
      reboot
    else
      echo "Reboot before the enclosure controls will work."
    fi
  fi
fi
