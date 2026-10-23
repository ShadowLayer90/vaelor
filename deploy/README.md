# Vaelor deployment files

`install-vaelor.sh` is the supported host installer for Debian-family `arm64`
and `amd64` systems. `maintain-vaelor.sh` provides bounded status, repair, and
uninstall operations. The `systemd` directory contains only active Vaelor
units.

`maintain-vaelor.sh uninstall` has three modes, each behind a typed
confirmation (`--help` prints the exact phrases). Every mode stops the
containers Vaelor runs: the managed application projects, the GPU AI-Chat
model, the LLM Server proxy, the Phoenix trace collector and each custom
agent's gate. It also stops and removes any GPU-cluster (Mode B) serving this
node runs - the vLLM servers, Ray workers and gates, the model-pull oneshots,
and the balancer container - so no LLM is left answering on the LAN. Vaelor's
containers are found by Vaelor's owner label, or by name when unlabelled and
outside any Compose project, so a worker, which has no Vaelor runtime, finds its
own and an owner's Compose app is never taken. A worker node's own replicas and gates are removed by that worker's
uninstall, not this one. Every mode also turns off GNOME Remote Login (RDP) when
Vaelor turned it on, and clears the credentials Vaelor stored for it; Remote
Login set up without Vaelor is left alone.

- **Keep data** (`--confirm uninstall-vaelor-keep-data`): removes the software
  and services; keeps users, credentials, models, workloads, chats, settings
  and every container image, so a reinstall is fast.
- **Purge data** (`--purge-data`): also deletes all of that, the browser-desktop
  account (`vaelor-desktop`) with its home and its AccountsService record,
  Vaelor's own InfluxDB database (`vaelor`), and the container images Vaelor
  pulled - the two gfx1151 serving images, the LLM Server's nginx, the
  multi-node vLLM image, and the Phoenix, web-search and backup-helper images -
  by the reference pinned in the installed package (the script asks the venv
  for the list; it never carries a copy). Images Vaelor did not pull, InfluxDB
  itself and Docker itself stay. On a cluster worker a controller enrolled it
  is a bare-OS purge (below): the worker goes back to bare OS, by Vaelor's
  record of what it installed (or the full stack on a worker that has no
  record) and Docker always - so EVERY container, image and volume on that
  worker goes, not only Vaelor's. A purge stopped part-way is finished by
  running it again. Bare OS removes Docker's, containerd's and InfluxDB's data
  only once each package is really gone, and its last line names anything
  still installed.
- **Bare OS** (`--purge-data --bare-os`): also removes the OS stack the
  installer added - Docker and containerd with both image stores
  (`/var/lib/docker` and `/var/lib/containerd`, so *every* container image on
  the box goes), the `docker` group and Docker's bridges, InfluxDB with its
  database and `/etc/influxdb`, the ROCm gfx1151 packages and `/opt/rocm`,
  amd-smi, novnc, the TigerVNC and GNOME Flashback packages the browser
  desktop installed, and the AMD apt source and keyring - and unmasks
  SunFounder's `pironman5.service`. It purges only the OS packages Vaelor
  recorded installing (the marker at
  `/usr/lib/vaelor/release/installed-by-vaelor`, written the moment the
  installer ran `apt-get` for each), so a Docker, InfluxDB or amd-smi that was
  already present when Vaelor was installed - one of the installer's three
  "already installed; leaving it alone" branches - is left alone; an appliance
  installed before Vaelor kept that record falls back to purging the full stack
  its installer can add and prints a warning that pre-existing packages cannot
  be told apart. Kept, deliberately: the shared OS tools the installer only made
  sure of (python3, curl, git, openssl) and anything it did not add, such as a
  hand-installed FastFlowLM package under `/opt/fastflowlm` or model copies
  under `/opt/flm-models-preserved`. A package apt leaves half-purged is
  finished, and anything still installed afterwards is named on stderr with the
  command that removes it.

The in-console "Remove Vaelor" runs the bare-OS mode from the copy the
installer stages at `/usr/lib/vaelor/release/maintain-vaelor.sh`.

Pironman-era unit names appear in migration discovery so an existing appliance
can be stopped, validated, rolled back on failure, and retired safely. Their
obsolete unit definitions and installers are intentionally not redistributed.

`oci/` defines the restricted portable control-plane core. It does not receive
host Docker, update, remote-desktop, power, or hardware privileges.
