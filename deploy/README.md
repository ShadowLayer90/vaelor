# Vaelor deployment files

`install-vaelor.sh` is the supported host installer for Debian-family `arm64`
and `amd64` systems. `maintain-vaelor.sh` provides bounded status, repair, and
uninstall operations. The `systemd` directory contains only active Vaelor
units.

Pironman-era unit names appear in migration discovery so an existing appliance
can be stopped, validated, rolled back on failure, and retired safely. Their
obsolete unit definitions and installers are intentionally not redistributed.

`oci/` defines the restricted portable control-plane core. It does not receive
host Docker, update, remote-desktop, power, or hardware privileges.
