# Vaelor OCI core

This image packages the hardware-independent Vaelor web, Assistant, AI Chat,
database, and credential-broker core for both `linux/arm64` and `linux/amd64`.
It is not the full host appliance and does not replace the native installer.

The default Compose file:

- runs as an unprivileged UID with all Linux capabilities dropped;
- binds the web port to loopback only;
- does not mount the Docker socket, host devices, systemd, or package-manager
  state;
- persists application state and a generated 32-byte credential key in named
  volumes.

Put an HTTPS reverse proxy or trusted VPN in front of port 34001. Use the native
package when Vaelor must control Docker, updates, remote desktop, cooling, RGB,
OLED, power, or other host hardware.

Build a local architecture:

```sh
docker build \
  --file deploy/oci/Dockerfile \
  --build-arg VAELOR_VERSION=2.0.4 \
  --tag vaelor-control-plane:2.0.4 .
```

Create a two-architecture OCI archive:

```sh
deploy/oci/build-multiarch.sh 2.0.4
```

QEMU is not installed in or required by this image. The QEMU lab remains
developer-only acceptance infrastructure.
