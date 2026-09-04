# Vaelor security policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or exposed secret.
Until a dedicated project security mailbox is published, report privately to
the repository owner through the hosting provider's private vulnerability
reporting feature. Include the affected version, impact, reproduction steps,
and whether any credentials or user data may have been exposed.

Do not include real passwords, API keys, private keys, or unredacted database
files. A maintainer should acknowledge a complete report within seven days.
Public disclosure is coordinated after affected users have a reasonable
upgrade path.

## Supported security boundary

Vaelor is an administrative appliance. Anyone with Vaelor administrator access
can approve host-level changes. Its protection depends on:

- TLS for the web interface;
- short-lived authenticated sessions;
- encrypted broker storage for provider and SSH credentials;
- scoped, revocable inference tokens;
- fingerprint-pinned SSH enrollment;
- fixed-command privileged brokers;
- explicit approval before mutations; and
- least-privilege systemd service identities.

Application research is isolated in `vaelor-application-research`. The service
accepts no shell commands, file downloads, credentials, or raw Compose. It can
retrieve only bounded metadata over public HTTPS and treats every response as
untrusted evidence. DNS answers, connected peers, and every redirect are
validated to prevent SSRF, metadata access, and DNS rebinding. Application
deployment requires a digest-pinned architecture match, deterministic policy
validation, an immutable draft digest, and a separate administrator approval.

Do not expose the dashboard, broker sockets, noVNC gateway, model endpoints, or
SSH service directly to the public internet. Use a trusted LAN or a
properly-authenticated VPN.

## Release handling

Security fixes receive a versioned release and a plain-language impact note.
Published artifacts must include checksums, GPL-2.0-only metadata, corresponding
source, third-party notices, and a dependency manifest.
