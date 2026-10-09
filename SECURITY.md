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

- TLS for the web interface, with a certificate that lists the machine's names
  and addresses;
- short-lived authenticated sessions, and sign-in attempts limited per address
  and per account;
- encrypted broker storage for provider and SSH credentials;
- scoped, revocable inference tokens, and LLM Server keys that are shown once,
  stored only as fingerprints, and can be rotated or revoked;
- fingerprint-pinned SSH enrollment;
- fixed-command privileged brokers;
- explicit approval before mutations;
- least-privilege systemd service identities; and
- an audit trail, written in words, that records every API-key change and refusal
  and the real client address of each request.

Internal services listen on loopback only: InfluxDB, the single-node model
server, every cluster vLLM API server, and the Arize Phoenix trace collector.
A cluster worker's model is reached only through a keyed gate. A model split
across machines is served only over a dedicated link that Vaelor fences with
nftables, and its Ray processes require a per-deployment token; Ray, RCCL and
Gloo have no other authentication, so that fence is their guard. Worker
telemetry reaches the controller only through a keyed endpoint over TLS pinned
to the controller's certificate.

Hosted AI services are available to AI Chat only and are reached through a
pinned HTTPS transport that accepts public addresses only, connects to the
address it checked, refuses redirects, verifies the certificate, enforces a
deadline, and redacts the key from every log and error. The Assistant never
sends a prompt to a hosted service.

Application research is isolated in `vaelor-application-research`. The service
accepts no shell commands, file downloads, credentials, or raw Compose. It can
retrieve only bounded metadata over public HTTPS and treats every response as
untrusted evidence. DNS answers, connected peers, and every redirect are
validated to prevent SSRF, metadata access, and DNS rebinding. Application
deployment requires a digest-pinned architecture match, deterministic policy
validation, an immutable draft digest, and a separate administrator approval.

Do not expose the dashboard, broker sockets, noVNC gateway, model endpoints
(including the LLM Server on port 11434), or SSH service directly to the public
internet. Use a trusted LAN or a
properly-authenticated VPN.

## Release handling

Security fixes receive a versioned release and a plain-language impact note.
Published artifacts must include checksums, GPL-2.0-only metadata, corresponding
source, third-party notices, and a dependency manifest.
