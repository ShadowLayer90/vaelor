# Vaelor support

Check `SUPPORTED_PLATFORMS.md` before reporting a platform problem. “Detected”
does not imply that every desktop, enclosure control, model runtime, or package
manager is supported, and GPU serving is verified only on the AMD Strix Halo GPU.

For a useful bug report, include:

- Vaelor version and installation method;
- OS name/version and CPU architecture;
- the machine model (for example HP Z2 Mini G1a), or the Raspberry Pi model and
  Pironman enclosure, when applicable;
- for a cluster problem, which machine is the controller and which are workers,
  and how the model is served (one copy per machine, or split across machines);
- the visible error and the operation that caused it;
- whether the failure repeats after a normal service restart; and
- redacted service logs and job IDs.

Never attach credential databases, API keys (including LLM Server keys and
hosted-service keys), passwords, SSH private keys, or unredacted exported chats.
Replace addresses and host names with placeholders such as `192.0.2.10` and
`mini-pc` before posting logs publicly.

Use the security process in `SECURITY.md` for vulnerabilities. General usage
questions and reproducible non-security bugs may use the repository's issue
tracker.
