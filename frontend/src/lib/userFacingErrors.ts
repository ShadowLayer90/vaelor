/**
 * Turn a raw operation failure into something a person can act on.
 *
 * Alpha 11 rendered the executor's exception text directly, so the activity
 * feed showed things like
 *   "[Errno 2] No such file or directory: '/var/lib/vaelor/workloads/bad-stack/compose.yaml'"
 *   "yaml: line 2: did not find expected ',' or ']'"
 * on a screen that promises "No terminal or Linux commands required".
 *
 * Nothing is discarded: the original text is returned as `technical` for a
 * disclosure and stays intact in logs. Only the *primary* message is rewritten.
 */

export interface UserFacingError {
  /** One sentence stating what happened, in plain language. */
  summary: string;
  /** Why it most likely happened, when that can be inferred. */
  cause?: string;
  /** What the user can do next. */
  recovery?: string;
  /** The original message, for a "Technical details" disclosure. */
  technical?: string;
  /** True when the wording was rewritten rather than passed through. */
  sanitized: boolean;
}

/** Anything here must never be the primary message shown to a user. */
const TECHNICAL_MARKERS = [
  /\[Errno \d+\]/i,
  /\b(?:Traceback|File "\/|line \d+, in )/,
  // No adjacent colon required: "Error response from daemon" must match too.
  /\b[A-Za-z_]+(?:Error|Exception)\b/,
  // Any absolute POSIX path, not a hand-picked list of top-level directories.
  /(?:^|[\s"'([])\/(?:[A-Za-z0-9._-]+\/)+[A-Za-z0-9._-]*/,
  // A Windows path has ONE backslash; `\\\\` in a literal is two and never matched.
  /\b[A-Za-z]:\\/,
  /\byaml:\s*line \d+/i,
  /\bsha256:[0-9a-f]{16,}/i,
  /\bexit (?:code|status) \d+/i,
  // Go/OCI runtime noise that reaches users through Docker.
  /\b\w+\.go:\d+/,
  /\bOCI runtime\b/i,
  /\bresponse from daemon\b/i,
  /\bnet\/http\b/i,
  /\bexecutable file not found\b/i,
  /\$PATH\b/,
  /\b0\.0\.0\.0:\d+|\b127\.0\.0\.1:\d+/,
];

export function containsTechnicalDetail(message: string): boolean {
  const text = String(message ?? "");
  return TECHNICAL_MARKERS.some((pattern) => pattern.test(text));
}

interface Rule {
  match: RegExp;
  summary: string;
  cause?: string;
  recovery?: string;
}

/**
 * Ordered most specific first. Each entry is a failure a user can actually do
 * something about, phrased as the outcome rather than the mechanism.
 */
const RULES: Rule[] = [
  {
    match: /rollback is incomplete/i,
    summary: "The change failed and Vaelor could not fully undo it.",
    cause: "The app may be left partly configured.",
    recovery: "Review this app before retrying, and restore a checkpoint if one exists.",
  },
  {
    match: /\[Errno 2\]|No such file or directory/i,
    summary: "Vaelor could not find a file this app needs.",
    cause: "The app's setup is incomplete, or its files were removed outside Vaelor.",
    recovery: "Review the app's configuration, or remove the incomplete entry and set it up again.",
  },
  {
    match: /\[Errno 13\]|Permission denied/i,
    summary: "Vaelor was not allowed to read or write a file this app needs.",
    recovery: "Check that the appliance's storage is writable, then retry.",
  },
  {
    match: /\[Errno 28\]|No space left on device/i,
    summary: "The appliance has run out of storage.",
    recovery: "Free space or remove unused apps and models, then retry.",
  },
  {
    match: /^yaml:|did not find expected|could not find expected|mapping values are not allowed/i,
    summary: "The Compose file could not be read.",
    cause: "There is a formatting mistake in the YAML.",
    recovery: "Check the line noted in the technical details, then paste the corrected file.",
  },
  {
    match: /connection refused|connection reset|timed out|timeout/i,
    summary: "Vaelor could not reach the service in time.",
    recovery: "Check the network and that the service is running, then retry.",
  },
  {
    match: /manifest unknown|not found: manifest|pull access denied|unauthorized/i,
    summary: "The container image could not be downloaded.",
    cause: "The image name or tag may be wrong, or the registry needs credentials.",
    recovery: "Check the image reference, then retry.",
  },
  {
    match: /port is already allocated|address already in use|bind for .* failed/i,
    summary: "The web address port this app wants is already taken.",
    cause: "Another app or service on this appliance is already using it.",
    recovery: "Choose a different port for this app, then retry.",
  },
  {
    match: /executable file not found|command not found|no such file or directory: "?docker/i,
    summary: "A program this operation needs is not installed.",
    cause: "Docker or one of its tools is missing from the appliance.",
    recovery: "Install Docker from System, then retry.",
  },
  {
    match: /tls handshake|certificate (?:has expired|verify failed|is not valid)|x509/i,
    summary: "Vaelor could not establish a secure connection.",
    cause: "The remote service's certificate was rejected, or the appliance clock is wrong.",
    recovery: "Check the appliance date and time, then retry.",
  },
  {
    match: /OCI runtime|container_linux\.go|failed to create shim/i,
    summary: "The container could not be started.",
    cause: "Its start command or image may be incompatible with this appliance.",
    recovery: "Review the app's configuration, then retry.",
  },
];

const GENERIC: Rule = {
  match: /.*/,
  summary: "This operation did not finish.",
  recovery: "Retry, or open the technical details to see exactly what failed.",
};

export function toUserFacingError(message: string): UserFacingError {
  const raw = String(message ?? "").replace(/\s+/g, " ").trim();
  if (!raw) {
    return { summary: "This operation did not finish.", sanitized: false };
  }

  const rule = RULES.find((candidate) => candidate.match.test(raw));
  if (!rule) {
    // Plain-language messages the executor already writes well are left alone.
    if (!containsTechnicalDetail(raw)) {
      return { summary: raw, sanitized: false };
    }
    return { ...GENERIC, summary: GENERIC.summary, technical: raw, sanitized: true };
  }

  return {
    summary: rule.summary,
    cause: rule.cause,
    recovery: rule.recovery,
    // Keep the original whenever it carries detail worth inspecting.
    technical: containsTechnicalDetail(raw) || raw !== rule.summary ? raw : undefined,
    sanitized: true,
  };
}
