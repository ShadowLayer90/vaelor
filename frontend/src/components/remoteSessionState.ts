export type RemoteSessionState =
  | "unavailable" | "not_configured" | "configuring" | "ready"
  | "connecting" | "connected" | "failed" | "ended";

/**
 * Labels for the browser-desktop session, where "Needs setup" is truthful:
 * there is an "Install browser desktop" button and Vaelor really can walk that
 * path. The physical-KVM path has its own labels in `PhysicalKvmStage`,
 * because there every remaining step happens at the machine.
 */
export function sessionStateLabel(state: RemoteSessionState) {
  return {
    unavailable: "Unavailable",
    not_configured: "Needs setup",
    configuring: "Configuring",
    ready: "Ready",
    connecting: "Connecting",
    connected: "Connected",
    failed: "Failed",
    ended: "Ended",
  }[state];
}
