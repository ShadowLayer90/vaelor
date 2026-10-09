/**
 * Which of an installed app's ports the owner's browser can open (W4d-D5).
 *
 * "Open" used to point at the controller's LAN address for every published
 * port, including ports Docker bound to 127.0.0.1 only (the web-research search
 * app on 8888, Phoenix on 6006) - links refused from every other machine. A
 * loopback-only port is not a link, and the card does not call it published.
 */

export interface AppPortBinding {
  container: string;
  host: string;
  address: string;
}

export interface AppPorts {
  ports: AppPortBinding[];
  /** Server-resolved host port of the web UI; null when no LAN-reachable one exists. */
  web_port?: number | null;
  /** Server-resolved ports bound only to loopback; absent on older servers. */
  local_only_ports?: number[];
  /** Shares the host's network: Docker lists no ports for it (W6 sweep). */
  host_network?: boolean;
}

const LOOPBACK = /^(127\.\d{1,3}\.\d{1,3}\.\d{1,3}|::1|\[::1\])$/;

/** Host ports every binding of which is loopback - the server's answer when it sent one. */
export function localOnlyPorts(app: AppPorts): number[] {
  if (app.local_only_ports) return app.local_only_ports;
  const byPort = new Map<number, boolean[]>();
  for (const port of app.ports) {
    const host = Number(port.host.split("/")[0]);
    if (!Number.isInteger(host) || host <= 0) continue;
    byPort.set(host, [...(byPort.get(host) ?? []), LOOPBACK.test(port.address ?? "")]);
  }
  return [...byPort].filter(([, flags]) => flags.every(Boolean)).map(([host]) => host).sort((a, b) => a - b);
}

/** The URL that opens the app from this browser, or null when none can. */
export function appEndpoint(app: AppPorts): string | null {
  const local = new Set(localOnlyPorts(app));
  // A multi-port app (Syncthing: 8384 GUI and 22000 sync) publishes more than
  // one host port; the server resolves the web one into `web_port`. A server
  // too old to send local_only_ports resolved it without the loopback rule, so
  // the port's own bindings are checked here too (R-F11).
  if (app.web_port && !local.has(app.web_port)) return `http://${window.location.hostname}:${app.web_port}`;
  // A server that reports loopback ports resolved web_port with them in mind:
  // no web_port means nothing here can be opened.
  if (app.local_only_ports) return null;
  const published = app.ports
    .map((port) => port.host.split("/")[0])
    .find((host) => host && !local.has(Number(host)));
  return published ? `http://${window.location.hostname}:${published}` : null;
}

/** "2 published ports · 80, 443", with loopback-only ports said as such. */
export function portSummary(app: AppPorts): string {
  const local = new Set(localOnlyPorts(app));
  const hosts = [...new Set(app.ports.map((port) => port.host).filter(Boolean))];
  const reachable = hosts.filter((host) => !local.has(Number(host)));
  const parts: string[] = [];
  if (reachable.length) {
    parts.push(`${reachable.length} published port${reachable.length === 1 ? "" : "s"} · ${reachable.join(", ")}`);
  }
  if (local.size) parts.push(`${[...local].join(", ")} on this machine only`);
  if (!parts.length && app.host_network) return "Uses this machine's network directly; Docker lists no ports for it";
  return parts.length ? parts.join("; ") : "No published ports";
}
