import { useEffect, useRef, useState } from "react";
import { Icon } from "./Icon";
import { Button, Input, Notice } from "./ui";

/**
 * Post-deploy setup guidance for a catalog app, surfaced in the App Manager so
 * a beginner knows how to actually sign in and finish configuring the app it
 * just installed. All fields are optional; the backend's public_catalog()
 * attaches this per template. Render only the fields that are present.
 */
export interface AppSetup {
  /** Default credentials or a short sign-in note, e.g. "admin / admin". */
  login?: string;
  /** Where to find a generated password, e.g. "Shown in the Logs tab…". */
  password_source?: string;
  /** The app only works over HTTPS; show an HTTPS-required warning. */
  https_required?: boolean;
  /** The app opens its own setup wizard on first visit. */
  first_run_wizard?: boolean;
  /** One app-specific caveat worth stating up front. */
  note?: string;
  /** Ordered, numbered first-run steps: the primary beginner walkthrough. */
  steps?: string[];
}

export interface AppTemplate {
  id: string;
  name: string;
  category: string;
  description: string;
  image: string;
  default_port: number;
  container_port: number;
  memory: string;
  storage: string;
  source: string;
  /** Optional post-deploy setup guidance; absent on older servers. */
  setup?: AppSetup | null;
  /**
   * Names of the Vaelor-generated secret environment variables this app needs
   * revealed after install (e.g. code-server's `PASSWORD`). Non-empty means the
   * App Manager fetches and shows the actual value in the setup panel. Absent on
   * older servers.
   */
  secret_env?: string[];
  /**
   * Relative paths of the user-editable configuration files this app reads
   * (e.g. Homepage's `services.yaml`). Non-empty means the App Manager offers a
   * Files tab for editing them. Absent on older servers.
   */
  config_files?: string[];
}

export interface PortPreflight {
  requested_port: number;
  available: boolean;
  conflict: boolean;
  suggested_port: number | null;
  reason: string;
}

export function AppCatalog({
  templates,
  busy,
  disabled,
  onClose,
  onInstall,
  onPreflight,
  initialTemplateId,
  initialPort,
  installedTemplateIds = [],
  onOpenInstalled,
}: {
  templates: AppTemplate[];
  busy: boolean;
  disabled: boolean;
  onClose: () => void;
  onInstall: (template: AppTemplate, port: number) => void | Promise<void>;
  onPreflight?: (port: number) => Promise<PortPreflight>;
  initialTemplateId?: string;
  initialPort?: number;
  installedTemplateIds?: string[];
  onOpenInstalled?: () => void;
}) {
  const allInstalled = templates.length > 0 && templates.every((template) => installedTemplateIds.includes(template.id));
  const initialTemplate = templates.find((template) => template.id === initialTemplateId) ?? null;
  const [selected, setSelected] = useState<AppTemplate | null>(initialTemplate);
  const [port, setPort] = useState(initialPort ?? initialTemplate?.default_port ?? 3000);
  const [preflight, setPreflight] = useState<PortPreflight | null>(null);
  const [preflightPending, setPreflightPending] = useState(false);
  const installInFlight = useRef(false);
  const clientPortProblem = !Number.isFinite(port) || !Number.isInteger(port)
    ? "Enter a whole-number port from 1024 to 65535."
    : port < 1024
      ? "Use a port from 1024 to 65535. Lower ports are reserved by the operating system."
    : port > 65535
      ? "Ports cannot be higher than 65535."
      : [34001, 34002].includes(port)
        ? "That port is reserved for the Vaelor control plane."
        : "";
  const portProblem = clientPortProblem || (preflight?.conflict ? preflight.reason : "");

  useEffect(() => {
    if (!selected || clientPortProblem || !onPreflight) {
      setPreflight(null);
      setPreflightPending(false);
      return;
    }
    let current = true;
    setPreflightPending(true);
    void onPreflight(port).then((result) => {
      if (current) setPreflight(result);
    }).catch(() => {
      if (current) setPreflight(null);
    }).finally(() => {
      if (current) setPreflightPending(false);
    });
    return () => { current = false; };
  }, [clientPortProblem, onPreflight, port, selected]);

  const choose = (template: AppTemplate) => {
    setSelected(template);
    setPort(template.default_port);
    setPreflight(null);
  };

  const approveInstall = async () => {
    if (!selected || installInFlight.current || busy || disabled || preflightPending || portProblem) return;
    installInFlight.current = true;
    try {
      await onInstall(selected, port);
    } finally {
      installInFlight.current = false;
    }
  };

  return (
    <section className="app-catalog" aria-labelledby="app-catalog-title">
      <div className="model-catalog__header">
        <div><span className="page-eyebrow">App catalog</span><h2 id="app-catalog-title">{allInstalled ? "All catalog apps are managed" : "Choose an app"}</h2><p>{allInstalled ? "Every available blueprint already has a managed instance. View an app in Manage to open it, inspect health, or change its setup." : "Every template uses resource limits, managed storage, and a configuration Vaelor can back up and repair."}</p></div>
        <Button onClick={onClose} type="button" variant="quiet">Close</Button>
      </div>
      <div className="app-template-grid">
        {templates.map((template) => {
          const installed = installedTemplateIds.includes(template.id);
          return (
          <article className="app-template" key={template.id}>
            <div className="app-template__icon"><Icon name={template.id === "grafana" ? "activity" : template.id === "uptime-kuma" ? "shield" : "network"} /></div>
            <small>{template.category}</small>
            <h3>{template.name}</h3>
            <p>{template.description}</p>
            <dl>
              <div><dt>Memory limit</dt><dd>{template.memory}</dd></div>
              <div><dt>Storage</dt><dd>{template.storage}</dd></div>
              <div><dt>Default address</dt><dd>Port {template.default_port}</dd></div>
            </dl>
            <Button disabled={disabled || (installed && !onOpenInstalled)} onClick={() => installed ? onOpenInstalled?.() : choose(template)} type="button" variant="secondary">{installed ? "View in Manage" : "Review installation"}</Button>
          </article>
          );
        })}
      </div>
      {selected && (
        <div className="app-install-review" role="region" aria-labelledby="app-install-review-title">
          <div>
            <span className="page-eyebrow">Installation review</span>
            <h3 id="app-install-review-title">Install {selected.name}?</h3>
            <p>Vaelor will download <strong>{selected.image}</strong>, create persistent storage where required, and start it automatically after a restart.</p>
          </div>
          <div className="app-port-field">
            <Input
              aria-describedby={portProblem ? "app-host-port-error" : undefined}
              aria-invalid={Boolean(portProblem)}
              className="app-port-field__input"
              id="app-host-port"
              label="Web address port"
              max={65535}
              min={1024}
              onChange={(event) => setPort(event.target.value === "" ? Number.NaN : Number(event.target.value))}
              step={1}
              type="number"
              value={Number.isFinite(port) ? port : ""}
            />
            {portProblem && <small className="field-error" id="app-host-port-error" role="alert">{portProblem}</small>}
          </div>
          <div className="app-install-review__checks">
            <span><Icon name="shield" />No privileged access</span>
            <span><Icon name="memory" />{selected.memory} memory limit</span>
            <span><Icon name="database" />Managed by Vaelor</span>
          </div>
          <div className="agent-plan__actions">
            <Button disabled={busy} onClick={() => setSelected(null)} type="button" variant="secondary">Go back</Button>
            <Button disabled={busy || disabled || preflightPending || Boolean(portProblem)} onClick={() => void approveInstall()} type="button" variant="primary">
              {busy ? "Adding to setup..." : "Approve and install"}
            </Button>
          </div>
          {preflight?.conflict && preflight.suggested_port && (
            <Notice severity="warning">
              <span>{preflight.reason} Vaelor found port {preflight.suggested_port} available.</span>
              <Button type="button" variant="quiet" onClick={() => { setPort(preflight.suggested_port as number); setPreflight(null); }}>Use port {preflight.suggested_port}</Button>
            </Notice>
          )}
        </div>
      )}
    </section>
  );
}
