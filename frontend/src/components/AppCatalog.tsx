import { useEffect, useRef, useState, type ReactNode } from "react";
import "../styles/apps-setup.css";
import { AppsDialog, AppsFacts, AppsIconTile } from "./appsKit";
import { AppsBanner } from "./appsSetupParts";
import { Icon, ICON_SIZE } from "./Icon";
import { Button, Input } from "./ui";

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
  /**
   * The port an install would be offered (W7-D2): the default, or the port the
   * backend suggests when a stored model holds it. Absent on older servers.
   */
  offered_port?: number | null;
  /** What holds the default port, when something does. */
  default_port_holder?: string;
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

/** The port a template's install would get: the backend's offer, else its default (W7-D2). */
export function offeredPort(template: AppTemplate): number {
  return template.offered_port ?? template.default_port;
}

/** Why the catalog's install buttons are held when the caller names no reason. */
const INSTALL_HELD = "Installing needs operator access and Docker ready on this machine.";

function clientPortProblem(port: number) {
  if (!Number.isFinite(port) || !Number.isInteger(port)) return "Enter a whole-number port from 1024 to 65535.";
  if (port < 1024) return "Use a port from 1024 to 65535. Lower ports are reserved by the operating system.";
  if (port > 65535) return "Ports cannot be higher than 65535.";
  if ([34001, 34002].includes(port)) return "That port is reserved for the Vaelor control plane.";
  return "";
}

/** One blueprint as the AppsCatalog board draws it: tile and category, name, line, facts, one action. */
function TemplateCard({
  disabledReason,
  installed,
  onChoose,
  onOpenInstalled,
  template,
}: {
  disabledReason?: string;
  installed: boolean;
  onChoose: () => void;
  onOpenInstalled?: () => void;
  template: AppTemplate;
}) {
  const held = template.default_port_holder;
  return (
    <article className="apps-catalog-card">
      <div className="apps-catalog-card__top">
        <AppsIconTile name="apps" />
        <span className="apps-catalog-card__category">{template.category}</span>
      </div>
      <h3>{template.name}</h3>
      <p>{template.description}</p>
      <AppsFacts
        rows={[
          { label: "Memory limit", value: template.memory },
          { label: "Storage", value: template.storage },
          { label: held ? "Address offered" : "Default address", value: `Port ${offeredPort(template)}` },
          ...(held ? [{ label: "Default port", value: `${template.default_port} is held by ${held}` }] : []),
        ]}
      />
      <div className="apps-catalog-card__action">
        {installed ? (
          <Button
            disabledReason={onOpenInstalled ? undefined : "Manage cannot be opened from here."}
            onClick={() => onOpenInstalled?.()}
          >
            View in Manage
          </Button>
        ) : (
          <Button disabledReason={disabledReason} onClick={onChoose}>Review installation</Button>
        )}
      </div>
    </article>
  );
}

export function AppCatalog({
  templates,
  busy,
  disabled,
  disabledReason,
  error,
  onClose,
  onDismiss,
  onInstall,
  onPreflight,
  initialTemplateId,
  initialPort,
  installedTemplateIds = [],
  onOpenInstalled,
  onRetry,
  readError = false,
}: {
  templates: AppTemplate[];
  busy: boolean;
  disabled: boolean;
  /** Why installing is held, in the caller's words; a general sentence otherwise. */
  disabledReason?: string;
  /** Why the install was refused (VD-189): shown inside the dialog, never on the inert page beneath. */
  error?: ReactNode;
  onClose: () => void;
  /** Escape, the backdrop and the header's Close: leave without clearing a resumed choice. Defaults to onClose. */
  onDismiss?: () => void;
  onInstall: (template: AppTemplate, port: number) => void | Promise<void>;
  onPreflight?: (port: number) => Promise<PortPreflight>;
  initialTemplateId?: string;
  initialPort?: number;
  installedTemplateIds?: string[];
  onOpenInstalled?: () => void;
  /** Read the blueprint list again after it could not be read. */
  onRetry?: () => void;
  /** The blueprint list could not be read: said, never drawn as an empty catalog (LESSONS 8). */
  readError?: boolean;
}) {
  const allInstalled = !readError && templates.length > 0 && templates.every((template) => installedTemplateIds.includes(template.id));
  const initialTemplate = templates.find((template) => template.id === initialTemplateId) ?? null;
  const [selected, setSelected] = useState<AppTemplate | null>(initialTemplate);
  const [port, setPort] = useState(initialPort ?? (initialTemplate ? offeredPort(initialTemplate) : 3000));
  const [preflight, setPreflight] = useState<PortPreflight | null>(null);
  const [preflightPending, setPreflightPending] = useState(false);
  const installInFlight = useRef(false);
  // W4d-D10: the review replaces the grid and takes focus. It used to render
  // under a 2,000 px grid with nothing moving, so the button looked dead.
  useEffect(() => {
    if (!selected) return;
    const heading = document.getElementById("app-install-review-title");
    heading?.setAttribute("tabindex", "-1");
    heading?.focus();
  }, [selected]);
  const localProblem = clientPortProblem(port);
  const conflict = preflight?.conflict ? preflight.reason : "";
  const portProblem = localProblem || conflict;
  const heldReason = disabled ? disabledReason || INSTALL_HELD : undefined;

  useEffect(() => {
    if (!selected || localProblem || !onPreflight) {
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
  }, [localProblem, onPreflight, port, selected]);

  const choose = (template: AppTemplate) => {
    setSelected(template);
    setPort(offeredPort(template));
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

  if (selected) {
    // A conflict is said under the field; the free port Vaelor found is
    // offered beside it, and Approve names what it is waiting for.
    const approveReason = heldReason
      ?? (conflict ? "Choose a free port first." : localProblem ? "Enter a usable port first." : undefined);
    return (
      <AppsDialog
        busy={busy}
        error={error}
        eyebrow="Installation review"
        footer={(
          <>
            <Button disabled={busy} onClick={() => setSelected(null)}>Go back</Button>
            <Button
              disabled={busy || preflightPending}
              disabledReason={busy ? undefined : approveReason}
              onClick={() => void approveInstall()}
              variant="primary"
            >
              {busy ? "Adding to setup..." : "Approve and install"}
            </Button>
          </>
        )}
        onClose={onDismiss ?? onClose}
        title={`Install ${selected.name}?`}
        titleId="app-install-review-title"
      >
        <p>Vaelor will download <code className="apps-setup-code">{selected.image}</code>, create persistent storage where required, and start it automatically after a restart.</p>
        <div className="apps-catalog-port">
          <Input
            aria-describedby={portProblem ? "app-host-port-error" : preflightPending ? "app-host-port-checking" : undefined}
            aria-invalid={Boolean(portProblem)}
            className="apps-catalog-port__input"
            id="app-host-port"
            label="Web address port"
            max={65535}
            min={1024}
            onChange={(event) => setPort(event.target.value === "" ? Number.NaN : Number(event.target.value))}
            step={1}
            type="number"
            value={Number.isFinite(port) ? port : ""}
          />
          {portProblem
            ? <small className="apps-field-error" id="app-host-port-error" role="alert">{portProblem}</small>
            : preflightPending && <small className="apps-field-hint" id="app-host-port-checking">Checking that port {port} is free…</small>}
        </div>
        {preflight?.conflict && preflight.suggested_port && (
          <AppsBanner
            action={<Button onClick={() => { setPort(preflight.suggested_port as number); setPreflight(null); }}>Use port {preflight.suggested_port}</Button>}
            tone="warning"
          >
            {preflight.reason} Vaelor found port {preflight.suggested_port} available.
          </AppsBanner>
        )}
        <ul aria-label="What this install is allowed" className="apps-setup-chips">
          <li><Icon aria-hidden="true" name="shield" size={ICON_SIZE.inline} />No privileged access</li>
          <li><Icon aria-hidden="true" name="memory" size={ICON_SIZE.inline} />{selected.memory} memory limit</li>
          <li><Icon aria-hidden="true" name="database" size={ICON_SIZE.inline} />Managed by Vaelor</li>
        </ul>
      </AppsDialog>
    );
  }

  const managedCount = templates.filter((template) => installedTemplateIds.includes(template.id)).length;
  return (
    <AppsDialog
      error={error}
      eyebrow="App catalog"
      footer={<Button onClick={onClose}>Close</Button>}
      footerStart={!readError && templates.length > 0
        ? <span>{templates.length} blueprint{templates.length === 1 ? "" : "s"}{managedCount ? ` · ${managedCount} already managed` : ""}</span>
        : undefined}
      onClose={onDismiss ?? onClose}
      size="wide"
      title={allInstalled ? "All catalog apps are managed" : "Choose an app"}
      titleId="app-catalog-title"
    >
      {readError ? (
        <AppsBanner action={onRetry && <Button onClick={onRetry}>Try again</Button>} tone="warning">
          The blueprint list could not be read. This is not a claim that there are none.
        </AppsBanner>
      ) : (
        <>
          <p>{allInstalled
            ? "Every available blueprint already has a managed instance. View an app in Manage to open it, inspect health, or change its setup."
            : "Every template uses resource limits, managed storage, and a configuration Vaelor can back up and repair."}</p>
          {templates.length === 0 && <p>This appliance offers no blueprints.</p>}
          <div className="apps-catalog-grid">
            {templates.map((template) => (
              <TemplateCard
                disabledReason={heldReason}
                installed={installedTemplateIds.includes(template.id)}
                key={template.id}
                onChoose={() => choose(template)}
                onOpenInstalled={onOpenInstalled}
                template={template}
              />
            ))}
          </div>
        </>
      )}
    </AppsDialog>
  );
}
