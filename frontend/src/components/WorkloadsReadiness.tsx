import { useId, useState } from "react";
import { AppsIconTile, AppsKv } from "./appsKit";
import { StatusPill } from "./StatusPill";
import { Button } from "./ui";
import type { StatusTone } from "./ui/status";
import type { WorkloadCapabilities } from "./workloads-types";
import "../styles/apps-install.css";

/*
 * Whether this machine can run apps, as the setup check found it (VD-200, the
 * Apps and AppsInstallStates boards). One derivation feeds the readiness card,
 * the Docker pill beside the tab strip and the install cards' disabled
 * reasons, so the three can never disagree about the same check.
 */

export interface DockerReadiness {
  /** The check has answered at all. Until it has, nothing here is a verdict. */
  read: boolean;
  composeReady: boolean;
  installAvailable: boolean;
  repairAvailable: boolean;
  pill: { label: string; tone: StatusTone; reading?: "unread" };
  title: string;
  detail: string;
}

export function dockerReadiness(capabilities: WorkloadCapabilities | null, checkFailed: boolean): DockerReadiness {
  // Docker installed and answering is not Docker able to start containers. A
  // wiped/broken data-root leaves the daemon up but every `docker run` failing;
  // the backend reports it as runtime "storage_missing". A node in that state is
  // NOT ready to run apps, and the fix is Repair, not Install (docker_health.py).
  // Any other value - "ok", "unknown", or missing (older backend) - is treated
  // as healthy, so this never cries wolf on a fine box whose control-plane user
  // simply cannot reach the Docker socket.
  const runtimeBroken = capabilities?.docker.runtime === "storage_missing";
  const composeReady = Boolean(capabilities?.docker.installed && capabilities.docker.compose && !runtimeBroken);
  // Mutually exclusive: repair needs Docker installed, install needs it absent.
  const installAvailable = Boolean(capabilities && !capabilities.docker.installed && capabilities.docker.installation?.available);
  const repairAvailable = Boolean(capabilities?.docker.installed && runtimeBroken);
  if (!capabilities) {
    /*
     * A check that has not answered is not a machine without Docker. Saying so
     * is the whole of task #120: for about a minute after every restart this
     * read "Container apps are unavailable on this OS" on an appliance running
     * five containers, and offered to install Docker over the top of it.
     */
    return {
      read: false, composeReady, installAvailable, repairAvailable,
      pill: checkFailed ? { label: "Check interrupted", tone: "warning" } : { label: "Checking", tone: "neutral", reading: "unread" },
      title: "Checking whether this node can run apps",
      detail: "Nothing has been established yet, so nothing here is a verdict about this machine.",
    };
  }
  const osName = capabilities.os?.name ?? "this host";
  if (composeReady) {
    return {
      read: true, composeReady, installAvailable, repairAvailable,
      pill: { label: "Docker ready", tone: "success" },
      title: "This machine is ready to run apps",
      detail: "No terminal or Linux commands required.",
    };
  }
  if (repairAvailable) {
    return {
      read: true, composeReady, installAvailable, repairAvailable,
      pill: { label: "Docker needs repair", tone: "warning" },
      title: "Docker needs repair on this node",
      detail: capabilities.docker.runtime_reason ?? "Docker's storage is not healthy.",
    };
  }
  const integration = capabilities.docker.installation?.reason || "Checking the operating-system integration.";
  if (installAvailable) {
    return {
      read: true, composeReady, installAvailable, repairAvailable,
      pill: { label: "Docker setup available", tone: "neutral" },
      title: `Docker is not installed on ${osName}`,
      detail: integration,
    };
  }
  return {
    read: true, composeReady, installAvailable, repairAvailable,
    pill: { label: "OS limits apply", tone: "neutral" },
    title: capabilities.docker.installed ? "Docker Compose is not ready" : "Container apps are unavailable on this OS",
    detail: integration,
  };
}

/** The five facts the setup check reads; a fact it did not return says "Not read". */
function systemFacts(capabilities: WorkloadCapabilities | null) {
  const notRead = "Not read";
  return [
    {
      label: "App engine",
      value: capabilities?.docker.compose_version
        ? `Docker Compose ${capabilities.docker.compose_version}`
        : capabilities ? "Not ready" : notRead,
    },
    { label: "Operating system", value: capabilities?.os?.name ?? notRead },
    { label: "Processor", value: capabilities?.runtime?.architecture ?? notRead },
    { label: "App storage", value: capabilities?.roots?.workloads ?? notRead },
    { label: "Model storage", value: capabilities?.roots?.models ?? notRead },
  ];
}

export function WorkloadsReadinessCard({
  administrator,
  busy,
  capabilities,
  checkedAt,
  readiness,
  onInstallDocker,
  onRepairDocker,
}: {
  administrator: boolean;
  busy: boolean;
  capabilities: WorkloadCapabilities | null;
  /** When the setup check last answered; null while it never has. */
  checkedAt: string | null;
  readiness: DockerReadiness;
  onInstallDocker: () => void;
  onRepairDocker: () => void;
}) {
  const [open, setOpen] = useState(false);
  const detailsId = "apps-readiness-" + useId().replaceAll(":", "");
  const icon = !readiness.read ? "refresh" : readiness.composeReady ? "done" : "alert";
  return (
    <section aria-label="App setup readiness" className="apps-readiness" data-state={readiness.read ? (readiness.composeReady ? "ready" : "not-ready") : "checking"}>
      <div className="apps-readiness__main">
        <AppsIconTile accent={readiness.composeReady} name={icon} />
        <div className="apps-readiness__text">
          <strong>{readiness.title}</strong>
          <span>{readiness.detail}</span>
        </div>
        <div className="apps-readiness__actions">
          {/* Repair and Install are offered to administrators only, and never together. */}
          {administrator && readiness.repairAvailable && (
            <Button busy={busy} onClick={onRepairDocker} variant="primary">{busy ? "Preparing repair…" : "Repair Docker"}</Button>
          )}
          {administrator && readiness.installAvailable && (
            <Button busy={busy} onClick={onInstallDocker} variant="primary">{busy ? "Preparing setup…" : "Review and install Docker"}</Button>
          )}
          <Button
            aria-controls={detailsId}
            aria-expanded={open}
            className="apps-readiness__toggle"
            onClick={() => setOpen((value) => !value)}
            variant="quiet"
          >
            Advanced system details
          </Button>
        </div>
      </div>
      {open && (
        <div className="apps-readiness__details" id={detailsId}>
          <span className="apps-readiness__details-label">
            Advanced system details · {checkedAt && capabilities ? `read at ${checkedAt}` : "the setup check has not answered"}
          </span>
          <AppsKv cells={systemFacts(capabilities)} label="Advanced system details" />
        </div>
      )}
    </section>
  );
}

/** The Docker pill the boards draw in the top bar, beside the tab strip here. */
export function DockerPill({ readiness }: { readiness: DockerReadiness }) {
  return <StatusPill label={readiness.pill.label} reading={readiness.pill.reading} tone={readiness.pill.tone} />;
}
