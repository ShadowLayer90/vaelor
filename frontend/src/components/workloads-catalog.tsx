import { useState } from "react";
import { canonicalOperationState, type JobProjectionInput } from "../lib/jobPresentation";
import { AppCatalog, type AppTemplate, type PortPreflight } from "./AppCatalog";
import { ModalShell } from "./ModalShell";
import type { WorkloadJob } from "./workloads-types";

export interface CatalogResume {
  templateId: string;
  port: number;
}

export function catalogFailureNeedsPortChange(job: Pick<WorkloadJob, "type" | "message" | "payload"> & JobProjectionInput) {
  return (
    job.type === "compose.install"
    && canonicalOperationState(job) === "failed"
    && typeof job.payload?.template === "string"
    && typeof job.payload?.port === "number"
    && /\bport\b/i.test(job.message)
  );
}

export function useWorkloadCatalogState() {
  const [catalogResume, setCatalogResume] = useState<CatalogResume | null>(null);

  return {
    catalogResume,
    clearCatalogResume: () => setCatalogResume(null),
    resumeCatalog: (job: WorkloadJob) => setCatalogResume({
      templateId: String(job.payload?.template),
      port: Number(job.payload?.port),
    }),
  };
}

export function WorkloadCatalogModal({
  busy,
  disabled,
  onClose,
  onDismiss,
  onInstall,
  onPreflight,
  open,
  resume,
  templates,
  installedTemplateIds,
  onOpenInstalled,
}: {
  busy: boolean;
  disabled: boolean;
  onClose: () => void;
  onDismiss: () => void;
  onInstall: (template: AppTemplate, port: number) => void | Promise<void>;
  onPreflight?: (port: number) => Promise<PortPreflight>;
  open: boolean;
  resume: CatalogResume | null;
  templates: AppTemplate[];
  installedTemplateIds?: string[];
  onOpenInstalled?: () => void;
}) {
  if (!open) return null;
  return (
    <ModalShell labelledBy="app-catalog-title" onClose={onDismiss}>
      <AppCatalog
        busy={busy}
        disabled={disabled}
        initialPort={resume?.port}
        initialTemplateId={resume?.templateId}
        onClose={onClose}
        onInstall={onInstall}
        onPreflight={onPreflight}
        templates={templates}
        installedTemplateIds={installedTemplateIds}
        onOpenInstalled={onOpenInstalled}
      />
    </ModalShell>
  );
}
