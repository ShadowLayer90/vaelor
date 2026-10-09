import { useState } from "react";
import { canonicalOperationState, type JobProjectionInput } from "../lib/jobPresentation";
import { AppCatalog, type AppTemplate, type PortPreflight } from "./AppCatalog";
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
  error,
  disabled,
  disabledReason,
  onClose,
  onDismiss,
  onInstall,
  onPreflight,
  open,
  resume,
  templates,
  installedTemplateIds,
  onOpenInstalled,
  onRetry,
  readError,
}: {
  busy: boolean;
  /** Why the install was refused (VD-189): shown in the catalog, never on the inert page beneath. */
  error?: string;
  disabled: boolean;
  /** Why installing is held (viewer, Docker not ready), in the page's words. */
  disabledReason?: string;
  onClose: () => void;
  onDismiss: () => void;
  onInstall: (template: AppTemplate, port: number) => void | Promise<void>;
  onPreflight?: (port: number) => Promise<PortPreflight>;
  open: boolean;
  resume: CatalogResume | null;
  templates: AppTemplate[];
  installedTemplateIds?: string[];
  onOpenInstalled?: () => void;
  /** Read `/apps/catalog` again after it could not be read. */
  onRetry?: () => void;
  /** True when `/apps/catalog` could not be read: the dialog says so instead of an empty grid. */
  readError?: boolean;
}) {
  if (!open) return null;
  // The catalog draws its own dialog (AppsDialog): its title and footer
  // change between the grid and the install review. Escape and the backdrop
  // keep a resumed choice (onDismiss); Close clears it (onClose).
  return (
    <AppCatalog
      busy={busy}
      disabled={disabled}
      disabledReason={disabledReason}
      error={error}
      initialPort={resume?.port}
      initialTemplateId={resume?.templateId}
      installedTemplateIds={installedTemplateIds}
      onClose={onClose}
      onDismiss={onDismiss}
      onInstall={onInstall}
      onOpenInstalled={onOpenInstalled}
      onPreflight={onPreflight}
      onRetry={onRetry}
      readError={readError}
      templates={templates}
    />
  );
}
