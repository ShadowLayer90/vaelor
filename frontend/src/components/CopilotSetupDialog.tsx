import type { ReactNode } from "react";
import { AppsDialog } from "./appsKit";
import { CopilotSetup, type CopilotSetupProps } from "./CopilotSetup";
import { Button } from "./ui";

/**
 * The Assistant setup as Apps and AI opens it (the AppsAssistantSetup board):
 * the area's one dialog around CopilotSetup's content. `error` is the refusal
 * of an action taken inside it (built-in basic mode), shown at the top of the
 * body because the page beneath is inert.
 */
export function CopilotSetupDialog({
  error,
  ...props
}: Omit<CopilotSetupProps, "showHeader"> & { error?: ReactNode }) {
  return (
    <AppsDialog
      error={error}
      eyebrow="Assistant setup"
      footer={<Button onClick={props.onClose}>Close</Button>}
      onClose={props.onClose}
      size="wide"
      title="Choose how Vaelor Assistant thinks"
      titleId="copilot-setup-title"
    >
      <CopilotSetup {...props} showHeader={false} />
    </AppsDialog>
  );
}
