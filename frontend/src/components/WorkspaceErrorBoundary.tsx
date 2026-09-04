import { Button } from "./ui";
import { Component, type ErrorInfo, type ReactNode } from "react";

interface WorkspaceErrorBoundaryProps {
  children: ReactNode;
  onBack: () => void;
  onReload?: () => void;
  workspaceKey: string;
}

interface WorkspaceErrorBoundaryState {
  error: Error | null;
}

export class WorkspaceErrorBoundary extends Component<
  WorkspaceErrorBoundaryProps,
  WorkspaceErrorBoundaryState
> {
  state: WorkspaceErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): WorkspaceErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Workspace failed to render", {
      error,
      componentStack: info.componentStack,
      workspace: this.props.workspaceKey,
    });
  }

  componentDidUpdate(previousProps: WorkspaceErrorBoundaryProps) {
    if (this.state.error && previousProps.workspaceKey !== this.props.workspaceKey) {
      this.setState({ error: null });
    }
  }

  private reload = () => (this.props.onReload ?? (() => window.location.reload()))();

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <section className="workspace-error" role="alert" aria-labelledby="workspace-error-title">
        <span className="page-eyebrow">Workspace unavailable</span>
        <h1 id="workspace-error-title">This page could not be displayed</h1>
        <p>
          Vaelor kept the rest of the control plane available. Try this workspace again,
          or return to the overview and continue elsewhere.
        </p>
        <div className="workspace-error__actions">
          <Button variant="primary" onClick={this.reload} type="button">
            Reload control plane
          </Button>
          <Button variant="quiet" onClick={this.props.onBack} type="button">
            Back to overview
          </Button>
        </div>
        <details>
          <summary>Technical details</summary>
          <code>{error.message || "Unexpected workspace rendering error"}</code>
        </details>
      </section>
    );
  }
}