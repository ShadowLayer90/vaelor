import { Button } from "./ui";
import { Component, type ErrorInfo, type ReactNode } from "react";
import { brand } from "../lib/brand";
import { Icon } from "./Icon";
import { ProductMark } from "./ProductMark";
import "../styles/notices.css";

interface WorkspaceErrorBoundaryProps {
  children: ReactNode;
  /** Omitted for the app-wide boundary (UX-A13): there is no overview to go back to. */
  onBack?: () => void;
  onReload?: () => void;
  /**
   * The app-wide boundary's way out (LESSONS 22): a crash caused by the
   * account's own data repeats on every reload, so the owner can sign out.
   */
  onSignOut?: () => void;
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

    // The GlobalNotices board (VD-200): one page failed - a red-edged card in
    // the page's place, the rail still working - or the whole console, which
    // stands alone under the brand mark with Sign out beside Reload.
    const page = Boolean(this.props.onBack);
    return (
      <div className={page ? "gn-error-wrap" : "gn-error-wrap gn-error-wrap--app"}>
        <section aria-labelledby="workspace-error-title" className={page ? "workspace-error gn-error" : "workspace-error gn-error gn-error--app"} role="alert">
          {!page && <div className="gn-error__brand"><span aria-hidden="true" className="gn-error__mark"><ProductMark /></span><strong>{brand.name}</strong></div>}
          <span className="gn-error__eyebrow">{page ? "Workspace unavailable" : "Control plane unavailable"}</span>
          <h1 id="workspace-error-title">{page ? "This page could not be displayed" : "Vaelor could not display this page"}</h1>
          <p>
            {page
              ? "Vaelor kept the rest of the control plane available. Try this workspace again, or return to the overview and continue elsewhere."
              : "Nothing on the appliance was changed. Reload the control plane; if this page fails again, the technical details below say what failed."}
          </p>
          <div className="gn-error__actions">
            <Button variant="primary" onClick={this.reload} type="button">
              Reload control plane
            </Button>
            {this.props.onBack && (
              <Button className="record-ghost" variant="quiet" onClick={this.props.onBack} type="button">
                Back to overview
              </Button>
            )}
            {this.props.onSignOut && (
              <Button className="record-ghost" variant="quiet" onClick={this.props.onSignOut} type="button">
                <Icon name="logout" size={16} />Sign out
              </Button>
            )}
          </div>
          <details open={page || undefined}>
            <summary>Technical details</summary>
            <code>{error.message || "Unexpected workspace rendering error"}</code>
          </details>
          {this.props.onSignOut && <small className="gn-error__note">Sign out is offered here because a failure caused by this account's own data comes back on every reload.</small>}
        </section>
      </div>
    );
  }
}