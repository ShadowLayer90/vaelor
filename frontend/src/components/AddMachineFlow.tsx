import { useState, type ReactNode } from "react";
import { Button, Input, Notice } from "./ui";
import { ModalError } from "./ui/ModalError";
import { ClusterCard, IconTile, Steps, type StepState } from "./ClusterPrimitives";
import { ClusterDialog } from "./ClusterDialog";
import { apiRequest } from "../lib/api";
import type { Session } from "../types";
import type { ClusterMode } from "../lib/clusterMode";
import "../styles/cluster-setup.css";

/**
 * The guided "add a machine" flow (VD-200: the ClusterSetup and
 * ClusterDialogsGrow boards). It is the one place a worker is enrolled into
 * the cluster (joining the swarm is the separate, reviewed step that follows).
 * Inline on the Setup tab it is the "Add a machine" card; given `onClose` it
 * renders itself as the "Add a machine / Grow the cluster" dialog the header's
 * Add machine action opens, with its buttons in the dialog's footer.
 *
 * The flow and its wire calls are unchanged: inspect the host over
 * `POST /cluster/ssh-fingerprint`, show the operator the host-key fingerprint to
 * confirm, then enrol over `POST /cluster/nodes`. The password is handed to the
 * credential broker by the enrol call and never logged, stored in plain text, or
 * placed in a queued job.
 *
 * Easy asks only for what the join needs — address, SSH user, and the password
 * or key. Advanced additively reveals the friendly name and a non-default SSH
 * port (both already carried by the enrol payload). Role, labels, and data-root
 * are intentionally NOT rendered yet: the enrol endpoint does not consume them,
 * and this codebase does not show a control that cannot act (VD-009a).
 *
 * When enrollment cannot run, no field — above all no password field — is
 * shown at all. Collecting the most sensitive input in the product for an
 * operation known in advance to fail is the #75 defect, and this component
 * refuses to reproduce it.
 *
 * ACC-123. Enrolment only stores the sign-in and reads the machine's hardware;
 * nothing is installed and nothing joins the swarm. The final button says
 * that, and step 3 is where the owner lands after a successful enrolment: it
 * hands the new node id to the same reviewed "join-node" plan the
 * awaiting-join card uses, so every step the strip shows is reachable.
 * The address field suggests the prefix the controller derived from its own
 * address (`enrollment.address_hint`), never a literal host (VD-124).
 */

/** What the Address field suggests when the controller could not derive a hint. */
export const NEUTRAL_ADDRESS_PLACEHOLDER = "worker hostname or address";

/**
 * The Address placeholder. The backend sends the first three octets of this
 * controller's own IPv4 address with a trailing dot (for example "10.1.2."),
 * or "" when it has none; any other shape is ignored rather than shown, so a
 * whole address can never be suggested as if it were a remembered machine.
 */
export function addressPlaceholder(hint: string | undefined): string {
  const prefix = (hint ?? "").trim();
  return /^\d{1,3}\.\d{1,3}\.\d{1,3}\.$/.test(prefix)
    ? `${prefix}x or a hostname`
    : NEUTRAL_ADDRESS_PLACEHOLDER;
}

export function validSshHost(value: string): boolean {
  const host = value.trim();
  if (!host || host.length > 253) return false;
  return /^(?:[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?|[A-Fa-f0-9:]+)$/.test(host)
    && !host.includes("..");
}

export function validSshPort(value: string): boolean {
  return /^\d+$/.test(value) && Number(value) >= 1 && Number(value) <= 65535;
}

export function validSshUsername(value: string): boolean {
  return /^[A-Za-z0-9._-]{1,64}$/.test(value);
}

interface Props {
  session: Session;
  mode: ClusterMode;
  /**
   * Whether enrollment can run right now. When false, the flow shows only why,
   * and never a credential field (VD-009a / #75).
   */
  actionable: boolean;
  disabledReason?: string;
  /** Reload the fleet after a worker is enrolled. */
  onEnrolled: () => void | Promise<void>;
  /**
   * Open the reviewed join plan for the machine just enrolled (step 3). Both
   * render sites wire this to the same `reviewPlan("join-node", nodeId)` path
   * the awaiting-join card uses.
   */
  onReviewJoin: (nodeId: string) => void;
  /** `summary.enrollment.address_hint`: this controller's own subnet prefix, or "". */
  addressHint?: string;
  setNotice: (message: string) => void;
  /**
   * Given, the flow is a dialog and this closes it (Cancel, Close, or once
   * the join plan opens). Absent, the flow is the inline Setup card.
   */
  onClose?: () => void;
}

const STEP_LABELS = ["Connect", "Confirm identity", "Review the join"] as const;

/** The step strip's states for the step the flow is on (1, 2 or 3). */
export function stepStates(step: 1 | 2 | 3): StepState[] {
  return STEP_LABELS.map((_, index) => (index + 1 < step ? "done" : index + 1 === step ? "on" : "todo"));
}

export function AddMachineFlow({
  session,
  mode,
  actionable,
  disabledReason,
  onEnrolled,
  onReviewJoin,
  addressHint,
  setNotice,
  onClose,
}: Props) {
  const advanced = mode === "advanced";
  const inDialog = Boolean(onClose);
  const [form, setForm] = useState({
    name: "",
    host: "",
    port: "22",
    username: "",
    password: "",
  });
  const [fingerprint, setFingerprint] = useState("");
  const [fingerprintAlgorithm, setFingerprintAlgorithm] = useState("");
  const [busy, setBusy] = useState(false);
  // VD-189: a refused inspection or enrolment is shown in this flow, never on
  // the page: in the Add machine dialog that page is inert and aria-hidden.
  const [error, setError] = useState("");
  // The node the enrol call returned. Only a successful enrolment sets it, and
  // it is the only thing that reaches step 3.
  const [enrolled, setEnrolled] = useState<{ id: string; name: string } | null>(null);

  const hostValid = validSshHost(form.host);
  // The port field only shows in Advanced; Easy always joins on 22. Gating and
  // the sent value both read the effective port, so a stale invalid port typed
  // in Advanced can never silently disable Connect after a switch back to Easy.
  const sentPort = advanced ? form.port : "22";
  const portValid = advanced ? validSshPort(form.port) : true;
  const usernameValid = validSshUsername(form.username);
  const step: 1 | 2 | 3 = enrolled ? 3 : fingerprint ? 2 : 1;

  const reset = () => {
    setForm({ name: "", host: "", port: "22", username: "", password: "" });
    setFingerprint("");
    setFingerprintAlgorithm("");
    setEnrolled(null);
    setError("");
  };

  const reviewJoin = () => {
    if (!enrolled) return;
    onReviewJoin(enrolled.id);
    reset();
    onClose?.();
  };

  const cancel = () => {
    reset();
    onClose?.();
  };

  const inspect = async () => {
    setBusy(true);
    setNotice("");
    setError("");
    try {
      const result = await apiRequest<{ fingerprint: string; algorithm: string }>(
        "/cluster/ssh-fingerprint",
        { method: "POST", body: JSON.stringify({ host: form.host, port: Number(sentPort) }) },
        session.csrf_token,
      );
      setFingerprint(result.fingerprint);
      setFingerprintAlgorithm(result.algorithm);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "SSH inspection failed.");
    } finally {
      setBusy(false);
    }
  };

  const enroll = async () => {
    setBusy(true);
    setNotice("");
    setError("");
    try {
      const node = await apiRequest<{ id?: string; name?: string; host?: string }>(
        "/cluster/nodes",
        {
          method: "POST",
          body: JSON.stringify({
            ...form,
            port: Number(sentPort),
            host_key_fingerprint: fingerprint,
            sudo_uses_login_password: true,
          }),
        },
        session.csrf_token,
      );
      const enrolledName = node?.name || node?.host || form.host;
      // reset() first: it drops the password from component state.
      reset();
      await onEnrolled();
      if (node?.id) {
        setEnrolled({ id: node.id, name: enrolledName });
      } else {
        // A reply without an id cannot open a plan; the machine is still
        // listed under "Machines awaiting join", which is where to go next.
        setNotice(`${enrolledName} is enrolled. Review its join plan under Machines awaiting join.`);
        onClose?.();
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Worker enrollment failed.");
    } finally {
      setBusy(false);
    }
  };

  const steps = (
    <Steps
      label="Enrollment steps"
      steps={STEP_LABELS.map((label, index) => ({ label, state: stepStates(step)[index] }))}
    />
  );

  // When enrollment cannot run, no field at all - above all no password field.
  const refusedNotice = (
    <Notice severity="info">
      {disabledReason || "This node cannot enrol a worker yet. Set up the controller first."}
    </Notice>
  );

  const fields = (
    <>
      <div className="cl-setup__fields">
        {advanced && (
          <Input
            label="Friendly name"
            onChange={(event) => setForm({ ...form, name: event.target.value })}
            placeholder="mini-pc"
            value={form.name}
          />
        )}
        <Input
          aria-invalid={Boolean(form.host) && !hostValid}
          error={form.host && !hostValid ? "Enter an IP address or hostname without spaces or command characters." : undefined}
          hint="An IP address or hostname."
          label="Address"
          onChange={(event) => {
            setFingerprint("");
            setForm({ ...form, host: event.target.value });
          }}
          placeholder={addressPlaceholder(addressHint)}
          value={form.host}
        />
        <Input
          aria-invalid={Boolean(form.username) && !usernameValid}
          autoComplete="username"
          error={form.username && !usernameValid ? "Use up to 64 letters, numbers, dots, dashes, or underscores." : undefined}
          label="SSH user"
          onChange={(event) => setForm({ ...form, username: event.target.value })}
          placeholder="ubuntu"
          value={form.username}
        />
        {advanced && (
          <Input
            aria-invalid={!portValid}
            error={!portValid ? "Use an SSH port from 1 to 65535." : undefined}
            inputMode="numeric"
            label="SSH port"
            onChange={(event) => setForm({ ...form, port: event.target.value })}
            value={form.port}
          />
        )}
      </div>
      <Input
        autoComplete="new-password"
        hint="Stored encrypted on this controller - never in plain text and never in a job."
        label="Password or key"
        onChange={(event) => setForm({ ...form, password: event.target.value })}
        placeholder="Password or private key"
        type="password"
        value={form.password}
      />
    </>
  );

  const hostKey = (
    <div className="cl-setup__identity">
      <IconTile accent name="shield" />
      <div className="cl-setup__identity-text">
        <strong className="cl-strong">Confirm this host key</strong>
        <span className="cl-meta">{fingerprintAlgorithm}</span>
        <code className="cl-code">{fingerprint}</code>
        <p className="cl-meta">
          Compare this with the fingerprint shown on the worker or from{" "}
          <code className="cl-setup__mono">ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub</code>.
        </p>
      </div>
    </div>
  );

  let content: ReactNode;
  let footer: ReactNode;
  if (enrolled) {
    content = (
      <>
        {steps}
        <div className="cl-setup__identity" role="status">
          <IconTile accent name="cluster" />
          <div className="cl-setup__identity-text">
            <strong className="cl-strong">{enrolled.name} is enrolled</strong>
            <p className="cl-meta">
              Vaelor stores its sign-in, encrypted, on this controller, and its hardware was
              read. It is not part of the cluster yet: the join plan shows what joining changes on it,
              and nothing happens until you approve it. Until then it waits under Machines
              awaiting join.
            </p>
          </div>
        </div>
      </>
    );
    footer = (
      <>
        {inDialog && <Button variant="quiet" onClick={cancel}>Close</Button>}
        <Button variant="quiet" onClick={reset}>Add another machine</Button>
        <Button variant="primary" onClick={reviewJoin}>Review join plan</Button>
      </>
    );
  } else if (!actionable) {
    content = inDialog ? refusedNotice : (
      <>
        {refusedNotice}
        <p className="cl-meta">No address or password field is shown while enrolment cannot run.</p>
      </>
    );
    footer = inDialog ? <Button onClick={cancel}>Close</Button> : null;
  } else {
    content = <>{steps}{fields}{fingerprint && hostKey}</>;
    footer = (
      <>
        {inDialog && <Button onClick={cancel}>Cancel</Button>}
        {fingerprint ? (
          <Button
            variant="primary"
            disabled={busy}
            disabledReason={!busy && (!usernameValid || !form.password) ? "Enter the SSH user and its password or key first." : undefined}
            onClick={() => void enroll()}
          >
            {busy ? "Enrolling…" : "Fingerprint matches — enrol this machine"}
          </Button>
        ) : (
          <Button
            variant="primary"
            disabled={busy || (Boolean(form.host) && !hostValid) || !portValid}
            disabledReason={!busy && !form.host ? "Enter the machine's address first." : undefined}
            onClick={() => void inspect()}
          >
            {busy ? "Inspecting…" : "Connect and inspect identity"}
          </Button>
        )}
      </>
    );
  }

  if (onClose) {
    return (
      <ClusterDialog
        error={error}
        eyebrow="Add a machine"
        footer={footer}
        onClose={() => { if (!busy) cancel(); }}
        title="Grow the cluster"
      >
        {content}
      </ClusterDialog>
    );
  }

  return (
    <ClusterCard
      actions={<span className="cl-meta">Administrators only</span>}
      className="cl-setup__add"
      description="Grow the cluster"
      icon="add"
      title="Add a machine"
    >
      {actionable && !enrolled && (
        <p className="cl-setup__muted">
          Checks its identity, stores its sign-in encrypted on this controller and reads its
          hardware. Nothing is installed on it, and nothing changes until you approve the join plan.
        </p>
      )}
      {content}
      {footer && (
        <div className="cl-setup__add-footer">
          {actionable && !enrolled && !advanced && (
            <span className="cl-meta">Advanced also asks for a friendly name and SSH port.</span>
          )}
          <span className="cl-actions cl-actions--end">{footer}</span>
        </div>
      )}
      <ModalError error={error} />
    </ClusterCard>
  );
}
