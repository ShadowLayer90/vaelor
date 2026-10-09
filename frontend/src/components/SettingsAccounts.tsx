import type { ReactNode } from "react";
import type { Role, Session } from "../types";
import { StatusPill } from "./StatusPill";
import { Button, Card, Input } from "./ui";
import "../styles/settings.css";

/*
 * Settings › Accounts (VD-200, the Settings board): the local accounts, the
 * form that adds one, and - beside them - your own two-factor card.
 * Administration reads the accounts and owns every request; this draws them.
 */

export interface ManagedUser {
  username: string;
  role: Role;
  enabled: boolean;
  created_at: number;
  active_sessions: number;
}

export const FINAL_ADMINISTRATOR_REASON =
  "This is the last enabled administrator. Add another enabled administrator before disabling, changing the role, or deleting this account.";
export const CURRENT_ACCOUNT_REASON = "You cannot disable or delete the account you are using.";

/** The three access levels, least first, in the words the form offers them. */
const ACCESS_LEVELS: ReadonlyArray<{ role: Role; name: string; detail: string }> = [
  { role: "viewer", name: "Viewer", detail: "Can see status" },
  { role: "operator", name: "Operator", detail: "Can control machines and deploy" },
  { role: "administrator", name: "Administrator", detail: "Can manage access" },
];

export interface NewAccountForm {
  username: string;
  password: string;
  role: Role;
  usernameError?: string;
  /** The lowercase name the account will be stored under, when that differs from what was typed. */
  normalizedName?: string;
  blockedReason?: string;
  busy: boolean;
  onUsername: (value: string) => void;
  onPassword: (value: string) => void;
  onRole: (value: Role) => void;
  onCreate: () => void;
}

function LocalAccounts({ busy, onDelete, onRoleChange, onToggle, session, users }: {
  busy: string;
  onDelete: (user: ManagedUser) => void;
  onRoleChange: (user: ManagedUser, role: Role) => void;
  onToggle: (user: ManagedUser) => void;
  session: Session;
  users: ManagedUser[];
}) {
  const enabledAdministrators = users.filter((user) => user.role === "administrator" && user.enabled).length;
  return (
    <Card
      actions={<span className="status-pill status-pill--neutral stg-count">{users.length} total</span>}
      as="section"
      className="stg-accounts"
      description="Disabling an account also ends its sessions."
      flush
      heading="Local accounts"
    >
      {users.map((user) => {
        const finalAdministrator = user.role === "administrator" && user.enabled && enabledAdministrators === 1;
        const isYou = user.username === session.user.username;
        const restrictionId = `admin-user-${user.username}-restriction`;
        const restricted = finalAdministrator || isYou;
        const rowBusy = busy === `user-${user.username}`;
        return (
          <div className="stg-row stg-row--account" key={user.username}>
            <span aria-hidden="true" className="stg-avatar">{user.username[0].toUpperCase()}</span>
            <div className="stg-row__text">
              <strong>{user.username}{isYou ? " · you" : ""}</strong>
              <small>{user.active_sessions === 0 ? "No active sessions" : `${user.active_sessions} active session${user.active_sessions === 1 ? "" : "s"}`}{user.enabled ? "" : " · disabled"}</small>
              {restricted && <small className="stg-row__reason" id={restrictionId}>{finalAdministrator ? FINAL_ADMINISTRATOR_REASON : CURRENT_ACCOUNT_REASON}</small>}
            </div>
            <div className="stg-row__controls record-card-button">
              <select
                aria-describedby={finalAdministrator ? restrictionId : undefined}
                aria-label={`${user.username} role`}
                className="ui-control ui-control--select stg-role"
                disabled={rowBusy || finalAdministrator}
                onChange={(event) => onRoleChange(user, event.target.value as Role)}
                value={user.role}
              >
                <option value="viewer">Viewer</option><option value="operator">Operator</option><option value="administrator">Administrator</option>
              </select>
              {/* #150: each action names its verb and its account; the
                  restriction text stays a description, never the name. */}
              <Button
                aria-describedby={restricted ? restrictionId : undefined}
                aria-label={`${user.enabled ? "Disable" : "Enable"} ${user.username}`}
                className="record-ghost"
                disabled={rowBusy || restricted}
                onClick={() => onToggle(user)}
                variant="quiet"
              >
                {user.enabled ? "Disable" : "Enable"}
              </Button>
              <Button
                aria-describedby={restricted ? restrictionId : undefined}
                aria-label={`Delete ${user.username}`}
                className="record-ghost stg-delete"
                disabled={rowBusy || restricted}
                onClick={() => onDelete(user)}
                variant="quiet"
              >
                Delete
              </Button>
            </div>
          </div>
        );
      })}
    </Card>
  );
}

function AddAccount({ form }: { form: NewAccountForm }) {
  return (
    <Card as="section" className="stg-add" description="Start with the least access the person needs." heading="Add an account">
      <form className="stg-form" onSubmit={(event) => { event.preventDefault(); if (!form.blockedReason && !form.busy) form.onCreate(); }}>
        <div className="stg-form__pair">
          {/*
            * #149: the Input primitive draws the red border, the error line and
            * the aria wiring, so an error is never byte-identical to the hint.
            */}
          <Input
            autoCapitalize="none"
            error={form.usernameError}
            hint="Use 2–64 letters, numbers, dots, dashes, or underscores. The name is saved in lowercase."
            id="account-username"
            label="Username"
            onChange={(event) => form.onUsername(event.target.value)}
            value={form.username}
          />
          <Input
            autoComplete="new-password"
            hint="Use at least 12 characters. This passphrase is not shown again."
            id="account-password"
            label="Temporary passphrase"
            maxLength={256}
            minLength={12}
            onChange={(event) => form.onPassword(event.target.value)}
            type="password"
            value={form.password}
          />
        </div>
        {form.normalizedName && <small className="ui-muted" id="account-username-normalized">This account will be created as {form.normalizedName}.</small>}
        <fieldset className="stg-levels">
          <legend>Access level</legend>
          {ACCESS_LEVELS.map((level) => (
            <label className={form.role === level.role ? "stg-level stg-level--on" : "stg-level"} key={level.role}>
              <input checked={form.role === level.role} className="stg-level__radio" name="account-role" onChange={() => form.onRole(level.role)} type="radio" value={level.role} />
              <span><strong>{level.name}</strong><small>{level.detail}</small></span>
            </label>
          ))}
        </fieldset>
        <div className="stg-form__end">
          {/* A disabled primary action says what is missing, beside it. */}
          <Button busy={form.busy} disabledReason={form.blockedReason} type="submit" variant="primary">
            {form.busy ? "Creating…" : "Create account"}
          </Button>
        </div>
      </form>
    </Card>
  );
}

export function SettingsAccounts({ busy, form, onDelete, onRoleChange, onToggle, session, twoFactor, users }: {
  busy: string;
  form: NewAccountForm;
  onDelete: (user: ManagedUser) => void;
  onRoleChange: (user: ManagedUser, role: Role) => void;
  onToggle: (user: ManagedUser) => void;
  session: Session;
  /** Your own two-factor card, drawn under the form. */
  twoFactor: ReactNode;
  users: ManagedUser[];
}) {
  return (
    <div className="stg-two-col">
      <LocalAccounts busy={busy} onDelete={onDelete} onRoleChange={onRoleChange} onToggle={onToggle} session={session} users={users} />
      <div className="stg-stack">
        <AddAccount form={form} />
        {twoFactor}
      </div>
    </div>
  );
}

/** The pill a read verdict gets: "Reading" until it was read, never a guess. */
export function ReadPill({ read, ok, okLabel, notLabel }: { read: boolean; ok: boolean; okLabel: string; notLabel: string }) {
  if (!read) return <StatusPill label="Reading" tone="neutral" />;
  return <StatusPill label={ok ? okLabel : notLabel} tone={ok ? "success" : "warning"} />;
}
