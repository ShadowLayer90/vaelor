import { useState, type FormEvent } from "react";
import { timeAgo } from "../lib/format";
import { useAlertChannels } from "../hooks/useAlertChannels";
import type { AlertChannel } from "./agentTypes";
import { ConfirmDialog } from "./ConfirmDialog";
import { Icon, ICON_SIZE } from "./Icon";
import { ModalShell } from "./ModalShell";
import { type CreateIn, useCreateIn } from "./routinesCreateIn";
import { StatusPill } from "./StatusPill";
import { Button, EmptyState, Input, Notice, SegmentedControl, Select, type StatusTone } from "./ui";

type EmailProvider = {
  label: string;
  host: string;
  port: string;
  security: string;
  secretLabel: string;
  secretHelp: string;
  fromLabel: string;
  fromHint: string;
  fixedUsername?: string;
};

// Known providers fill in the server, port, and encryption for the person, so
// setting up email is "pick your provider, type your address and an app
// password". "custom" reveals the raw SMTP fields for anything not listed.
const EMAIL_PROVIDERS: Record<string, EmailProvider> = {
  gmail: {
    label: "Gmail", host: "smtp.gmail.com", port: "587", security: "starttls",
    secretLabel: "App password",
    secretHelp: "Gmail needs an app password - NOT your normal password. Turn on 2-Step Verification, then create one under Google Account -> Security -> App passwords.",
    fromLabel: "Your Gmail address",
    fromHint: "e.g. you@gmail.com. Alerts are sent from, and signed in to, this mailbox.",
  },
  outlook: {
    label: "Outlook / Microsoft 365", host: "smtp.office365.com", port: "587", security: "starttls",
    secretLabel: "App password",
    secretHelp: "Outlook needs an app password. Create one under your Microsoft account -> Security -> Advanced security options -> App passwords.",
    fromLabel: "Your Outlook address",
    fromHint: "e.g. you@outlook.com or you@yourcompany.com.",
  },
  yahoo: {
    label: "Yahoo Mail", host: "smtp.mail.yahoo.com", port: "587", security: "starttls",
    secretLabel: "App password",
    secretHelp: "Yahoo needs an app password. Create one under Yahoo Account -> Account security -> Generate app password.",
    fromLabel: "Your Yahoo address",
    fromHint: "e.g. you@yahoo.com.",
  },
  icloud: {
    label: "iCloud Mail", host: "smtp.mail.me.com", port: "587", security: "starttls",
    secretLabel: "App-specific password",
    secretHelp: "iCloud needs an app-specific password. Create one at appleid.apple.com -> Sign-In and Security -> App-Specific Passwords.",
    fromLabel: "Your iCloud address",
    fromHint: "e.g. you@icloud.com.",
  },
  sendgrid: {
    label: "SendGrid", host: "smtp.sendgrid.net", port: "587", security: "starttls",
    secretLabel: "API key", fixedUsername: "apikey",
    secretHelp: "Paste a SendGrid API key as the password - the login name is always \"apikey\". The address below must be a verified sender in SendGrid.",
    fromLabel: "Verified sender address",
    fromHint: "The From address you verified in SendGrid.",
  },
};

const EMPTY_FORM = {
  kind: "email" as "email" | "webhook",
  provider: "gmail",
  name: "",
  // Guided email path
  emailAddress: "",
  toAddress: "",
  secret: "",
  // Custom SMTP path
  smtpHost: "",
  smtpPort: "587",
  security: "starttls",
  fromAddress: "",
  username: "",
  // Webhook path
  url: "",
  authHeader: "Authorization",
};

type ChannelForm = typeof EMPTY_FORM;

function deliveryTone(status: string): StatusTone {
  if (status === "delivered") return "success";
  if (status === "failed") return "danger";
  return "neutral";
}

function deliveryLabel(channel: AlertChannel): string {
  if (channel.last_delivery_status === "delivered") return "last delivery ok";
  if (channel.last_delivery_status === "failed") return "last delivery failed";
  return "never delivered";
}

// A blank name is filled in for the person so "Name" never blocks setup.
function channelName(form: ChannelForm): string {
  const named = form.name.trim();
  if (named) return named.slice(0, 100);
  if (form.kind === "webhook") return "Webhook alert";
  if (form.provider === "custom") return (form.toAddress.trim() || "Email alert").slice(0, 100);
  return `${EMAIL_PROVIDERS[form.provider].label} alert`;
}

function buildBody(form: ChannelForm): Record<string, unknown> {
  const shared = { kind: form.kind, name: channelName(form), secret: form.secret };
  if (form.kind === "webhook") {
    return { ...shared, url: form.url.trim(), auth_header: form.authHeader };
  }
  if (form.provider === "custom") {
    return {
      ...shared,
      smtp_host: form.smtpHost, smtp_port: Number(form.smtpPort) || 0, security: form.security,
      from_address: form.fromAddress, to_address: form.toAddress, username: form.username,
    };
  }
  const preset = EMAIL_PROVIDERS[form.provider];
  const email = form.emailAddress.trim();
  return {
    ...shared,
    smtp_host: preset.host, smtp_port: Number(preset.port), security: preset.security,
    from_address: email,
    to_address: form.toAddress.trim() || email,
    username: preset.fixedUsername ?? email,
  };
}

function isFormValid(form: ChannelForm): boolean {
  if (form.kind === "webhook") return /^https?:\/\//.test(form.url.trim());
  if (form.provider === "custom") {
    return Boolean(form.smtpHost.trim() && form.fromAddress.trim() && form.toAddress.trim());
  }
  // A listed provider always needs a login, so require the address + secret.
  return Boolean(form.emailAddress.trim() && form.secret.trim());
}

/**
 * Where a fired alert is delivered, configured beside the rules that fire it.
 *
 * Guided by design: a person picks their email provider and the server, port,
 * and encryption are filled in for them; only "Other" exposes raw SMTP. Secrets
 * are typed here but never read back - the server stores them in the credential
 * broker and returns only whether one is held.
 *
 * `createIn` says where the Add channel form lives: "inline" (the default;
 * Cluster > Activity) inside this card, "dialog" (the Assistant's Schedules and
 * alerts view) behind the card's Add channel button.
 *
 * `headless` draws the contents without the panel's own card and header, for
 * a page that frames it in its own card (Cluster > Activity, a third of the
 * row on the ClusterActivityAlerts board). The delete confirmation is the same
 * alertdialog either way.
 */
export function AlertChannelsPanel({
  csrfToken,
  canManage,
  createIn: createInProp,
  headless = false,
}: {
  csrfToken: string;
  canManage: boolean;
  /** Defaults to the Routines tab's choice (dialog), else inline. */
  createIn?: CreateIn;
  /** Contents only: the placing page draws the card, its icon and its title. */
  headless?: boolean;
}) {
  const createIn = useCreateIn(createInProp);
  const { channels, busy, notice, noticeRefused, createChannel, toggleChannel, deleteChannel, testChannel } =
    useAlertChannels({ csrfToken, enabled: canManage });
  const [form, setForm] = useState<ChannelForm>(EMPTY_FORM);
  const [pendingDelete, setPendingDelete] = useState<AlertChannel | null>(null);
  const [adding, setAdding] = useState(false);
  // The add dialog shows the refusal of its own submission only, never a
  // notice left over from an earlier action on the page (VD-189).
  const [submitted, setSubmitted] = useState(false);

  if (!canManage) return null;

  const inDialog = createIn === "dialog";
  const update = (patch: Partial<ChannelForm>) => setForm((current) => ({ ...current, ...patch }));
  const preset = form.provider === "custom" ? null : EMAIL_PROVIDERS[form.provider];

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitted(true);
    if (await createChannel(buildBody(form))) {
      setForm({ ...EMPTY_FORM, kind: form.kind, provider: form.provider });
      setAdding(false);
      setSubmitted(false);
    }
  };
  const openAdd = () => { setSubmitted(false); setAdding(true); };

  const fields = (
    <>
      {inDialog ? (
        <div className="ar-field">
          <span className="ar-field__label">How should we reach you?</span>
          <SegmentedControl label="How should we reach you?" onChange={(kind) => update({ kind })} options={[{ value: "email", label: "Email" }, { value: "webhook", label: "Slack, Discord, or other webhook" }]} value={form.kind} />
        </div>
      ) : (
        <Select id="channel-kind" label="How should we reach you?" onChange={(event) => update({ kind: event.target.value as ChannelForm["kind"] })} value={form.kind}>
          <option value="email">Email</option>
          <option value="webhook">Slack, Discord, or other webhook</option>
        </Select>
      )}
      {form.kind === "email" ? (
        <>
          <Select id="channel-provider" label="Email provider" onChange={(event) => update({ provider: event.target.value })} value={form.provider}>
            {Object.entries(EMAIL_PROVIDERS).map(([key, value]) => (
              <option key={key} value={key}>{value.label}</option>
            ))}
            <option value="custom">Other (enter server settings)</option>
          </Select>
          {preset ? (
            <>
              <Input hint={preset.fromHint} id="channel-email" label={preset.fromLabel} maxLength={255} onChange={(event) => update({ emailAddress: event.target.value })} type="email" value={form.emailAddress} />
              <Input hint={preset.secretHelp} id="channel-secret" label={preset.secretLabel} maxLength={512} onChange={(event) => update({ secret: event.target.value })} type="password" value={form.secret} />
              <Input hint="Leave blank to send to the address above." id="channel-to" label="Send alerts to" maxLength={255} onChange={(event) => update({ toAddress: event.target.value })} placeholder={form.emailAddress || "you@example.com"} type="email" value={form.toAddress} />
            </>
          ) : (
            <>
              <Input hint="e.g. smtp.example.com - your email provider lists this as the outgoing/SMTP server." id="channel-host" label="Outgoing mail server (SMTP)" maxLength={255} onChange={(event) => update({ smtpHost: event.target.value })} value={form.smtpHost} />
              <Input hint="Usually 587 for STARTTLS, or 465 for SSL/TLS." id="channel-port" label="Port" onChange={(event) => update({ smtpPort: event.target.value })} type="number" value={form.smtpPort} />
              <Select id="channel-security" label="Encryption" onChange={(event) => update({ security: event.target.value })} value={form.security}>
                <option value="starttls">STARTTLS (recommended)</option>
                <option value="ssl">SSL / TLS</option>
                <option value="none">None (loopback relay only)</option>
              </Select>
              <Input id="channel-from" label="From address" maxLength={255} onChange={(event) => update({ fromAddress: event.target.value })} type="email" value={form.fromAddress} />
              <Input id="channel-custom-to" label="Send alerts to" maxLength={255} onChange={(event) => update({ toAddress: event.target.value })} type="email" value={form.toAddress} />
              <Input hint="Often your full email address." id="channel-username" label="Login name (optional)" maxLength={255} onChange={(event) => update({ username: event.target.value })} value={form.username} />
              <Input hint="Leave blank if the server needs no login." id="channel-custom-secret" label="Password (optional)" maxLength={512} onChange={(event) => update({ secret: event.target.value })} type="password" value={form.secret} />
            </>
          )}
        </>
      ) : (
        <>
          <Input hint="Paste an Incoming Webhook URL from Slack or Discord, or any HTTPS endpoint. Vaelor POSTs the alert as JSON when a rule fires." id="channel-url" label="Webhook URL" maxLength={2000} onChange={(event) => update({ url: event.target.value })} placeholder="https://hooks.slack.com/services/..." value={form.url} />
          <Input hint="Only if your endpoint requires one, e.g. Authorization." id="channel-header" label="Auth header name (optional)" maxLength={100} onChange={(event) => update({ authHeader: event.target.value })} value={form.authHeader} />
          <Input hint="Sent as the header value. Stored encrypted." id="channel-token" label="Auth token (optional)" maxLength={512} onChange={(event) => update({ secret: event.target.value })} type="password" value={form.secret} />
        </>
      )}
      <Input hint="So you recognise it in the list. We'll name it for you if you leave this blank." id="channel-name" label="Name this channel (optional)" maxLength={100} onChange={(event) => update({ name: event.target.value })} value={form.name} />
    </>
  );
  const addButton = <Button disabled={busy || !isFormValid(form)} type="submit" variant="primary">Add channel</Button>;

  const dialogs = (
    <>
      {adding && (
        <ModalShell className="as-dialog ar-dialog-sm" error={submitted && noticeRefused ? notice : ""} labelledBy="add-channel-title" onClose={() => setAdding(false)}>
          <form className="ar-dialog-form" onSubmit={submit}>
            <div className="as-dialog__head">
              <div><span className="as-label">Where should alerts go?</span><h2 id="add-channel-title">Add a delivery channel</h2></div>
              <Button aria-label="Close add channel" className="as-btn-ghost" onClick={() => setAdding(false)} variant="quiet">Close</Button>
            </div>
            <div className="as-dialog__body">{fields}</div>
            <div className="as-dialog__foot"><Button onClick={() => setAdding(false)}>Cancel</Button>{addButton}</div>
          </form>
        </ModalShell>
      )}
      <ConfirmDialog
        busy={busy}
        confirmLabel="Delete channel"
        description={pendingDelete ? `Delete "${pendingDelete.name}"? Its stored secret is removed from the credential broker.` : ""}
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => { if (pendingDelete) { void deleteChannel(pendingDelete); setPendingDelete(null); } }}
        open={Boolean(pendingDelete)}
        title="Delete delivery channel?"
      />
    </>
  );

  const contents = (
    <>
      <div className={headless ? "ar-channels__body" : "ar-card__pad"}>
        <p className="ar-step__hint">
          When an alert rule fires, Vaelor sends the details here so you find out even while you are
          away - an email, or a post to Slack, Discord, or any webhook. Passwords are stored encrypted
          and never shown again. Add a channel, then send a test to be sure it works.
        </p>
        {headless && inDialog && (
          <div className="ar-actions ar-actions--end"><Button onClick={openAdd}><Icon className="ar-btn-icon" name="add" size={ICON_SIZE.inline} />Add channel</Button></div>
        )}
        {!inDialog && (
          <form className="ar-form" onSubmit={submit}>
            {fields}
            <div className="ar-actions ar-actions--end">{addButton}</div>
          </form>
        )}
        {notice && !adding && <Notice severity={noticeRefused ? "danger" : "success"}>{notice}</Notice>}
        <div className="ar-rule-list">
          {channels.map((channel) => (
            <article className="ar-channel" key={channel.id}>
              <div className="ar-split-head">
                <div className="ar-channel__who">
                  <span className="ar-meta">{channel.kind === "email" ? `email · ${channel.to_address}` : `webhook · ${channel.url}`}</span>
                  <h3 className="ar-rule__name">{channel.name}</h3>
                </div>
                <StatusPill tone={channel.enabled ? "success" : "neutral"} label={channel.enabled ? "Enabled" : "Paused"} />
              </div>
              <div className="ar-inline">
                <StatusPill tone={deliveryTone(channel.last_delivery_status)} label={deliveryLabel(channel)} />
                {channel.last_delivery_status === "failed" && channel.last_delivery_error && <span className="ar-meta">{channel.last_delivery_error}</span>}
                {channel.last_delivery_at ? <span className="ar-meta">{timeAgo(channel.last_delivery_at * 1000)}</span> : null}
              </div>
              <div className="ar-actions">
                <Button disabled={busy} onClick={() => void testChannel(channel)}>Send test</Button>
                <Button aria-pressed={channel.enabled} disabled={busy} onClick={() => void toggleChannel(channel)}>{channel.enabled ? "Pause" : "Enable"}</Button>
                <Button className="as-btn-danger" disabled={busy} onClick={() => setPendingDelete(channel)}>Delete</Button>
              </div>
            </article>
          ))}
        </div>
        {channels.length === 0 && (
          <EmptyState
            action={inDialog ? <Button onClick={openAdd}>Add channel</Button> : undefined}
            text="Add an email or webhook channel so a fired alert reaches a person."
            title="No delivery channels yet"
          />
        )}
      </div>
      {dialogs}
    </>
  );

  if (headless) return <div className="ar-channels ar-channels--headless">{contents}</div>;

  return (
    <section aria-labelledby="alert-channels-title" className="card ui-card ar-card ar-channels" id="alert-channels">
      <header className="ar-card__head ar-card__head--start">
        {!inDialog && <span aria-hidden="true" className="ar-icon"><Icon name="shield" size={16} /></span>}
        <div>
          {inDialog && <span className="as-label">Get told when something needs attention</span>}
          <h2 className="ar-card__title" id="alert-channels-title">Where should alerts go?</h2>
          {!inDialog && <span className="ar-meta">Delivery channels</span>}
        </div>
        {inDialog && <Button onClick={openAdd}><Icon className="ar-btn-icon" name="add" size={ICON_SIZE.inline} />Add channel</Button>}
      </header>
      {contents}
    </section>
  );
}
