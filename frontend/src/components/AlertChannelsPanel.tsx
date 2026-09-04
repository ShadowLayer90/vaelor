import { useState, type FormEvent } from "react";
import { timeAgo } from "../lib/format";
import { useAlertChannels } from "../hooks/useAlertChannels";
import type { AlertChannel } from "./agentTypes";
import { ConfirmDialog } from "./ConfirmDialog";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button, Input, Notice, Select } from "./ui";

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

function deliveryTone(status: string): "healthy" | "degraded" | "neutral" {
  if (status === "delivered") return "healthy";
  if (status === "failed") return "degraded";
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
 */
export function AlertChannelsPanel({
  csrfToken,
  canManage,
}: {
  csrfToken: string;
  canManage: boolean;
}) {
  const { channels, busy, notice, createChannel, toggleChannel, deleteChannel, testChannel } =
    useAlertChannels({ csrfToken, enabled: canManage });
  const [form, setForm] = useState<ChannelForm>(EMPTY_FORM);
  const [pendingDelete, setPendingDelete] = useState<AlertChannel | null>(null);

  if (!canManage) return null;

  const update = (patch: Partial<ChannelForm>) => setForm((current) => ({ ...current, ...patch }));
  const preset = form.provider === "custom" ? null : EMAIL_PROVIDERS[form.provider];

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (await createChannel(buildBody(form))) setForm({ ...EMPTY_FORM, kind: form.kind, provider: form.provider });
  };

  return (
    <section aria-labelledby="alert-channels-title" className="automation-center" id="alert-channels">
      <div className="automation-divider">
        <span className="page-eyebrow">Get told when something needs attention</span>
        <h2 id="alert-channels-title">Where should alerts go?</h2>
        <p>
          When an alert rule fires, Vaelor sends the details here so you find out even while you are
          away - an email, or a post to Slack, Discord, or any webhook. Passwords are stored encrypted
          and never shown again. Add a channel, then send a test to be sure it works.
        </p>
      </div>
      <form className="skill-proposal" onSubmit={submit}>
        <Select id="channel-kind" label="How should we reach you?" onChange={(event) => update({ kind: event.target.value as ChannelForm["kind"] })} value={form.kind}>
          <option value="email">Email</option>
          <option value="webhook">Slack, Discord, or other webhook</option>
        </Select>
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
        <Button disabled={busy || !isFormValid(form)} type="submit" variant="primary">Add channel</Button>
      </form>
      {notice && <Notice severity="info">{notice}</Notice>}
      <div className="skill-list">
        {channels.map((channel) => (
          <article className="skill-card" key={channel.id}>
            <div className="agent-task-card__header">
              <div>
                <small>{channel.kind === "email" ? `email · ${channel.to_address}` : `webhook · ${channel.url}`}</small>
                <h2>{channel.name}</h2>
              </div>
              <StatusPill status={channel.enabled ? "healthy" : "neutral"} label={channel.enabled ? "enabled" : "paused"} />
            </div>
            <div className="alert-channel-delivery">
              <StatusPill status={deliveryTone(channel.last_delivery_status)} label={deliveryLabel(channel)} />
              {channel.last_delivery_status === "failed" && channel.last_delivery_error && <small>{channel.last_delivery_error}</small>}
              {channel.last_delivery_at ? <small>{timeAgo(channel.last_delivery_at * 1000)}</small> : null}
            </div>
            <div className="agent-task-card__actions">
              <Button disabled={busy} onClick={() => void testChannel(channel)} type="button" variant="secondary">Send test</Button>
              <Button aria-pressed={channel.enabled} disabled={busy} onClick={() => void toggleChannel(channel)} type="button" variant="quiet">{channel.enabled ? "Pause" : "Enable"}</Button>
              <Button className="danger-text" disabled={busy} onClick={() => setPendingDelete(channel)} type="button" variant="danger">Delete</Button>
            </div>
          </article>
        ))}
        {channels.length === 0 && (
          <div className="empty-state">
            <Icon name="activity" />
            <h3>No delivery channels yet</h3>
            <p>Add an email or webhook channel above so a fired alert reaches a person.</p>
          </div>
        )}
      </div>
      <ConfirmDialog
        busy={busy}
        confirmLabel="Delete channel"
        description={pendingDelete ? `Delete "${pendingDelete.name}"? Its stored secret is removed from the credential broker.` : ""}
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => { if (pendingDelete) { void deleteChannel(pendingDelete); setPendingDelete(null); } }}
        open={Boolean(pendingDelete)}
        title="Delete delivery channel?"
      />
    </section>
  );
}
