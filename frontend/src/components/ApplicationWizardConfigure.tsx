import { type FormEvent, Fragment, useEffect, useMemo, useState } from "react";
import { AppsInset, AppsKv } from "./appsKit";
import {
  type ApplicationIntent,
  defaultDeploymentName,
  type DeploymentConfiguration,
  groupByService,
  MAX_DEPLOYMENT_NAME,
  portConfigKey,
  PRIVILEGED_HOST_MOUNTS,
  type ResearchReport,
} from "./applicationDeploymentModel";
import { Button, Checkbox, Input, Select } from "./ui";

/*
 * Step 3 of the custom-application wizard (AppsWizardDeploy board, "3 ·
 * Configure"): each field group is a fieldset; the extra-port and host-access
 * groups appear only when the plan names services. The state lives in
 * `useDeploymentConfiguration` so the wizard can reset it, read it for the
 * draft, and say in the footer why Generate is held.
 */

const emptyPortRow = (service: string) => ({
  id: Math.random().toString(36).slice(2),
  service,
  target: "",
  published: "",
  protocol: "tcp" as "tcp" | "udp",
});
type PortRow = ReturnType<typeof emptyPortRow>;

function portFieldValid(value: string) {
  const trimmed = value.trim();
  if (!/^\d+$/.test(trimmed)) return false;
  const port = Number(trimmed);
  return port >= 1 && port <= 65535;
}

/**
 * The deployment-name rule as an HTML `pattern`. Browsers compile `pattern`
 * with the `v` flag, where a bare `-` inside a class is a syntax error; the old
 * "[a-z0-9][a-z0-9-]{1,47}" was rejected and the browser skipped the check
 * entirely. The hyphen is escaped here.
 */
export const DEPLOYMENT_NAME_PATTERN = String.raw`[a-z0-9][a-z0-9\-]{1,47}`;

export const MISSING_SECRET_REASON = "Enter each required application secret.";
export const PORT_ROW_REASON = "Each published port needs a service and a container port between 1 and 65535.";
export const SOCKET_SERVICE_REASON = "Choose the service that should receive the Docker socket mount.";

export function useDeploymentConfiguration(intent: ApplicationIntent | null, research: ResearchReport | null) {
  const [name, setName] = useState("");
  const [memoryMb, setMemoryMb] = useState(1024);
  const [storageGb, setStorageGb] = useState(5);
  const [ports, setPorts] = useState<Record<string, number>>({});
  const [settings, setSettings] = useState<Record<string, string>>({});
  const [secretReferences, setSecretReferences] = useState<Record<string, string>>({});
  const [secretValues, setSecretValues] = useState<Record<string, string>>({});
  const [replaceExisting, setReplaceExisting] = useState(false);
  const [addPorts, setAddPorts] = useState<PortRow[]>([]);
  const [dockerSocketConsent, setDockerSocketConsent] = useState(false);
  const [dockerSocketService, setDockerSocketService] = useState("");
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    if (!research || research.status !== "complete") return;
    setName((current) => current || defaultDeploymentName(intent?.application));
    setMemoryMb((current) => Math.max(current, research.minimumMemoryMb ?? 0));
    setStorageGb((current) => Math.max(current, research.minimumStorageGb ?? 0));
    setPorts((current) => Object.keys(current).length ? current : Object.fromEntries(research.ports.map((port) => [portConfigKey(port), port.container])));
    setSettings((current) => Object.keys(current).length ? current : Object.fromEntries(research.environment.filter((item) => !item.secret).map((item) => [item.name, item.defaultValue ?? ""])));
    setSecretReferences((current) => Object.keys(current).length ? current : Object.fromEntries(research.environment.filter((item) => item.secret).map((item) => [item.name, ""])));
    // When the plan pins exactly one service, the advanced port and host-mount
    // pickers have only one possible target, so seed it rather than force a
    // redundant choice. Multi-service plans leave the operator to pick.
    const services = (research.services ?? []).filter(Boolean);
    if (services.length === 1) setDockerSocketService((current) => current || services[0]);
  }, [intent, research]);

  const availableServices = useMemo(
    () => (research?.services ?? []).filter((value): value is string => Boolean(value)),
    [research?.services],
  );
  // Each operator-added port must name a pinned-image service and a valid
  // container port; a host port and protocol are optional. Rows with any of
  // these unmet block the draft rather than sending the backend a value it will
  // reject, and the offending field carries the message inline.
  const portRowErrors = addPorts.map((row) => ({
    service: !row.service,
    target: !portFieldValid(row.target),
    published: row.published.trim() !== "" && !portFieldValid(row.published),
  }));
  const portRowsInvalid = portRowErrors.some((issue) => issue.service || issue.target || issue.published);
  // Consent without a service to attach the mount to cannot be sent.
  const consentMissingService = dockerSocketConsent && !dockerSocketService;
  const requiredSecretsMissing = research?.environment.some((item) => item.secret && item.required && !secretValues[item.name]?.trim() && !secretReferences[item.name]?.trim()) ?? false;

  /** Why Generate is held, in the order the form asks for them. */
  const blockedReasons = [
    requiredSecretsMissing && MISSING_SECRET_REASON,
    portRowsInvalid && PORT_ROW_REASON,
    consentMissingService && SOCKET_SERVICE_REASON,
  ].filter((reason): reason is string => Boolean(reason));

  function edit<T>(setter: (value: T) => void) {
    return (value: T) => { setter(value); setDirty(true); };
  }

  function reset() {
    setName(""); setMemoryMb(1024); setStorageGb(5); setPorts({}); setSettings({}); setSecretReferences({}); setSecretValues({});
    setReplaceExisting(false); setAddPorts([]); setDockerSocketConsent(false); setDockerSocketService(""); setDirty(false);
  }

  /** The configuration the draft is generated from; secrets are references only. */
  function build(request: string, intentId: string, researchId: string, references: Record<string, string>): DeploymentConfiguration {
    const configuredAddPorts = addPorts.map((row) => ({
      service: row.service,
      target: Number(row.target),
      ...(row.published.trim() ? { published: Number(row.published) } : {}),
      protocol: row.protocol,
    }));
    const hostMounts: DeploymentConfiguration["hostMounts"] =
      dockerSocketConsent && dockerSocketService
        ? [{ service: dockerSocketService, source: "/var/run/docker.sock", consent: true }]
        : [];
    return { request, intentId, researchId, name, ports, memoryMb, storageGb, settings, secretReferences: references, replaceExisting, addPorts: configuredAddPorts, hostMounts };
  }

  return {
    name, setName: edit(setName),
    memoryMb, setMemoryMb: edit(setMemoryMb),
    storageGb, setStorageGb: edit(setStorageGb),
    ports, setPort: (key: string, value: number) => { setPorts((current) => ({ ...current, [key]: value })); setDirty(true); },
    settings, setSetting: (key: string, value: string) => { setSettings((current) => ({ ...current, [key]: value })); setDirty(true); },
    secretReferences, setSecretReferences,
    secretValues, setSecretValue: (key: string, value: string) => { setSecretValues((current) => ({ ...current, [key]: value })); setDirty(true); },
    clearSecretValues: () => setSecretValues({}),
    replaceExisting, setReplaceExisting: edit(setReplaceExisting),
    addPorts,
    addPortRow: () => { setAddPorts((current) => [...current, emptyPortRow(availableServices.length === 1 ? availableServices[0] : "")]); setDirty(true); },
    updatePortRow: (id: string, patch: Partial<PortRow>) => { setAddPorts((current) => current.map((item) => item.id === id ? { ...item, ...patch } : item)); setDirty(true); },
    removePortRow: (id: string) => { setAddPorts((current) => current.filter((item) => item.id !== id)); setDirty(true); },
    portRowErrors,
    dockerSocketConsent, setDockerSocketConsent: edit(setDockerSocketConsent),
    dockerSocketService, setDockerSocketService: edit(setDockerSocketService),
    consentMissingService,
    availableServices,
    multiService: availableServices.length > 1,
    requiredSecretsMissing,
    blockedReasons,
    dirty, markClean: () => setDirty(false),
    reset,
    build,
  };
}

export type DeploymentConfigurationState = ReturnType<typeof useDeploymentConfiguration>;

function ServiceLabel({ service }: { service: string }) {
  return <p className="apps-wizard__service-label">{service || "Application"}</p>;
}

export function ConfigureStep({
  config,
  formId,
  onSubmit,
  research,
}: {
  config: DeploymentConfigurationState;
  formId: string;
  onSubmit: (event: FormEvent) => void;
  research: ResearchReport;
}) {
  const multi = config.multiService;
  const settingsItems = research.environment.filter((item) => !item.secret);
  const secretItems = research.environment.filter((item) => item.secret);
  const services = config.availableServices;
  return (
    <form className="apps-wizard__configuration" id={formId} onSubmit={onSubmit}>
      <fieldset className="apps-wizard__fieldset">
        <legend>Identity and resources</legend>
        <div className="apps-wizard__field-row apps-wizard__field-row--three">
          <Input id="application-name" label="Deployment name" maxLength={MAX_DEPLOYMENT_NAME} onChange={(event) => config.setName(event.target.value)} pattern={DEPLOYMENT_NAME_PATTERN} required value={config.name} />
          <Input id="application-memory" label="Memory limit (MiB)" min={research.minimumMemoryMb ?? 128} onChange={(event) => config.setMemoryMb(event.currentTarget.valueAsNumber)} required step="128" type="number" value={config.memoryMb} />
          <Input id="application-storage" label="Reserved storage (GB)" min={research.minimumStorageGb ?? 1} onChange={(event) => config.setStorageGb(event.currentTarget.valueAsNumber)} required type="number" value={config.storageGb} />
        </div>
      </fieldset>

      {!!research.ports.length && (
        <fieldset className="apps-wizard__fieldset">
          <legend>Network ports</legend>
          {groupByService(research.ports).map(([service, servicePorts]) => (
            <Fragment key={service || "_"}>
              {multi && <ServiceLabel service={service} />}
              {servicePorts.map((port) => {
                const key = portConfigKey(port);
                return <Input id={`application-port-${key.replaceAll(/[/:]/g, "-")}`} key={key} label={port.purpose + " - container " + port.container + "/" + port.protocol} max="65535" min="1" onChange={(event) => config.setPort(key, event.currentTarget.valueAsNumber)} required type="number" value={config.ports[key] ?? port.container} />;
              })}
            </Fragment>
          ))}
        </fieldset>
      )}

      {!!settingsItems.length && (
        <fieldset className="apps-wizard__fieldset">
          <legend>Application settings</legend>
          {groupByService(settingsItems).map(([service, items]) => (
            <Fragment key={service || "_"}>
              {multi && <ServiceLabel service={service} />}
              {items.map((item) => <Input id={`application-setting-${item.name}`} key={item.name} label={<>{item.name}{item.description && <small className="apps-wizard__label-detail">{item.description}</small>}</>} onChange={(event) => config.setSetting(item.name, event.target.value)} required={item.required} value={config.settings[item.name] ?? ""} />)}
            </Fragment>
          ))}
        </fieldset>
      )}

      {!!secretItems.length && (
        <fieldset className="apps-wizard__fieldset">
          <legend>Application secrets</legend>
          <p className="apps-wizard__fieldset-note">Enter each value once. Vaelor stores it immediately in the encrypted broker; only a purpose-bound reference enters the deployment draft.</p>
          {groupByService(secretItems).map(([service, items]) => (
            <Fragment key={service || "_"}>
              {multi && <ServiceLabel service={service} />}
              {items.map((item) => {
                const stored = Boolean(config.secretReferences[item.name]);
                const detail = [item.description, stored ? "Stored securely; enter a new value only to replace it." : ""].filter(Boolean).join(" - ");
                return <Input autoComplete="new-password" id={`application-secret-${item.name}`} key={item.name} label={<>{item.name}{detail && <small className="apps-wizard__label-detail">{detail}</small>}</>} onChange={(event) => config.setSecretValue(item.name, event.target.value)} placeholder={stored ? "Stored securely" : "Enter secret value"} required={item.required && !stored} type="password" value={config.secretValues[item.name] ?? ""} />;
              })}
            </Fragment>
          ))}
        </fieldset>
      )}

      {services.length > 0 && (
        <fieldset className="apps-wizard__fieldset">
          <legend>Publish an extra port (advanced)</legend>
          <p className="apps-wizard__fieldset-note">Most deployments need nothing here. Use it to reach a service the researched plan left with no published port. Vaelor only pins a host port onto the already-verified image.</p>
          {config.addPorts.map((row, index) => (
            <div className="apps-wizard__port-row" key={row.id}>
              <Select id={`application-add-port-service-${row.id}`} label="Service" error={config.portRowErrors[index]?.service ? "Choose a service." : undefined} value={row.service} onChange={(event) => config.updatePortRow(row.id, { service: event.target.value })}>
                <option value="">Choose a service</option>
                {services.map((service) => <option key={service} value={service}>{service}</option>)}
              </Select>
              <Input id={`application-add-port-target-${row.id}`} label="Container port" error={config.portRowErrors[index]?.target ? "1 to 65535." : undefined} inputMode="numeric" max="65535" min="1" type="number" value={row.target} onChange={(event) => config.updatePortRow(row.id, { target: event.target.value })} />
              <Input id={`application-add-port-published-${row.id}`} label="Host port" hint="Optional; defaults to the container port." error={config.portRowErrors[index]?.published ? "1 to 65535." : undefined} inputMode="numeric" max="65535" min="1" placeholder="Optional" type="number" value={row.published} onChange={(event) => config.updatePortRow(row.id, { published: event.target.value })} />
              <Select id={`application-add-port-protocol-${row.id}`} label="Protocol" value={row.protocol} onChange={(event) => config.updatePortRow(row.id, { protocol: event.target.value === "udp" ? "udp" : "tcp" })}>
                <option value="tcp">TCP</option>
                <option value="udp">UDP</option>
              </Select>
              <Button className="apps-wizard__row-remove" onClick={() => config.removePortRow(row.id)}>Remove</Button>
            </div>
          ))}
          <div><Button onClick={config.addPortRow}>Add published port</Button></div>
        </fieldset>
      )}

      {services.length > 0 && (
        <fieldset className="apps-wizard__fieldset">
          <legend>Privileged host access (advanced)</legend>
          <p className="apps-wizard__fieldset-note">Off by default. Only the Docker socket is allowed, and only read-only. Nothing is mounted unless you tick the box below.</p>
          {services.length > 1 && (
            <Select id="application-host-mount-service" label="Service" error={config.consentMissingService ? "Choose a service." : undefined} value={config.dockerSocketService} onChange={(event) => config.setDockerSocketService(event.target.value)}>
              <option value="">Choose a service</option>
              {services.map((service) => <option key={service} value={service}>{service}</option>)}
            </Select>
          )}
          <div className="apps-wizard__check-inset">
            <Checkbox checked={config.dockerSocketConsent} id="application-host-mount-docker" label={<><span className="apps-wizard__check-title">{PRIVILEGED_HOST_MOUNTS[0].title}</span> <small>{PRIVILEGED_HOST_MOUNTS[0].consentLabel}</small></>} onChange={(event) => config.setDockerSocketConsent(event.target.checked)} />
          </div>
        </fieldset>
      )}

      <AppsKv
        cells={[
          { label: "Managed volumes", value: research.volumes.length },
          { label: "Published ports", value: research.ports.length + config.addPorts.length },
          { label: "Configuration keys", value: research.environment.length },
        ]}
        label="Configuration summary"
      />
      <div className="apps-wizard__check-inset">
        <Checkbox checked={config.replaceExisting} id="application-replace-existing" label={<><span className="apps-wizard__check-title">Allow replacement if this application is already managed</span> <small>The current configuration is backed up first. Leave this off for a new installation.</small></>} onChange={(event) => config.setReplaceExisting(event.target.checked)} />
      </div>
    </form>
  );
}

/** A finalized draft (its Compose already generated) is shown read-only, never as a form whose Generate only errors. */
export function ConfigureLocked() {
  return (
    <AppsInset detail="Its immutable Compose draft was generated, so its configuration can no longer be edited. Review and approve it, or discard the saved research to configure this application again from scratch." title="This plan has already been configured" />
  );
}
