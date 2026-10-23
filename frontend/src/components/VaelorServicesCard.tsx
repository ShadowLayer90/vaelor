import { useState } from "react";
import "../styles/apps-manage-vaelor.css";
import { AppLogViewer } from "./AppLogViewer";
import { AppsIconTile } from "./appsKit";
import { appStatePill, type ManagedApp } from "./manageModel";
import { StatusPill } from "./StatusPill";
import { Button, Card } from "./ui";

/*
 * VD-133 1.5.5 #12 (review F5): the containers Vaelor runs for itself -
 * tracing, web search, model serving - are not the owner's apps, so "Your
 * services" no longer lists them. They stay reachable here, collapsed and
 * read-only: their state and their logs, and nothing that changes them.
 */

/** The part of Vaelor a container belongs to, in the owner's words. */
const PARTS: Record<string, string> = {
  "phoenix": "Tracing (Phoenix)",
  "web-research": "Web search",
  "llm-server": "LLM Server",
  "gpu-model": "GPU AI Chat model",
  "gpu-serving": "GPU model serving",
  "cluster-serving": "Cluster model serving",
  "cluster-agent": "Custom agent gate",
};

export function vaelorPart(component: string | undefined): string {
  return PARTS[component ?? ""] ?? (component ? component.replaceAll("-", " ") : "Part of Vaelor");
}

export function VaelorServicesCard({ services, viewer }: { services: ManagedApp[]; viewer: boolean }) {
  const [logsFor, setLogsFor] = useState<string | null>(null);
  if (!services.length) return null;
  const running = services.filter((service) => service.running).length;
  return (
    <Card as="section" className="manage-list" flush heading="Vaelor's own services">
      <details className="manage-vaelor__details">
        <summary className="manage-vaelor__summary">
          {`${services.length} ${services.length === 1 ? "service" : "services"}, ${running} running · read-only`}
        </summary>
        <p className="manage-vaelor__note">
          These run Vaelor itself, so they are not removed, restarted or updated from here. Their logs are open to read.
        </p>
        <ul aria-label="Vaelor's own services" className="manage-vaelor__list">
          {services.map((service) => {
            const pill = appStatePill(service);
            const open = logsFor === service.id;
            return (
              <li className="manage-vaelor__item" key={service.id}>
                <div className="manage-row manage-vaelor__row">
                  <AppsIconTile name="server" />
                  <span className="manage-row__text">
                    <span className="manage-row__title">{service.name}</span>
                    <span className="manage-row__detail">{vaelorPart(service.vaelor_component)} · part of Vaelor</span>
                  </span>
                  <StatusPill description={pill.description} label={pill.label} tone={pill.tone} />
                  <Button aria-expanded={open} onClick={() => setLogsFor(open ? null : service.id)} variant="quiet">
                    {open ? "Hide logs" : "Logs"}<span className="sr-only">{` · ${service.name}`}</span>
                  </Button>
                </div>
                {open && (
                  <div className="manage-vaelor__logs">
                    <AppLogViewer
                      appId={service.id}
                      appName={service.name}
                      refusal={viewer ? "Operator access is required to read a service's logs." : undefined}
                    />
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </details>
    </Card>
  );
}
