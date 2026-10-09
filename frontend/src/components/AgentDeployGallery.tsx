import { useEffect, useMemo, useState } from "react";
import { Button, Notice } from "./ui";
import { IconTile, KeyValues } from "./ClusterPrimitives";
import { ClusterDialog } from "./ClusterDialog";
import "../styles/cluster-dialogs.css";
import { apiRequest } from "../lib/api";
import type { AgentProfile } from "./agentTypes";
import type { AgentDeployValues } from "./AgentDeployModal";

/**
 * F6c-2b: the Easy agent-deploy gallery - a pick-and-go card view over the
 * deployable custom agents, layered on the Advanced {@link AgentDeployModal}.
 *
 * A card opens the SAME Advanced modal PREFILLED (via its `initial` prop) with
 * just the chosen agent and a default name, so the Easy path is "prefilled, not
 * unreviewed": the operator still lands on the modal's configure/review step,
 * the one honest backing is still resolved there, and the deploy still runs
 * through `buildAgentDeployPayload` -> `queueClusterJob` in `FleetCenter`. This
 * gallery never builds a payload or deploys; it only chooses what to prefill.
 *
 * "Advanced setup" opens the modal blank. A profile that carries an acting
 * permission cannot back a read-only cluster agent (the modal blocks it at
 * `assert_read_only`), so its card's Deploy is disabled with the same reason
 * rather than opening a modal that will refuse.
 *
 * The deployable set is read the way the modal reads it
 * (`/assistant/profiles?surface=inference`, the inference agents only). The two
 * surfaces are shown in sequence - this gallery closes as the modal opens - so
 * the modal's own fetch is a fresh read, not a simultaneous duplicate.
 */

/**
 * A controller-friendly default name derived from the profile name. The
 * controller slugs the name anyway (see the modal's "Agent name" hint); this
 * just seeds a sensible, readable default for the Easy path.
 */
export function defaultAgentName(profileName: string): string {
  const slug = profileName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug ? slug + "-agent" : "cluster-agent";
}

interface AgentDeployGalleryProps {
  onClose: () => void;
  /** Open the Advanced modal with no prefill. */
  onAdvanced: () => void;
  /** Open the Advanced modal prefilled for the chosen agent (the Easy path). */
  onDeploy: (initial: Partial<AgentDeployValues>) => void;
}

export function AgentDeployGallery({ onClose, onAdvanced, onDeploy }: AgentDeployGalleryProps) {
  const [profiles, setProfiles] = useState<AgentProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Read the inference agents (the only kind a cluster deploy offers).
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    apiRequest<AgentProfile[]>("/assistant/profiles?surface=inference", { signal: controller.signal })
      .then((rows) => {
        if (controller.signal.aborted) return;
        setProfiles(Array.isArray(rows) ? rows : []);
        setError("");
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setError(reason instanceof Error ? reason.message : "The custom agents could not be read.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  // Deployable = an enabled inference agent. The fetch already returns inference
  // agents only; the custom+enabled filter stays as a defensive match to the
  // modal.
  const deployableProfiles = useMemo(
    () => profiles.filter((profile) => profile.custom === true && profile.enabled === true),
    [profiles],
  );

  return (
    <ClusterDialog
      className="cd-dialog"
      eyebrow="Cluster agents"
      footer={(
        <>
          <Button variant="secondary" onClick={onClose}>Cancel</Button>
          <Button variant="secondary" onClick={onAdvanced}>Advanced setup</Button>
        </>
      )}
      onClose={onClose}
      size="wide"
      title="Deploy an agent"
      titleId="agent-gallery-title"
    >
      <p>
        Pick a custom agent to deploy as a read-only cluster agent backed by
        the cluster model serving now, with no MCP tool grants and no
        skills. You review exactly what it will be allowed before it is
        deployed. For tool grants or skills, open Advanced setup.
      </p>

      {error ? (
        <Notice severity="danger" heading="Custom agents unavailable">{error}</Notice>
      ) : loading ? (
        <p className="cd-note" role="status">Reading custom agents...</p>
      ) : deployableProfiles.length === 0 ? (
        <Notice severity="info" heading="Create an inference agent first">
          No inference agent is available to deploy. Build one in Cluster ›
          Agents &amp; tools › Inference agents, then it will appear here to
          deploy in one step.
        </Notice>
      ) : (
        <div className="cd-gallery">
          {deployableProfiles.map((profile) => {
            const actingPermissions = profile.permissions ?? [];
            const blocked = actingPermissions.length > 0;
            // Mirror the modal's read-only gate: a profile holding an acting
            // permission would be refused at deploy, so its Deploy is disabled
            // with the reason rather than opening a modal that will block.
            const blockReason = blocked
              ? "Holds acting permission(s): " + actingPermissions.join(", ")
                + ". A cluster agent must be read-only, so this cannot be deployed."
              : undefined;
            const scopeCount = profile.scopes?.length ?? 0;
            return (
              <article className="cd-agent" key={profile.id} aria-labelledby={`agent-card-${profile.id}`}>
                <IconTile accent name="assistant" />
                <span className="cd-note">Custom agent</span>
                <h3 id={`agent-card-${profile.id}`}>{profile.name}</h3>
                <p>{profile.description || "No description provided."}</p>
                <KeyValues
                  items={[
                    { label: "Access", value: blocked ? "Acting permission" : "Read-only", mono: false },
                    { label: "Read scopes", value: scopeCount ? scopeCount + " scope(s)" : "None", mono: false },
                    { label: "Version", value: profile.version ?? 1, mono: false },
                  ]}
                  label={`${profile.name} access`}
                />
                <Button
                  variant="secondary"
                  disabled={blocked}
                  disabledReason={blockReason}
                  onClick={() => onDeploy({ customAgentId: profile.id, name: defaultAgentName(profile.name) })}
                >
                  Deploy
                </Button>
              </article>
            );
          })}
        </div>
      )}
    </ClusterDialog>
  );
}
