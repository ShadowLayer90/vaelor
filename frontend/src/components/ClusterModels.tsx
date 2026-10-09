import { useCallback, useEffect, useState } from "react";
import { Button, Input, LoadingLines, type StatusTone } from "./ui";
import { ClusterCard, OptionCard } from "./ClusterPrimitives";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { apiRequest } from "../lib/api";
import { formatBytes } from "../lib/format";
import type { ClusterModel, ClusterModelInventory, FleetNode } from "./fleetTypes";
import type { Session } from "../types";

const STATE_TONE: Record<string, StatusTone> = {
  ready: "success",
  pulling: "info",
  error: "danger",
};

function stateTone(state: string): StatusTone {
  return STATE_TONE[state] ?? "neutral";
}

/**
 * The model library: the per-node cached weights `GET /cluster/models` reports,
 * with the free space beside them, plus the two lifecycle actions — pull a repo
 * onto chosen workers, and remove a repo's weights from a node.
 *
 * It reads its own inventory (a plain operator GET) and reloads on demand; the
 * pull and remove *mutations* are queued as `cluster.model.pull` /
 * `cluster.model.remove` jobs by `FleetCenter`, so their progress renders in the
 * same operation modal every cluster change uses, and the library is reloaded
 * when the caller signals a change through `reloadKey`.
 *
 * Every terminal state is truthful: a cluster it cannot reach shows the error, a
 * library with nothing cached says so, a node mid-pull shows `pulling`, and a
 * failed pull keeps its `error` row with the reason rather than vanishing.
 *
 * VD-200 (the ClusterDeploymentsModels board): the "Model library" card beside
 * the LLM Server under Deployments > Models - the pull field, the machines as
 * "Cache onto" option cards, and one box of cached weights per machine.
 */

interface ClusterModelsProps {
  session: Session;
  /** The machines a pull may target: the controller (over its hardware bridge)
   *  and every eligible SSH worker - the backend caches weights on either. */
  pullTargets: FleetNode[];
  /** Bumped by the caller after a pull/remove job finishes, to force a reload. */
  reloadKey?: number;
  onPull: (payload: Record<string, unknown>) => void;
  onRemove: (payload: Record<string, unknown>) => void;
}

/** A cached row's label: the repo, and the pinned revision when one exists. */
function modelLabel(model: ClusterModel): string {
  return model.revision ? `${model.repo} @ ${model.revision}` : model.repo;
}

/**
 * Whether a deployment still needs a row's weights, from the backend's own
 * reading (ACC-110): a paused (scaled-to-zero) deployment loads again from
 * them, so they read "In use" and Remove waits. Unknown use is never "none".
 */
function weightsUse(model: ClusterModel): { pill: { label: string; tone: StatusTone } | null; note: string; blocked: "" | "in-use" | "unknown" } {
  if (model.in_use_known === false) {
    return {
      pill: { label: "Use not checked", tone: "warning" },
      note: "Vaelor could not read the deployments, so it cannot tell whether these weights are needed.",
      blocked: "unknown",
    };
  }
  const users = model.in_use_by ?? [];
  if (!users.length) return { pill: null, note: "", blocked: "" };
  return {
    pill: { label: "In use", tone: "info" },
    note: "Needed by " + users.map((user) => `${user.name} (${user.state_label})`).join("; ")
      + ". Remove that deployment before removing these weights.",
    blocked: "in-use",
  };
}

export function ClusterModels({ session, pullTargets, reloadKey, onPull, onRemove }: ClusterModelsProps) {
  const [inventory, setInventory] = useState<ClusterModelInventory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [source, setSource] = useState("");
  const [pullNodes, setPullNodes] = useState<string[]>([]);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      setInventory(await apiRequest<ClusterModelInventory>("/cluster/models", { cache: "no-store" }));
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The model library is unavailable.");
      setInventory(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void reload(); }, [reload, reloadKey]);

  const togglePullNode = (id: string) => {
    setPullNodes((current) =>
      current.includes(id) ? current.filter((entry) => entry !== id) : [...current, id]);
  };

  const canPull = Boolean(source.trim()) && pullNodes.length > 0 && pullTargets.length > 0;

  const submitPull = () => {
    if (!canPull) return;
    onPull({ model_source: source.trim(), node_ids: pullNodes });
    setSource("");
    setPullNodes([]);
  };

  // The library lists every enrolled node; a node with a ready model somewhere is
  // what "not empty" means, so an all-empty fleet reports honestly rather than
  // showing rows of zeroes as if they were content.
  const nodes = inventory?.nodes ?? [];
  const cachedCount = nodes.reduce((total, node) => total + node.models.length, 0);


  return (
    <ClusterCard
      actions={(
        <Button variant="quiet" onClick={() => void reload()} disabled={loading}>
          <Icon name="refresh" size={16} />
          Reload
        </Button>
      )}
      description="Cached weights beneath serving"
      icon="download"
      title="Model library"
    >
      <Input
        label="Pull a model"
        placeholder="https://huggingface.co/org/model, org/model, or a catalog id"
        value={source}
        onChange={(event) => setSource(event.target.value)}
      />
      {pullTargets.length ? (
        <>
          <fieldset className="cl-lib-targets">
            <legend>Cache onto</legend>
            {pullTargets.map((node) => (
              <OptionCard
                checked={pullNodes.includes(node.id)}
                // The machine's processor family, so weights are not cached
                // onto a machine that cannot run them (as at 68ccdfb).
                detail={node.inventory?.architecture ?? "Unknown architecture"}
                key={node.id}
                onChange={() => togglePullNode(node.id)}
                title={node.name}
                type="checkbox"
              />
            ))}
          </fieldset>
          <div className="cl-actions">
            <Button
              variant="primary"
              disabled={!canPull}
              disabledReason={canPull ? undefined : "Enter a model source and choose at least one machine."}
              onClick={submitPull}
            >
              Pull to selected machines
            </Button>
          </div>
        </>
      ) : (
        <p className="cl-meta" role="status">
          No machine can hold model weights yet. Join a machine with a supported GPU to pull a model to it.
        </p>
      )}

      {loading && !inventory ? (
        <LoadingLines label="Reading the model library…" />
      ) : error ? (
        <p className="cl-bad-text" role="alert">{error}</p>
      ) : !cachedCount ? (
        <p className="cl-meta">No models are cached on any machine yet.</p>
      ) : (
        nodes.map((node) => (
          <article className="cl-lib-node" key={node.node_id}>
            <header>
              <h3>{node.name}</h3>
              <span className="cl-meta">{formatBytes(node.cached_bytes)} cached · {formatBytes(node.root_free_bytes)} free</span>
            </header>
            {node.models.length ? (
              <ul className="cl-lib-models">
                {node.models.map((model) => {
                  const use = weightsUse(model);
                  const noteId = `cluster-model-use-${node.node_id}-${model.repo}`.replace(/[^A-Za-z0-9_-]/g, "-");
                  return (
                    <li key={`${node.node_id}-${model.repo}-${model.revision ?? ""}`}>
                      <div className="cl-rows__text">
                        <span className="cl-ep-mono">
                          <span>{modelLabel(model)}</span>
                          {" · "}
                          <span>
                            {model.state === "ready"
                              ? formatBytes(model.bytes_on_disk ?? 0)
                              : model.message || model.state}
                          </span>
                        </span>
                        {use.note && <span className={use.blocked === "unknown" ? "cl-warn-text" : "cl-lib-use"} id={noteId}>{use.note}</span>}
                      </div>
                      <StatusPill
                        label={use.pill?.label ?? model.state}
                        tone={use.pill?.tone ?? stateTone(model.state)}
                      />
                      <Button
                        variant="quiet"
                        // The reason is the row's own use line (beside the
                        // repo), not a second line under the button.
                        disabled={Boolean(use.blocked)}
                        aria-describedby={use.blocked ? noteId : undefined}
                        onClick={() => onRemove({ model_source: model.repo, node_ids: [node.node_id] })}
                      >
                        Remove
                      </Button>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="cl-meta">Nothing cached here.</p>
            )}
          </article>
        ))
      )}
    </ClusterCard>
  );
}
