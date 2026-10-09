import "../styles/apps-manage.css";
import { formatQuantity } from "../lib/format";
import type { Session } from "../types";
import { AppsIconTile } from "./appsKit";
import { appRowDetail, appStatePill, modelUsePill, servedByLine, type ManagedApp, type ManagedModel } from "./manageModel";
import { PaginatedItems } from "./PaginatedItems";
import { StatusPill } from "./StatusPill";
import { Button, Card, Notice } from "./ui";
import type { RemovalResource } from "./WorkloadRemovalDialog";

/*
 * The Manage tab's left column (VD-200, the Manage and ManageModelSwitch
 * boards): "Your services" and "Downloaded models". A list that was not read
 * says so in words; it never reads as an empty one (Task #122).
 */

export type ListState = "loading" | "ok" | "error";

const count = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

export function ManageServicesCard({
  apps,
  selectedId,
  state,
  onChoose,
}: {
  apps: ManagedApp[];
  selectedId: string | null;
  state: ListState;
  onChoose: (app: ManagedApp) => void;
}) {
  return (
    <Card actions={state === "ok" ? <span>{count(apps.length, "app", "apps")}</span> : undefined} as="section" className="manage-list" flush heading="Your services">
      {apps.length ? (
        <PaginatedItems items={apps} label="Installed applications" pageSize={8} render={(app) => {
          const pill = appStatePill(app);
          const chosen = app.id === selectedId;
          const detailId = `manage-row-detail-${app.id}`;
          // The row opens the app's manager. Its name starts with a word that says so and
          // keeps the visible name and state inside it (WCAG 2.5.3), so the sweep finds it
          // as "Manage …": named by the app alone it was the one door into the manager the
          // inventory could not name (VD-200 merge, LESSONS 7).
          return (
            <Button
              aria-current={chosen ? "true" : undefined}
              aria-describedby={detailId}
              aria-label={`Manage ${app.name}, ${pill.label}`}
              className={chosen ? "manage-row manage-row--button is-chosen" : "manage-row manage-row--button"}
              key={app.id}
              onClick={() => onChoose(app)}
              variant="quiet"
            >
              <AppsIconTile name="apps" />
              <span className="manage-row__text">
                <span className="manage-row__title">{app.name}</span>
                <span className="manage-row__detail" id={detailId}>{appRowDetail(app)}</span>
              </span>
              <StatusPill description={pill.description} label={pill.label} tone={pill.tone} />
            </Button>
          );
        }} />
      ) : state === "error" ? (
        <div className="manage-list__message">
          <Notice severity="warning" standing>The installed-app list could not be read. This is not a claim that nothing is installed - the appliance did not answer in time. Use Reload above.</Notice>
        </div>
      ) : state === "loading" ? (
        <p className="manage-list__quiet" role="status">Reading the installed apps…</p>
      ) : (
        <div className="manage-list__message">
          <p className="manage-empty"><strong>No apps installed yet.</strong> Return to Install and choose an app. It will appear here automatically.</p>
        </div>
      )}
    </Card>
  );
}

export function ManageModelsCard({
  busy,
  models,
  selectedId,
  session,
  state,
  onRemove,
  onUse,
}: {
  busy: boolean;
  models: ManagedModel[];
  selectedId: string;
  session: Session;
  state: ListState;
  onRemove: (resource: RemovalResource) => void;
  onUse: (model: ManagedModel) => void;
}) {
  const isAdministrator = session.user.role === "administrator";
  return (
    <Card actions={state === "ok" ? <span>{count(models.length, "model", "models")}</span> : undefined} as="section" className="manage-list" flush heading="Downloaded models">
      {models.length ? (
        <PaginatedItems items={models} label="Downloaded AI models" pageSize={8} render={(model) => {
          const pill = modelUsePill(model);
          const served = servedByLine(model);
          const chosen = model.id === selectedId;
          // #147: a file the catalog does not name has no verified identity and
          // no measured footprint, so it is not offerable. It must not guess at
          // provenance: `catalog: false` says the filename is unknown to the
          // catalog, not who put the file there.
          const uncatalogued = model.catalog === false && !model.in_use;
          return (
            <div className={chosen ? "manage-row manage-row--model is-chosen" : "manage-row manage-row--model"} key={model.id}>
              <AppsIconTile name={model.kind === "npu" ? "npu" : "server"} />
              <div className="manage-row__text">
                <span className="manage-row__title">{model.name}</span>
                <span className="manage-row__detail">{model.kind === "npu" ? "On-device (NPU) model" : model.file}</span>
                {served && <span className="manage-row__detail">{served}</span>}
                {model.size_reason && <span className="manage-row__detail">{model.size_reason}</span>}
              </div>
              <div className="manage-row__trail">
                <span className="manage-row__size">{model.size_bytes === null ? "Size unknown" : formatQuantity(model.size_bytes, "model")}</span>
                {!uncatalogued && <StatusPill label={pill.label} tone={pill.tone} />}
              </div>
              <div className="manage-row__below">
                {uncatalogued && <StatusPill label={pill.label} tone={pill.tone} />}
                {model.removable === false ? (
                  <p className="manage-row__note">{model.status_reason || "This model is not removed from this list."}</p>
                ) : (
                  <>
                    {uncatalogued && <p className="manage-row__note">Not in Vaelor&apos;s verified catalog - no measured settings, so the Assistant does not offer it.</p>}
                    <div className="manage-row__actions">
                      {/* #145: "Use model" sat on the row badged "In use" - an offer to
                          switch to the model already serving. The in-use row offers its
                          runtime settings instead, under a label that says what it is. */}
                      {!uncatalogued && (
                        <Button
                          className={model.in_use ? undefined : "manage-use-button"}
                          disabled={busy}
                          disabledReason={session.user.role === "viewer" ? "Viewers cannot change the served model." : undefined}
                          onClick={() => onUse(model)}
                        >
                          {model.in_use ? "Adjust runtime…" : "Use model"}
                        </Button>
                      )}
                      <Button
                        disabled={busy}
                        disabledReason={isAdministrator ? undefined : "Administrator access is required to remove a model."}
                        onClick={() => onRemove({ kind: "model", id: model.id, name: model.name, display_identity: model.name, path: model.path })}
                        variant="danger"
                      >
                        {model.in_use || model.selected || model.in_use_known === false ? "Review and remove…" : "Remove…"}
                      </Button>
                    </div>
                  </>
                )}
              </div>
            </div>
          );
        }} />
      ) : state === "error" ? (
        <div className="manage-list__message">
          <Notice severity="warning" standing>The downloaded-model list could not be read. Nothing here says a model is missing - the appliance did not answer in time. Use Reload above.</Notice>
        </div>
      ) : state === "loading" ? (
        <p className="manage-list__quiet" role="status">Reading the downloaded models…</p>
      ) : (
        <div className="manage-list__message">
          <p className="manage-empty"><strong>No local models yet.</strong> Choose a model from Install to add private AI.</p>
        </div>
      )}
    </Card>
  );
}
