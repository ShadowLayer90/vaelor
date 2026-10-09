import type { AgentConnector } from "../components/agentTypes";

/**
 * The words for a custom agent's integration calls (ffb8234 review note).
 *
 * "Recent integration activity" printed the connector runtime's raw operation
 * id and status (`op_3f9a · approval_required`). These are connector events
 * from `vaelor/custom_connector_runtime.py`, not audit actions, so they are
 * not auditLabels' to name; this is their one owner. The operation is named
 * by the description its owner wrote, the connector by its name.
 */

/** One row of `GET /assistant/custom-agents/<id>/connector-audit`. */
export interface ConnectorEvent {
  connector_id?: unknown;
  operation_id?: unknown;
  status?: unknown;
}

/** The statuses `custom_connector_runtime._audit` writes, in words. */
const statusWords: Record<string, string> = {
  completed: "Completed",
  failed: "Failed",
  denied: "Refused by the integration's rules",
  approval_required: "Waiting for approval",
};

export function connectorStatusLabel(status: unknown): string {
  return statusWords[String(status ?? "")] ?? "Outcome not recognised";
}

export function connectorEventText(event: ConnectorEvent, connectors: AgentConnector[]): string {
  const connector = connectors.find((item) => item.id === event.connector_id);
  const operation = connector?.operations.find((item) => item.id === event.operation_id);
  const what = operation?.description.trim() || "An operation since removed";
  const where = connector?.name.trim() || "an integration since removed";
  return `${what} on ${where} - ${connectorStatusLabel(event.status)}`;
}
