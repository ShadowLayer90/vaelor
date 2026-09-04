interface SearchableAuditEvent {
  action: string;
  actor: string;
  target: string;
  result: string;
}

const normalize = (value: string) =>
  value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

export function auditMatches(
  event: SearchableAuditEvent,
  query: string,
) {
  const term = normalize(query);
  if (!term) return true;
  return normalize(
    `${event.action} ${event.actor} ${event.target} ${event.result}`,
  ).includes(term);
}
