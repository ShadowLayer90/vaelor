export const isValidAutomationSchedule = (value: string, now = Date.now()) => {
  const clean = value.trim().toLowerCase();
  let match = clean.match(/^in\s+(\d+)\s+(minute|minutes|hour|hours)$/);
  if (match) {
    const seconds = Number(match[1]) * (match[2].startsWith("hour") ? 3600 : 60);
    return seconds >= 60 && seconds <= 365 * 86400;
  }
  match = clean.match(/^every\s+(\d+)\s+(minute|minutes|hour|hours|day|days)$/);
  if (match) {
    const multiplier = match[2].startsWith("day") ? 86400 : match[2].startsWith("hour") ? 3600 : 60;
    const seconds = Number(match[1]) * multiplier;
    return seconds >= 300 && seconds <= 30 * 86400;
  }
  const timestamp = Date.parse(clean);
  return Number.isFinite(timestamp) && timestamp > now;
};

export const triggerLimits: Record<string, [number, number]> = {
  cpu_temperature: [40, 100],
  memory_percent: [1, 100],
  storage_percent: [1, 100],
  service_failures: [1, 10],
  fan_failure: [1, 1],
};

export const isValidTriggerThreshold = (source: string, value: number) =>
  Boolean(triggerLimits[source] && Number.isFinite(value) && value >= triggerLimits[source][0] && value <= triggerLimits[source][1]);
