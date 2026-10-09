/**
 * Which GPU temperature a screen shows, and what it is called (VD-147, owner
 * decision D5).
 *
 * A GPU here has two temperature readings: the graphics engine's own (from
 * amd-smi, `gpu_gfx_temperature_c`) and the edge sensor (hwmon,
 * `gpu_temperature_c`). Every screen used to say "GPU temp" and show edge.
 *
 * **The choice between them is the backend's**
 * (`vaelor.platforms.gpu_temperature.gpu_temperature_reading`), sent as
 * `gpu_temperature_sensor`. This module holds no rule of its own: it maps that
 * word onto the field to read and the label to show, so the label always names
 * the sensor. A payload that carries no choice (an older one, or a worker's
 * stored row read directly) is shown as what `gpu_temperature_c` has always
 * been: the edge sensor, under that name.
 *
 * A row whose values were withheld (a stale worker: its old figures are never
 * shown as current) still carries the backend's sensor. It keeps that sensor,
 * its name and its trend series with no value; it does not change sensor and
 * does not say "no sensor read" (S-34).
 */

/** `vaelor.platforms.gpu_temperature.SENSOR_*`. "hotspot" appears only in the on-demand profile. */
export type GpuTemperatureSensor = "graphics engine" | "edge" | "hotspot";

/** The stored field behind each sensor a telemetry row can carry. */
export type GpuTemperatureField = "gpu_gfx_temperature_c" | "gpu_temperature_c";

export interface GpuTemperatureView {
  value: number | null;
  sensor: GpuTemperatureSensor | null;
  /** The field (and trend series) the value came from. */
  field: GpuTemperatureField;
  label: string;
}

const finite = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

/** The label for a GPU temperature from a named sensor, or the honest one when no sensor was read. */
export function gpuTemperatureLabel(sensor: string | null | undefined): string {
  return sensor ? `GPU temperature (${sensor})` : "GPU temperature (no sensor read)";
}

/** The short form for a narrow stat cell. */
export function gpuTemperatureShortLabel(sensor: string | null | undefined): string {
  return sensor ? `GPU temp (${sensor})` : "GPU temp (no sensor read)";
}

/** The GPU temperature a telemetry row shows: the backend's chosen sensor, its value, field and label. */
export function gpuTemperature(metrics: Record<string, unknown> | null | undefined): GpuTemperatureView {
  const row = metrics ?? {};
  const edge = finite(row.gpu_temperature_c);
  const engine = finite(row.gpu_gfx_temperature_c);
  const chosen = row.gpu_temperature_sensor;
  if (chosen === "graphics engine" && engine !== null) {
    return { value: engine, sensor: "graphics engine", field: "gpu_gfx_temperature_c", label: gpuTemperatureLabel(chosen) };
  }
  if (edge !== null) {
    return { value: edge, sensor: "edge", field: "gpu_temperature_c", label: gpuTemperatureLabel("edge") };
  }
  if (engine === null && (chosen === "graphics engine" || chosen === "edge")) {
    const field = chosen === "graphics engine" ? "gpu_gfx_temperature_c" : "gpu_temperature_c";
    return { value: null, sensor: chosen, field, label: gpuTemperatureLabel(chosen) };
  }
  return { value: null, sensor: null, field: "gpu_temperature_c", label: gpuTemperatureLabel(null) };
}
