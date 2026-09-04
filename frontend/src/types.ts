export type Role = "viewer" | "operator" | "administrator";

export interface User {
  username: string;
  role: Role;
}

export interface Session {
  user: User;
  csrf_token: string;
  expires_at: number;
}

export interface Device {
  name: string;
  id: string;
  version: string;
  peripherals: string[];
  platform?: {
    product: {
      id: string;
      name: string;
      icon: string;
      fan_count: number;
      nvme_slots: number;
      capabilities: string[];
      confident: boolean;
      detected_from: string;
    };
    board: {
      model: string;
      cpu_model: string;
      architecture: string;
      cpu_cores: number;
      memory_total_bytes: number;
      is_raspberry_pi: boolean;
    };
    os: {
      id: string;
      name: string;
      version: string;
      family: string;
      supported: boolean;
      support_level: "verified" | "compatible" | "limited" | "unknown";
      support_label: string;
      support_note: string;
    };
    power: {
      input_voltage: number | null;
      output_watts: number | null;
      undervoltage_now: boolean;
      /** `null` off-Pi: a workstation has no PMIC to name. */
      source?: string | null;
      battery: {
        available: boolean;
        percentage: number | null;
        charging: boolean | null;
        /**
         * How long the appliance would run on the battery, and whether that
         * figure rests on a discharge this appliance has actually been
         * through. Both optional because the backend does not record a
         * discharge observation yet — and until it does, the panel says the
         * runtime is unknown rather than deriving a number from a capacity it
         * has never watched drain. A UPS panel that guesses a runtime is the
         * exact failure this work exists to remove.
         */
        runtime_minutes?: number | null;
        discharge_observed?: boolean;
      };
    };
    feature_policy: Record<string, boolean | string>;
  };
  model_choices?: Array<{ id: string; name: string; icon: string }>;
}

/**
 * One value in a telemetry sample.
 *
 * Structured entries are part of this payload and always have been —
 * `cpu_core_temperatures` and `storage_temperatures` are lists, and the board
 * sensors add `wmi_fans` and `wmi_temperatures` as lists of records. The type
 * said scalar-or-null, so every consumer of a structured key had to cast its
 * way out, which is how a reading ends up parsed twice in two different ways.
 * Readers of scalars are unaffected: `metricNumber` and the formatters already
 * test the runtime type before trusting it.
 */
export type MetricValue =
  | string
  | number
  | boolean
  | null
  | readonly unknown[]
  | { readonly [key: string]: unknown };
export type Metrics = Record<string, MetricValue>;

export interface TelemetrySample {
  sampled_at: number;
  metrics: Metrics;
}

export interface Health {
  status: "healthy" | "degraded" | "critical" | "offline";
  reasons: string[];
  sampled_at: number;
  /**
   * The categories the server actually evaluated for this verdict.
   *
   * `vaelor/health_evaluation.py` **serves this today**, in the order the
   * sentence should read it. Optional only for an appliance too old to send
   * it: that absence is the single case where the client may fall back to the
   * categories `/health` is known to compute.
   *
   * An **empty array is an answer** — "nothing on this sample was measurable"
   * — and must never be read as a missing field. See `lib/health.ts`; the
   * first version of that module conflated the two and told the reader two
   * categories were clear on a machine that had checked none.
   */
  checked?: string[];
}

export interface AuditEvent {
  id: number;
  created_at: number;
  actor: string;
  action: string;
  target: string;
  result: "success" | "failure";
  remote_addr: string;
  details: Record<string, unknown>;
}

export interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    [key: string]: unknown;
  };
}
