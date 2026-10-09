import {
  machinePayload,
  z2Accelerators,
  z2Graphics,
  z2NeuralAccelerators,
} from "./machineFixture";

/**
 * What the backend actually returns on an HP Z2 Mini G1a.
 *
 * These are not invented shapes. They are the payloads the existing Python
 * package produces on a host with no Pironman bridge: `GENERIC_PRODUCT` with
 * `fan_count: 0` and no capabilities, a fan state where neither controller is
 * detected but all five Pironman airflow profiles are still returned, a
 * lighting state that is undetected yet still reports four LEDs, and the
 * `LinuxTelemetryProvider` snapshot — which emits CPU, memory and network and
 * nothing else, so `pwm_fan_speed`, `boot_time` and every `disk_*_percent` key
 * is simply absent.
 */
export const z2Machine = {
  ...machinePayload("workstation", { gpu: true, npu: true, fan_readings: true }),
  accelerators: z2Accelerators,
  neural_accelerators: z2NeuralAccelerators,
  graphics: z2Graphics,
};

/** The same workstation before its accelerator payloads are served. */
export const z2MachineWithoutInventory = machinePayload("workstation", { gpu: true });

export const piMachine = machinePayload("pi-appliance", {
  case_fan: true,
  case_lighting: true,
  oled: true,
  cpu_fan: true,
});

export const z2Device = {
  name: "HP Z2 Mini G1a",
  id: "generic",
  version: "2.0.0",
  peripherals: [],
  platform: {
    product: {
      id: "generic", name: "Generic host", icon: "generic",
      fan_count: 0, nvme_slots: 0, capabilities: [],
      confident: true, detected_from: "fallback",
    },
    board: {
      // 16 physical cores, 32 logical threads — the live `lscpu` reads
      // `Core(s) per socket: 16`, `Thread(s) per core: 2`, `CPU(s): 32`. This
      // said `cpu_cores: 32`, which is `os.cpu_count()` (threads) wearing the
      // core field's name — the #107/VD-062 defect the fixture must not repeat,
      // or it cannot tell the bug from the fix.
      model: "HP Z2 Mini G1a", cpu_model: "AMD Ryzen AI MAX+ PRO 395",
      architecture: "x86_64", cpu_cores: 16, cpu_threads: 32,
      cpu_cores_are_threads: false,
      memory_total_bytes: 48_318_382_080, is_raspberry_pi: false,
    },
    os: {
      id: "ubuntu", name: "Ubuntu 26.04 LTS", version: "26.04", family: "debian",
      supported: true, support_level: "compatible" as const,
      support_label: "Compatible", support_note: "Tested configuration",
    },
    power: {
      input_voltage: null, output_watts: null, undervoltage_now: false,
      battery: { available: false, percentage: null, charging: null },
    },
    feature_policy: {},
  },
};

export const z2FanState = {
  profiles: [
    { id: 0, name: "Continuous", description: "Keep enclosure airflow running at every CPU fan level." },
    { id: 1, name: "Early start", description: "Start enclosure airflow with CPU cooling level 1 (about 50°C)." },
    { id: 2, name: "Normal start", description: "Start enclosure airflow with CPU cooling level 2 (about 60°C)." },
    { id: 3, name: "Late start", description: "Start enclosure airflow with CPU cooling level 3 (about 67.5°C)." },
    { id: 4, name: "Emergency only", description: "Start enclosure airflow only at maximum CPU cooling (about 75°C)." },
  ],
  fans: [
    {
      id: "cpu-pwm" as const, name: "CPU fan", kind: "pwm" as const, detected: false,
      control: "unavailable" as const, rpm: null, mode: "automatic" as const,
      current_state: null, max_state: null, temperature: null, writable: false,
      curve: [
        { temperature: 50, percent: 30, state: 1 },
        { temperature: 60, percent: 50, state: 2 },
        { temperature: 67.5, percent: 70, state: 3 },
        { temperature: 75, percent: 100, state: 4 },
      ],
      safety_limit: 80,
    },
    {
      id: "case-gpio" as const, name: "Case fans", kind: "gpio" as const, detected: false,
      control: "profile" as const, running: null, profile: 2, led: "follow" as const,
      fan_count: 0, shared_control: true, rpm_available: false,
    },
  ],
};

export const z2LightingState = {
  detected: false, hardware: "WS2812 SPI strip", led_count: 4, enabled: true,
  color: "#00ffff", brightness: 100, speed: 50, style: "breathing",
  styles: [
    { id: "solid", name: "Solid" },
    { id: "breathing", name: "Breathing" },
    { id: "rainbow", name: "Rainbow" },
  ],
};

/** Everything `LinuxTelemetryProvider` emits, and nothing it does not. */
export const z2Metrics = {
  cpu_percent: 4,
  cpu_freq: 3_800,
  cpu_temperature: 61.4,
  // `linux_sensors.cpu_temperature()` emits both of these on every sample and
  // the client displayed neither, so a labelled k10temp reading and an ACPI
  // board sensor standing in for one were indistinguishable on screen.
  cpu_temperature_source: "k10temp/Tctl",
  cpu_temperature_labelled: true,
  memory_percent: 74,
  memory_used: 33_900_000_000,
  memory_total: 45_600_000_000,
  network_download_speed: 1_020,
  network_upload_speed: 520,
  // `LinuxTelemetryProvider` emits all three on every sample. `cpu_cores` is
  // PHYSICAL cores (16), `cpu_threads` the logical count (32); on this SMT part
  // they differ, so a fixture carrying `cpu_cores: 32` re-committed #107.
  cpu_cores: 16,
  cpu_threads: 32,
  cpu_cores_are_threads: false,
};

/**
 * The same machine when labelled selection fails and the hottest thermal zone
 * stands in — on an AMD workstation, typically the ACPI board sensor.
 */
export const z2UnlabelledMetrics = {
  ...z2Metrics,
  cpu_temperature_source: "thermal/acpitz",
  cpu_temperature_labelled: false,
};

/**
 * The same host once the GPU telemetry keys land, at the values measured
 * mid-inference on the real machine.
 *
 * The reserved video carve-out sits at about **1%** throughout every ROCm run
 * while the shared aperture carries the model. Both pairs are emitted by
 * `accelerator_telemetry()`; a fixture that showed only the carve-out filling
 * would hide the exact defect the GPU tile had.
 */
export const z2GpuMetrics = {
  ...z2Metrics,
  gpu_temperature_c: 71.5,
  gpu_power_watts: 96,
  gpu_vram_used_bytes: 163_577_856,
  gpu_vram_total_bytes: 17_179_869_184,
  gpu_gtt_used_bytes: 11_200_000_000,
  gpu_gtt_total_bytes: 24_493_297_664,
  gpu_busy_percent: 63,
  gpu_clock_mhz: 2_400,
  gpu_name: "AMD Radeon 8060S (Strix Halo)",
};

/**
 * The keys the appliance actually emits on this workstation.
 *
 * Transcribed from `tests/z2_sysfs.py`'s `Z2_METRICS`, which is itself taken
 * from a live host: the amdgpu driver publishes `mem_info_gtt_total` and **no**
 * `mem_info_gtt_used`, so `accelerator_telemetry()` correctly omits
 * `gpu_gtt_used_bytes` entirely. A client that substitutes zero here reports an
 * aperture holding 19 GB mid-inference as empty.
 */
export const z2MetricsNoGttUsed = {
  ...z2Metrics,
  gpu_temperature_c: 31.0,
  gpu_power_watts: 24.1,
  gpu_busy_percent: 0,
  gpu_vram_total_bytes: 17_179_869_184,
  gpu_vram_used_bytes: 168_349_696,
  gpu_gtt_total_bytes: 24_460_939_264,
  gpu_clock_mhz: 600,
};

/** The same adapter with neither used figure: the carve-out branch, unmeasured. */
export const z2MetricsNoUsedAtAll = {
  ...z2Metrics,
  gpu_busy_percent: 0,
  gpu_vram_total_bytes: 17_179_869_184,
};

/** An adapter with no shared aperture: only the carve-out is reportable. */
export const z2DiscreteGpuMetrics = {
  ...z2Metrics,
  gpu_temperature_c: 71.5,
  gpu_busy_percent: 63,
  gpu_vram_used_bytes: 6_200_000_000,
  gpu_vram_total_bytes: 17_179_869_184,
};

/**
 * What `hp-wmi` publishes on this chassis, at the values measured under load.
 *
 * Three tachometers and seven labelled temperature channels — on a machine
 * whose fan *control* is genuinely absent. Ambient and M.2 move 1-2 °C where
 * the processor moves 28; they are measuring a different thing slowly, and a
 * client that filtered them by range would drop exactly the channels that
 * prove the board is instrumented.
 */
export const z2BoardSensors = {
  wmi_fans: [
    { label: "CPU Fan1", rpm: 1_820, fault: false },
    { label: "CPU Fan2", rpm: 1_790, fault: false },
    { label: "Power Supply Fan", rpm: 1_240, fault: false },
  ],
  wmi_temperatures: [
    { label: "CPU", celsius: 60, fault: false },
    { label: "VDDCR VR", celsius: 41, fault: false },
    { label: "VDDCR_CCD", celsius: 37, fault: false },
    { label: "M.2 SSD1", celsius: 33, fault: false },
    { label: "M.2 SSD2", celsius: 34, fault: false },
    { label: "North Ambient", celsius: 33, fault: false },
    { label: "South Ambient", celsius: 31, fault: false },
  ],
  wmi_temperatures_uncorroborated: [],
  cpu_temperature_sources_agree: true,
  cpu_temperature_source_delta_c: 1.4,
};

/** The same host with its board sensors reporting. */
export const z2MetricsWithBoardSensors = {
  ...z2GpuMetrics,
  ...z2BoardSensors,
};
