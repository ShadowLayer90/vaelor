import { lazy, useEffect } from "react";

/*
 * Every route chunk's import() factory is named once and shared between its
 * React.lazy() wrapper and the idle-prefetch hook, so warming a route and later
 * rendering it hit the same module in the bundler's and the browser's caches.
 *
 * The production build code-splits each of these into its own chunk
 * (vite.config.ts), which keeps the initial load to the entry + vendor chunks.
 * `useRouteChunkPrefetch` then fetches them on idle after first paint so the
 * first navigation to a route is not a cold fetch - the light first load is
 * preserved while navigation still feels instant.
 */
const importFanControl = () =>
  import("./FanControl").then((module) => ({ default: module.FanControl }));
const importWorkloads = () =>
  import("./Workloads").then((module) => ({ default: module.Workloads }));
const importAgentCenter = () =>
  import("./AgentCenter").then((module) => ({ default: module.AgentCenter }));
const importAiChat = () =>
  import("./AiChat").then((module) => ({ default: module.AiChat }));
const importActivityCenter = () =>
  import("./ActivityCenter").then((module) => ({ default: module.ActivityCenter }));
const importAdministration = () =>
  import("./Administration").then((module) => ({ default: module.Administration }));
const importRemoteConsole = () =>
  import("./RemoteConsole").then((module) => ({ default: module.RemoteConsole }));
const importFleetCenter = () =>
  import("./FleetCenter").then((module) => ({ default: module.FleetCenter }));
const importMemoryCenter = () =>
  import("./MemoryCenter").then((module) => ({ default: module.MemoryCenter }));

const routeChunkFactories: Array<() => Promise<unknown>> = [
  importFanControl,
  importWorkloads,
  importAgentCenter,
  importAiChat,
  importActivityCenter,
  importAdministration,
  importRemoteConsole,
  importFleetCenter,
  importMemoryCenter,
];

export const FanControl = lazy(importFanControl);
export const Workloads = lazy(importWorkloads);
export const AgentCenter = lazy(importAgentCenter);
export const AiChat = lazy(importAiChat);
export const ActivityCenter = lazy(importActivityCenter);
export const Administration = lazy(importAdministration);
export const RemoteConsole = lazy(importRemoteConsole);
export const FleetCenter = lazy(importFleetCenter);
export const MemoryCenter = lazy(importMemoryCenter);

/**
 * Warm the lazy route chunks once, after the first paint, on idle time. Empty
 * deps so it runs a single time; the browser and React both cache the module,
 * so a subsequent real navigation is a no-op fetch. It never blocks the initial
 * render - the warm-up is deferred to `requestIdleCallback` (or a short timeout
 * fallback) and cancelled on unmount.
 */
export function useRouteChunkPrefetch(): void {
  useEffect(() => {
    let cancelled = false;
    const warm = () => {
      if (cancelled) return;
      for (const load of routeChunkFactories) void load().catch(() => undefined);
    };
    const idle = window as Window & {
      requestIdleCallback?: (callback: () => void) => number;
      cancelIdleCallback?: (handle: number) => void;
    };
    if (typeof idle.requestIdleCallback === "function") {
      const handle = idle.requestIdleCallback(warm);
      return () => {
        cancelled = true;
        idle.cancelIdleCallback?.(handle);
      };
    }
    const timer = window.setTimeout(warm, 2_000);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);
}
