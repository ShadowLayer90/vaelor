import { useCallback, useState } from "react";

/**
 * The Cluster page's Easy / Advanced detail level, persisted per user.
 *
 * Easy is the default and shows the plain, derived view; Advanced only ever
 * *adds* controls and figures on top of Easy — it never hides anything Easy
 * shows, so a user cannot be stranded by their own choice of mode. The
 * preference is stored under `vaelor.cluster.mode.<username>` so two accounts on
 * the same browser keep their own setting, and every access is wrapped so a
 * browser that refuses storage (private windows, blocked site data) falls back
 * to Easy for the session rather than throwing.
 */

export type ClusterMode = "easy" | "advanced";

const KEY_PREFIX = "vaelor.cluster.mode.";

export function clusterModeStorageKey(username: string): string {
  return `${KEY_PREFIX}${username}`;
}

export function readClusterMode(username: string): ClusterMode {
  try {
    return window.localStorage.getItem(clusterModeStorageKey(username)) === "advanced"
      ? "advanced"
      : "easy";
  } catch {
    return "easy";
  }
}

export function writeClusterMode(username: string, mode: ClusterMode): void {
  try {
    window.localStorage.setItem(clusterModeStorageKey(username), mode);
  } catch {
    // Storage is unavailable (private window, quota, blocked site data). The
    // mode still applies for this session; it simply is not remembered.
  }
}

/** Read/write the persisted mode as React state. */
export function useClusterMode(
  username: string,
): [ClusterMode, (mode: ClusterMode) => void] {
  const [mode, setMode] = useState<ClusterMode>(() => readClusterMode(username));
  const update = useCallback((next: ClusterMode) => {
    setMode(next);
    writeClusterMode(username, next);
  }, [username]);
  return [mode, update];
}
