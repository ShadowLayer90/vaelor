import type { NodeCapacity } from "./fleetTypes";
import type { FleetLiveReading } from "../hooks/useFleetLiveMemory";
import { nodeStateWord } from "../lib/fleetNodeState";

/**
 * How much of a machine's memory is in use, for the Fleet headline and the
 * machine cards (VD-129, ACC-120).
 *
 * The figure is what the machine MEASURED, the same figure the Home screen
 * shows: the operating system's memory in use (MemTotal - MemAvailable).
 * - the controller reports it in bytes (`/telemetry/current` `memory_used`);
 * - a worker reports it as a percentage of its total (its fresh `latest`
 *   `memory_percent`, computed by the worker agent exactly as Home computes
 *   it), which is scaled by the machine's own memory from the capacity ledger.
 *
 * Resident unified (GPU/GTT) memory is NOT added on top: on these unified-memory
 * machines it is drawn from the same system pool, so the operating system's
 * figure already contains it, and adding it would count it twice.
 *
 * The placement ledger's Swarm reservation is never used as an in-use figure:
 * Swarm reserves almost nothing for what actually runs, so "capacity minus
 * reservation" read several GB more free than the machine had. With no fresh
 * measurement the answer is `null` - the card says its memory is not reported
 * and the headline leaves the machine out and says so - never a reservation
 * figure presented as free.
 *
 * This is the ONE derivation both the fleet headline (summed over the machines
 * it counts) and the per-machine card call, so the aggregate can never disagree
 * with the rows below it. The result is capped at the machine's own memory so
 * free is never negative.
 */
export function residentInUseBytes(
  node: NodeCapacity,
  reading: FleetLiveReading | null | undefined,
): number | null {
  const total = Math.max(0, node.capacity.memory_bytes);
  if (!reading) return null;
  if (reading.memoryUsedBytes !== null) {
    return Math.min(total, Math.max(0, reading.memoryUsedBytes));
  }
  if (reading.memoryPercent !== null && total > 0) {
    return Math.min(total, Math.round((Math.max(0, reading.memoryPercent) / 100) * total));
  }
  return null;
}

/**
 * Why a machine has no current memory figure, as the headline and the card
 * name it: its reading aged out, its telemetry could not be read at all, or
 * nothing was reported.
 */
export function unmeasuredPhrase(reading: FleetLiveReading | null | undefined): string {
  if (reading?.stale) return "memory reading out of date";
  if (reading?.readFailed) return "memory could not be read";
  return "memory not reported";
}

/** The fleet headline's memory figures and what they cover. */
export interface FleetMemoryHeadline {
  /** Machines whose memory is counted: schedulable AND measured. */
  countedNodes: number;
  totalNodes: number;
  /** The counted machines' own memory, what is in use on them, and what is free. */
  total: number;
  used: number;
  free: number;
  /** Each machine left out, named with why: "ZBook (drained)". */
  leftOut: string[];
}

/**
 * The fleet headline (ACC-091, ACC-120): only machines that can take work now
 * (the ledger's `schedulable`) AND whose memory was measured are summed - each
 * through `residentInUseBytes`, the same derivation its card shows - and every
 * machine left out is named with the reason, so the headline never reads
 * "free" over memory a drained, offline, not-joined or unreported machine holds.
 */
export function fleetMemoryHeadline(
  nodes: NodeCapacity[],
  readingFor: (node: NodeCapacity) => FleetLiveReading | null | undefined,
): FleetMemoryHeadline {
  let countedNodes = 0;
  let total = 0;
  let used = 0;
  const leftOut: string[] = [];
  for (const node of nodes) {
    if (node.schedulable !== true) {
      leftOut.push(`${node.name} (${nodeStateWord(node.state).phrase})`);
      continue;
    }
    const reading = readingFor(node);
    const inUse = residentInUseBytes(node, reading);
    if (inUse === null) {
      leftOut.push(`${node.name} (${unmeasuredPhrase(reading)})`);
      continue;
    }
    countedNodes += 1;
    total += Math.max(0, node.capacity.memory_bytes);
    used += inUse;
  }
  return { countedNodes, totalNodes: nodes.length, total, used, free: Math.max(0, total - used), leftOut };
}
