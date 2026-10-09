import { useEffect, useState, type ReactNode } from "react";
import { Button, Notice, Select } from "./ui";
import { Icon, ICON_SIZE } from "./Icon";
import { DashboardPanels } from "./DashboardPanels";
import { DashboardStatus } from "./DashboardStatus";
import { DashboardTiles } from "./DashboardTiles";
import type { ClusterMode } from "../lib/clusterMode";
import { useDashboardNow, useDashboardRange, type RangeOption } from "../lib/performanceDashboard";

/**
 * "Updated 4 s ago": how long since this page last received an answer. It is
 * the page's fetch age only, not a verdict on the data (the badges say that),
 * and it ticks on its own so the charts do not re-render every second.
 */
function UpdatedAgo({ at }: { at: number | null }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = globalThis.setInterval(() => setNow(Date.now()), 1000);
    return () => globalThis.clearInterval(timer);
  }, []);
  const seconds = at === null ? null : Math.max(0, Math.round((now - at) / 1000));
  return (
    <span className="perf-header__updated" aria-live="off">
      {seconds === null ? "Not updated yet" : `Updated ${seconds} s ago`}
    </span>
  );
}

/** What the shell needs to read Diagnostics over the same range. */
export interface ResolvedRange { range: string; label: string; seconds: number }

export interface PerformanceDashboardProps {
  mode: ClusterMode;
  /** The chosen range, or null for the backend's default. */
  range: string | null;
  /** A state setter: choosing a range only changes what this page shows (the shell remembers it). */
  setRange: (range: string | null) => void;
  auto: boolean;
  setAuto: (auto: boolean) => void;
  reloadKey: number;
  onReload: () => void;
  /** Told which range the backend answered, with its words. */
  setResolved: (resolved: ResolvedRange) => void;
  /** The Why headline, from the Diagnostics snapshot the shell reads. */
  why?: ReactNode;
  /** The decode floor the speed verdict is judged against, from the same snapshot; drawn on the decode-speed card. */
  decodeFloor?: number | null;
}

/**
 * The Performance dashboard (VD-147 S6; VD-200, the ClusterPerformance board):
 * the header controls, the status card, six tiles and the chart grid. Its two layers poll on their
 * own periods. The Range options and every word shown come from the range
 * layer; until it answers, the select shows only what is chosen.
 */
export function PerformanceDashboard({ mode, range, setRange, auto, setAuto, reloadKey, onReload, setResolved, why, decodeFloor = null }: PerformanceDashboardProps) {
  const now = useDashboardNow(auto, reloadKey);
  const layer = useDashboardRange(range, auto, reloadKey);
  // Only a payload of the right shape is drawn: anything else is treated as not read yet.
  const nowData = now.data && typeof now.data === "object" && now.data.engine && typeof now.data.engine === "object" ? now.data : null;
  const rangeData = layer.data && typeof layer.data === "object" && layer.data.panels && typeof layer.data.panels === "object" ? layer.data : null;
  // The last options and units the backend sent stay usable while a read fails (pass-4 B3).
  const [known, setKnown] = useState<{ options: RangeOption[]; units: Record<string, string> }>({ options: [], units: {} });
  const sentOptions = rangeData?.labels?.range_options;
  const sentUnits = rangeData?.labels?.units;
  useEffect(() => {
    if (sentOptions?.length) setKnown({ options: sentOptions, units: sentUnits ?? {} });
  }, [sentOptions, sentUnits]);
  const options = sentOptions?.length ? sentOptions : known.options;
  const units = sentUnits ?? known.units;
  const shown = range ?? rangeData?.range ?? "";

  useEffect(() => {
    // Only the backend saying no drops a remembered range: it is not in the list it sent, or it refused
    // it (a 400). A failed transport keeps the choice, and Reload tries it again.
    if (range && sentOptions?.length && !sentOptions.some((option) => option.value === range)) setRange(null);
    else if (range && layer.refused) setRange(null);
  }, [range, sentOptions, layer.refused, setRange]);
  // A dashboard that cannot be read still lets Diagnostics read its own default window.
  const unreadable = !rangeData && (Boolean(layer.error) || (layer.data !== null && !layer.loading));
  useEffect(() => {
    if (unreadable && !range) setResolved({ range: "", label: "", seconds: 0 });
  }, [unreadable, range, setResolved]);
  useEffect(() => {
    if (!rangeData?.range) return;
    const label = rangeData.caption_lead ?? options.find((option) => option.value === rangeData.range)?.label ?? "";
    setResolved({ range: rangeData.range, label, seconds: rangeData.range_seconds });
  }, [rangeData?.range, rangeData?.range_seconds, options, setResolved]);

  const received = [now.receivedAt, layer.receivedAt].filter((value): value is number => value !== null);
  const errors = [...new Set([now.error, layer.error].filter(Boolean))];
  return (
    <div className="perf-dashboard">
      <header className="perf-header">
        <div className="perf-header__title">
          <h2>Performance</h2>
          <p>How fast the cluster is serving, per machine.</p>
        </div>
        <div className="perf-header__controls">
          <Select className="perf-header__range" disabled={!options.length} id="perf-range" label="Range" onChange={(event) => setRange(event.target.value)} value={shown}>
            {options.length
              ? options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)
              : <option value={shown}>{layer.error && !layer.loading ? "Not read" : "Reading…"}</option>}
          </Select>
          <span className="perf-switch">
            <Button aria-checked={auto} aria-labelledby="perf-auto-refresh-label" className="perf-switch__control" id="perf-auto-refresh" onClick={() => setAuto(!auto)} role="switch" variant="quiet">
              <span aria-hidden="true" className="perf-switch__track"><span className="perf-switch__knob" /></span>
            </Button>
            <span id="perf-auto-refresh-label">Auto-refresh</span>
          </span>
          <Button className="perf-header__reload" onClick={onReload} type="button" variant="quiet">
            <Icon name="refresh" size={ICON_SIZE.inline} />
            Reload
          </Button>
          <UpdatedAgo at={received.length ? Math.max(...received) : null} />
        </div>
      </header>
      {errors.map((error) => <Notice key={error} severity="warning">{error}</Notice>)}
      {/* Review S4: a now layer that could not be read is a status card that
          says so, not an empty strip. */}
      {!nowData && now.error && !now.loading ? (
        <section aria-label="Serving status" className="ui-card perf-strip">
          <p className="perf-strip__text"><strong>Not read.</strong> {now.error}</p>
        </section>
      ) : <DashboardStatus engine={nowData?.engine} hostNote={rangeData?.host_note} nodes={nowData?.nodes ?? []} why={why} />}
      <DashboardTiles now={nowData} nowError={nowData ? "" : now.error} range={rangeData} rangeError={rangeData ? "" : layer.error} units={units} />
      <DashboardPanels floor={decodeFloor} mode={mode} now={nowData} range={rangeData} unread={rangeData ? "" : layer.error} units={units} />
    </div>
  );
}
