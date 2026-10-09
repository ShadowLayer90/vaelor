import { useId, type ReactNode } from "react";
import { joinClassNames } from "./field";

/*
 * The redesign's page building blocks (VD-200, the mockup boards). A page is
 * composed from these rather than restyled: a header, stat tiles, cards (see
 * Card), list rows, tables, and the empty and loading states. Their look lives
 * in styles/page-primitives.css and nowhere else.
 */

/**
 * The page's opening line, large and light, with a grey second line and the
 * page's actions on the right (one of them the area's primary action).
 *
 * `as` is the element the title renders as. A page that already has its `<h1>`
 * (Home's is the destination name in the top bar) renders its title as a `p`.
 */
export function PageHeader({
  actions,
  as: Title = "h1",
  subtitle,
  title,
}: {
  actions?: ReactNode;
  as?: "h1" | "h2" | "p";
  subtitle?: ReactNode;
  title: ReactNode;
}) {
  return (
    <div className="ui-page-header">
      <Title className="ui-page-header__title">
        {title}
        {subtitle && <><br /><span>{subtitle}</span></>}
      </Title>
      {actions && <div className="ui-page-header__actions">{actions}</div>}
    </div>
  );
}

/** One bar of a stat tile's short trend; `null` is a bucket with no reading. */
export type TrendBar = number | null;

/**
 * A stat tile: label and side note, the reading large, an optional short trend
 * and a foot line. A reading that was not taken is `null` and reads "Not read"
 * in words, never as a zero or a full bar (LESSONS 1).
 */
export function StatTile({
  foot,
  href,
  label,
  side,
  trend,
  unit,
  value,
}: {
  foot?: ReactNode;
  /** Where the reading's detail lives; the whole tile is then a link there. */
  href?: string;
  label: string;
  side?: ReactNode;
  /** Bars as fractions 0..1 of the tile's own range, oldest first. */
  trend?: TrendBar[];
  unit?: string;
  value: string | null;
}) {
  const Root = href ? "a" : "div";
  return (
    <Root className={joinClassNames("ui-card", "ui-stat-tile", href && "ui-stat-tile--link")} href={href}>
      <div className="ui-stat-tile__top"><span>{label}</span>{side !== undefined && <span>{side}</span>}</div>
      {value === null
        ? <div className="ui-stat-tile__value ui-stat-tile__value--unread">Not read</div>
        : <div className="ui-stat-tile__value">{value}{unit && <span>{unit}</span>}</div>}
      {trend && trend.length > 0 && (
        <div aria-hidden="true" className="ui-trend">
          {trend.map((bar, index) => (
            <i
              className={joinClassNames(
                index === trend.length - 1 && bar !== null && "is-latest",
                bar === null && "is-empty",
              )}
              // A 6% floor so a measured zero is still a mark; an empty bucket draws none.
              key={index}
              style={{ height: bar === null ? undefined : `${Math.max(6, Math.min(100, bar * 100))}%` }}
            />
          ))}
        </div>
      )}
      {foot && <div className="ui-stat-tile__foot">{foot}</div>}
    </Root>
  );
}

/** A row inside a card: an icon box, a title over a detail line, and what trails it (a pill, an age). */
export function ListRow({
  detail,
  icon,
  iconAccent = false,
  title,
  trailing,
  align = "center",
}: {
  detail?: ReactNode;
  icon?: ReactNode;
  iconAccent?: boolean;
  title: ReactNode;
  trailing?: ReactNode;
  align?: "center" | "start";
}) {
  return (
    <div className={joinClassNames("ui-row", align === "start" && "ui-row--start")}>
      {icon && <span aria-hidden="true" className={joinClassNames("ui-row__icon", iconAccent && "ui-row__icon--accent")}>{icon}</span>}
      <div className="ui-row__text">
        <div className="ui-row__title">{title}</div>
        {detail && <div className="ui-row__detail">{detail}</div>}
      </div>
      {trailing && <div className="ui-row__trailing">{trailing}</div>}
    </div>
  );
}

export interface TableColumn {
  key: string;
  label: string;
}

/**
 * A table inside a card. It scrolls sideways inside its own box below
 * `minWidth`, so the page never does.
 */
export function DataTable({
  caption,
  columns,
  minWidth = 720,
  rows,
}: {
  caption: string;
  columns: TableColumn[];
  minWidth?: number;
  rows: Array<{ key: string; cells: Record<string, ReactNode> }>;
}) {
  return (
    <div className="ui-table-scroll">
      <table className="ui-table" style={{ minWidth }}>
        <caption className="sr-only">{caption}</caption>
        <thead><tr>{columns.map((column) => <th key={column.key} scope="col">{column.label}</th>)}</tr></thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key}>{columns.map((column) => <td key={column.key}>{row.cells[column.key]}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A used share as a 6 px bar, with its figure in words beneath (never alone). */
export function MeterBar({ fraction, label }: { fraction: number | null; label: ReactNode }) {
  return (
    <div className="ui-meter">
      {fraction !== null && <div aria-hidden="true" className="ui-meter__track"><i style={{ width: `${Math.max(0, Math.min(1, fraction)) * 100}%` }} /></div>}
      <div className="ui-meter__label">{label}</div>
    </div>
  );
}

/** Nothing here yet, said plainly, with the way forward when there is one. */
export function EmptyState({
  action,
  icon,
  text,
  title,
}: {
  action?: ReactNode;
  icon?: ReactNode;
  text?: ReactNode;
  title: ReactNode;
}) {
  return (
    <div className="ui-empty">
      {icon && <span aria-hidden="true" className="ui-empty__icon">{icon}</span>}
      <strong>{title}</strong>
      {text && <span>{text}</span>}
      {action}
    </div>
  );
}

/** A reading on its way: grey bars where the text will be, announced once. */
export function LoadingLines({ label, lines = 2 }: { label: string; lines?: number }) {
  return (
    <div className="ui-loading" role="status">
      <span className="sr-only">{label}</span>
      {Array.from({ length: lines }, (_, index) => <span aria-hidden="true" className="ui-skeleton" key={index} />)}
    </div>
  );
}

/** One choice from a few, as a row of pills; the chosen one carries the accent. */
export function SegmentedControl<T extends string>({
  label,
  onChange,
  options,
  value,
}: {
  label: string;
  onChange: (value: T) => void;
  options: Array<{ value: T; label: string; disabled?: boolean }>;
  value: T;
}) {
  const id = "ui-seg-" + useId().replaceAll(":", "");
  return (
    <div aria-labelledby={id} className="ui-segmented" role="group">
      <span className="sr-only" id={id}>{label}</span>
      {options.map((option) => (
        <button
          aria-pressed={option.value === value}
          className={joinClassNames("ui-segmented__option", option.value === value && "is-on")}
          disabled={option.disabled}
          key={option.value}
          onClick={() => onChange(option.value)}
          type="button"
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
