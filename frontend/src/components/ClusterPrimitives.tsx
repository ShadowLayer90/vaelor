import { useId, type ChangeEvent, type HTMLAttributes, type ReactNode } from "react";
import { Icon, type IconName } from "./Icon";
import { Button } from "./ui/Button";
import { joinClassNames } from "./ui/field";
import "../styles/cluster.css";

/*
 * The Cluster page's own building blocks (VD-200, the Cluster boards): the
 * card with an icon tile in its header, the dialog with an eyebrow, a body
 * that scrolls and a fixed footer, the key/value grid, the step strip, the
 * option card, the filter chips. They sit beside the shared primitives in
 * components/ui (frozen for this build) and compose them; their look lives in
 * styles/cluster.css. Claude may lift any of them into components/ui later.
 */

/** A 34 px icon tile, grey or on the accent's ground. */
export function IconTile({ name, accent = false }: { name: IconName; accent?: boolean }) {
  return (
    <span aria-hidden="true" className={joinClassNames("cl-ico", accent && "cl-ico--accent")}>
      <Icon name={name} size={18} />
    </span>
  );
}

/**
 * A card whose header carries an icon tile before its title and description
 * (the boards' "Head controller", "Add a machine", "GPU memory pool"), with
 * what trails it on the right. Without an icon it is the plain card.
 */
export function ClusterCard({
  actions,
  as: Element = "section",
  children,
  className,
  description,
  flush = false,
  footer,
  icon,
  iconAccent = false,
  title,
  ...props
}: {
  actions?: ReactNode;
  as?: "article" | "section";
  children?: ReactNode;
  className?: string;
  description?: ReactNode;
  flush?: boolean;
  footer?: ReactNode;
  icon?: IconName;
  iconAccent?: boolean;
  title?: ReactNode;
} & Omit<HTMLAttributes<HTMLElement>, "children" | "className" | "title">) {
  const headingId = "cl-card-" + useId().replaceAll(":", "") + "-title";
  return (
    <Element
      {...props}
      aria-labelledby={title ? headingId : props["aria-labelledby"]}
      className={joinClassNames("ui-card", "cl-card", className)}
    >
      {(title || actions) && (
        <header className="ui-card__header cl-card__header">
          <div className="cl-card__lead">
            {icon && <IconTile accent={iconAccent} name={icon} />}
            <div className="ui-card__titles">
              {title && <h2 id={headingId}>{title}</h2>}
              {description && <p>{description}</p>}
            </div>
          </div>
          {actions && <div className="ui-card__actions">{actions}</div>}
        </header>
      )}
      {children !== undefined && children !== null && children !== false && (
        <div className={joinClassNames("ui-card__body", "cl-card__body", flush && "ui-card__body--flush")}>{children}</div>
      )}
      {footer && <footer className="ui-card__footer cl-card__footer">{footer}</footer>}
    </Element>
  );
}

// The dialog lives in its own module so the sweep inventory can name each use
// of it by its title (tools/ui_inventory.py reads dialog primitives by module).
export { ClusterDialog, type DialogTone } from "./ClusterDialog";

export interface KeyValue {
  label: ReactNode;
  value: ReactNode;
  /** Values are JetBrains Mono by default (the board's key/value cells); `false` for words. */
  mono?: boolean;
}

/** The boards' key/value grid: hairline gaps, a quiet label over its value. */
export function KeyValues({ className, items, label }: { className?: string; items: KeyValue[]; label?: string }) {
  return (
    <dl aria-label={label} className={joinClassNames("cl-kv", className)}>
      {items.map((item, index) => (
        <div key={index}>
          <dt>{item.label}</dt>
          <dd className={item.mono === false ? undefined : "cl-kv__mono"}>{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export type StepState = "todo" | "on" | "done";

/** A numbered step strip (Connect, Confirm identity, Review the join). */
export function Steps({ label, steps }: { label: string; steps: Array<{ label: ReactNode; state: StepState }> }) {
  return (
    <ol aria-label={label} className="cl-steps">
      {steps.map((step, index) => (
        <li aria-current={step.state === "on" ? "step" : undefined} className={"cl-steps__step is-" + step.state} key={index}>
          <span aria-hidden="true">{index + 1}</span>
          {step.label}
        </li>
      ))}
    </ol>
  );
}

/**
 * One choice drawn as a card: a real radio or checkbox inside a label, its
 * title and the line under it; the chosen card carries the accent. A
 * disabled card says why in its own detail line.
 */
export function OptionCard({
  checked,
  className,
  detail,
  disabled = false,
  name,
  onChange,
  title,
  type,
  value,
}: {
  checked: boolean;
  className?: string;
  detail?: ReactNode;
  disabled?: boolean;
  name?: string;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
  title: ReactNode;
  type: "radio" | "checkbox";
  value?: string;
}) {
  return (
    <label className={joinClassNames("cl-opt", checked && "is-on", disabled && "is-disabled", className)}>
      {type === "radio"
        ? <input checked={checked} className="cl-opt__radio" disabled={disabled} name={name} onChange={onChange} type="radio" value={value} />
        : <input checked={checked} className="cl-opt__checkbox" disabled={disabled} name={name} onChange={onChange} type="checkbox" value={value} />}
      <span className="cl-opt__text">
        <strong>{title}</strong>
        {detail && <span>{detail}</span>}
      </span>
    </label>
  );
}

/** A row of filter chips (All, Models, Apps, Agents); the chosen one is pressed and carries the accent. */
export function FilterChips<T extends string>({
  label,
  onChange,
  options,
  value,
}: {
  label: string;
  onChange: (value: T) => void;
  options: Array<{ value: T; label: ReactNode; count?: number }>;
  value: T;
}) {
  return (
    <div aria-label={label} className="cl-chips" role="group">
      {options.map((option) => (
        <Button
          aria-pressed={option.value === value}
          className={joinClassNames("cl-chip", option.value === value && "is-on")}
          key={option.value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
          {option.count !== undefined && <span className="cl-chip__count">{option.count}</span>}
        </Button>
      ))}
    </div>
  );
}

/** The boards' uppercase section label ("Machines awaiting join", "Other states"). */
export function SectionLabel({ children, as: Element = "p" }: { children: ReactNode; as?: "p" | "h2" | "h3" }) {
  return <Element className="cl-section-label">{children}</Element>;
}

/** A small grey tag (a deployment's type, an approved tool's name). */
export function Tag({ children, mono = false }: { children: ReactNode; mono?: boolean }) {
  return <span className={joinClassNames("cl-tag", mono && "cl-tag--mono")}>{children}</span>;
}

/** A block of copyable text: an address, a key, a fingerprint. */
export function CodeBlock({ children, className }: { children: ReactNode; className?: string }) {
  return <code className={joinClassNames("cl-code", className)}>{children}</code>;
}
