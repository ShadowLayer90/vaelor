import { useId, type HTMLAttributes, type ReactNode } from "react";
import { joinClassNames } from "./field";

export type CardProps = {
  children: ReactNode;
  heading?: ReactNode;
  description?: ReactNode;
  /** What sits at the right of the header: a link, a count pill, one button. */
  actions?: ReactNode;
  footer?: ReactNode;
  /**
   * Rows or a table that run edge to edge (ListRow, DataTable): the body has
   * no padding of its own, the rows carry it.
   */
  flush?: boolean;
  /** The element the card is: `section` for a page panel, `article` for an item. */
  as?: "article" | "section";
  className?: string;
} & Omit<HTMLAttributes<HTMLElement>, "children" | "className">;

export function Card({
  children,
  heading,
  description,
  actions,
  footer,
  flush = false,
  as: Element = "article",
  className,
  ...props
}: CardProps) {
  const headingId = "ui-card-" + useId().replaceAll(":", "") + "-heading";
  return (
    <Element
      {...props}
      aria-labelledby={heading ? headingId : props["aria-labelledby"]}
      className={joinClassNames("card", "ui-card", className)}
    >
      {(heading || description || actions) && (
        <header className="ui-card__header">
          <div className="ui-card__titles">
            {heading && <h2 id={headingId}>{heading}</h2>}
            {description && <p>{description}</p>}
          </div>
          {actions && <div className="ui-card__actions">{actions}</div>}
        </header>
      )}
      <div className={joinClassNames("ui-card__body", flush && "ui-card__body--flush")}>{children}</div>
      {footer && <footer className="ui-card__footer">{footer}</footer>}
    </Element>
  );
}
