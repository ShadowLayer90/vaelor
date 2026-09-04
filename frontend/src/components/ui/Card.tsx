import { useId, type HTMLAttributes, type ReactNode } from "react";
import { joinClassNames } from "./field";

export type CardProps = {
  children: ReactNode;
  heading?: ReactNode;
  description?: ReactNode;
  footer?: ReactNode;
  className?: string;
} & Omit<HTMLAttributes<HTMLElement>, "children" | "className">;

export function Card({
  children,
  heading,
  description,
  footer,
  className,
  ...props
}: CardProps) {
  const headingId = "ui-card-" + useId().replaceAll(":", "") + "-heading";
  return (
    <article
      {...props}
      aria-labelledby={heading ? headingId : props["aria-labelledby"]}
      className={joinClassNames("card", "ui-card", className)}
    >
      {(heading || description) && (
        <header className="ui-card__header">
          {heading && <h2 id={headingId}>{heading}</h2>}
          {description && <p>{description}</p>}
        </header>
      )}
      <div className="ui-card__body">{children}</div>
      {footer && <footer className="ui-card__footer">{footer}</footer>}
    </article>
  );
}
