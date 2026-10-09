import type { MouseEvent } from "react";

/** The main region the skip link moves focus to (Overview renders it). */
export const MAIN_CONTENT_ID = "main-content";

/**
 * "Skip to main content" for a keyboard user (R-F5). This app is hash-routed,
 * so a plain `href="#main-content"` set the route to one nobody has and the
 * router showed Home from every page. The link moves focus to the main region
 * itself and never touches the hash; the href stays for its link semantics.
 */
export function SkipLink() {
  const skip = (event: MouseEvent<HTMLAnchorElement>) => {
    event.preventDefault();
    const main = document.getElementById(MAIN_CONTENT_ID);
    main?.focus();
    main?.scrollIntoView?.({ block: "start" });
  };
  return <a className="skip-link" href={`#${MAIN_CONTENT_ID}`} onClick={skip}>Skip to main content</a>;
}
