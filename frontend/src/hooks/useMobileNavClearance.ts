import { type RefObject, useEffect } from "react";

/** The custom property the phone layout keeps the page clear of the bar by. */
export const MOBILE_NAV_BAR_HEIGHT = "--mobile-nav-bar-height";

/**
 * Publish the navigation bar's real height (W8-D2). Under a raised text size
 * the phone's bottom bar takes a second row, and a height guessed in CSS would
 * leave the page's last lines, its focus scrolling and the More sheet under
 * it. The stylesheet reads this through --mobile-nav-clearance and never lets
 * it fall below one row.
 */
export function useMobileNavClearance(bar: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const element = bar.current;
    const root = document.documentElement;
    if (!element || typeof ResizeObserver === "undefined") return undefined;
    const publish = () => root.style.setProperty(MOBILE_NAV_BAR_HEIGHT, `${Math.ceil(element.getBoundingClientRect().height)}px`);
    publish();
    const observer = new ResizeObserver(publish);
    observer.observe(element);
    return () => {
      observer.disconnect();
      root.style.removeProperty(MOBILE_NAV_BAR_HEIGHT);
    };
  }, [bar]);
}
