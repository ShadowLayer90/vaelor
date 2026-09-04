import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

/**
 * Keeps a scrolling region pinned to its newest content without stealing the
 * viewport from a reader.
 *
 * The region follows while the user is at (or near) the bottom, releases as
 * soon as they scroll up to read earlier content, and resumes the moment they
 * return to the bottom. This is the same follow/pause contract the log viewer
 * uses, generalised so every append path — optimistic echo, pending state,
 * answer, error and cancellation — reveals itself the same way.
 */
export const FOLLOW_THRESHOLD_PX = 80;

export function distanceFromBottom(element: HTMLElement) {
  return element.scrollHeight - element.scrollTop - element.clientHeight;
}

/**
 * An explicit `behavior: "smooth"` overrides the CSS `scroll-behavior`
 * property, so the global `prefers-reduced-motion` stylesheet cannot suppress
 * it. Motion started from script has to check the query itself.
 */
function prefersReducedMotion() {
  return typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function pinToBottom(element: HTMLElement, behavior: ScrollBehavior) {
  const resolved = behavior === "smooth" && prefersReducedMotion() ? "auto" : behavior;
  if (typeof element.scrollTo === "function") {
    element.scrollTo({ top: element.scrollHeight, behavior: resolved });
  } else {
    element.scrollTop = element.scrollHeight;
  }
}

export function useFollowScroll<T extends HTMLElement>(
  containerRef: RefObject<T | null>,
  dependencies: ReadonlyArray<unknown>,
  { threshold = FOLLOW_THRESHOLD_PX, enabled = true }: { threshold?: number; enabled?: boolean } = {},
) {
  const [following, setFollowing] = useState(true);
  // Read inside effects without making them re-run when only the flag changes.
  const followingRef = useRef(following);
  followingRef.current = following;

  // A programmatic scroll emits the same events as a user one. Without this
  // guard, an animated scroll reports itself as "the user scrolled away" on
  // every intermediate frame and flips follow off and on again mid-animation.
  const programmatic = useRef(false);

  const scrollToBottom = useCallback((behavior: ScrollBehavior = "auto") => {
    const element = containerRef.current;
    if (!element) return;
    // Only an animated scroll needs the guard: it emits intermediate events
    // that look like the reader scrolling away. An instant pin lands at the
    // bottom in one event and needs no suppression — latching the flag for it
    // would swallow the reader's next genuine scroll-away.
    programmatic.current = behavior === "smooth";
    pinToBottom(element, behavior);
    setFollowing(true);
  }, [containerRef]);

  // Track intent: scrolling away from the bottom pauses follow, returning resumes it.
  useEffect(() => {
    const element = containerRef.current;
    if (!element || !enabled) return;
    const onScroll = () => {
      const atBottom = distanceFromBottom(element) <= threshold;
      if (programmatic.current) {
        // Stay in follow mode until the animation lands.
        if (!atBottom) return;
        programmatic.current = false;
      }
      setFollowing(atBottom);
    };
    element.addEventListener("scroll", onScroll, { passive: true });
    return () => element.removeEventListener("scroll", onScroll);
  }, [containerRef, enabled, threshold]);

  // Reveal new content only when the reader has not deliberately scrolled up.
  useEffect(() => {
    const element = containerRef.current;
    if (!element || !enabled || !followingRef.current) return;
    // "auto" rather than smooth: streamed updates land in rapid succession and
    // queued smooth animations fight each other and lag behind the content.
    pinToBottom(element, "auto");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [containerRef, enabled, ...dependencies]);

  /*
   * Content keeps growing after the first paint — evidence disclosures, fonts
   * and images all settle late. Scrolling once on mount therefore left the
   * view part-way up a restored conversation, and because that position was
   * never a user action, follow stayed nominally on while the newest turn sat
   * off-screen. Re-pin whenever the scrollable height changes and the reader
   * has not scrolled away.
   *
   * Growth is only observable through the children. The container is a scroll
   * box whose own border box stays exactly the same size as its content grows,
   * so a ResizeObserver watching it alone never fires. The child list is not
   * fixed either: every new turn is a new element, and an observer set up once
   * at mount would never see any of them — which is what made the effect
   * inert for the case it was written for. The membership is therefore
   * maintained as the list changes.
   */
  useEffect(() => {
    const element = containerRef.current;
    if (!element || !enabled || typeof ResizeObserver === "undefined") return;
    let lastHeight = element.scrollHeight;
    const repin = () => {
      if (element.scrollHeight === lastHeight) return;
      lastHeight = element.scrollHeight;
      if (!followingRef.current) return;
      pinToBottom(element, "auto");
    };
    const sizes = new ResizeObserver(repin);
    sizes.observe(element);
    for (const child of Array.from(element.children)) sizes.observe(child);

    const membership = typeof MutationObserver === "undefined" ? null : new MutationObserver((records) => {
      for (const record of records) {
        for (const node of Array.from(record.removedNodes)) {
          if (node instanceof Element) sizes.unobserve(node);
        }
        for (const node of Array.from(record.addedNodes)) {
          if (node instanceof Element) sizes.observe(node);
        }
      }
      // Insertion is itself a height change; do not wait for a later resize.
      repin();
    });
    membership?.observe(element, { childList: true });

    return () => {
      sizes.disconnect();
      membership?.disconnect();
    };
  }, [containerRef, enabled]);

  return { following, scrollToBottom, setFollowing };
}
