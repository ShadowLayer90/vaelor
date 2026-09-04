import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

/**
 * Follow/pause for a conversation that scrolls with the page.
 *
 * `useFollowScroll` needs a scroll container of its own. The Assistant's
 * transcript no longer has one: a fixed-height panel with `overflow-y: auto`
 * around a second `overflow-y: auto` stream measured 0px of visible transcript
 * at 1280x720 and 1024x768 against 2,187px and 2,874px of content, so the
 * answer was invisible while its chrome was not. The page is the scroller now,
 * and "am I looking at the newest turn?" is answered by where the end of the
 * transcript sits in the viewport.
 *
 * The contract is the same one the stream had: follow while the reader is at
 * (or near) the latest turn, release the moment they scroll up to read
 * something earlier, resume when they come back.
 */
export const LATEST_THRESHOLD_PX = 96;

/** An explicit `behavior: "smooth"` overrides the CSS-level reduced-motion
 *  stylesheet, so motion started from script has to check the query itself. */
function prefersReducedMotion() {
  return typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function revealAnchor(anchor: HTMLElement, behavior: ScrollBehavior) {
  if (typeof anchor.scrollIntoView !== "function") return;
  const resolved = behavior === "smooth" && prefersReducedMotion() ? "auto" : behavior;
  anchor.scrollIntoView({ behavior: resolved, block: "end" });
}

export function useLatestInView<T extends HTMLElement>(
  anchorRef: RefObject<T | null>,
  dependencies: ReadonlyArray<unknown>,
  { threshold = LATEST_THRESHOLD_PX, enabled = true }: { threshold?: number; enabled?: boolean } = {},
) {
  const [following, setFollowing] = useState(true);
  // Read inside effects without making them re-run when only the flag changes.
  const followingRef = useRef(following);
  followingRef.current = following;

  const atLatest = useCallback(() => {
    const anchor = anchorRef.current;
    if (!anchor || typeof anchor.getBoundingClientRect !== "function") return true;
    const viewport = window.innerHeight || document.documentElement.clientHeight;
    return anchor.getBoundingClientRect().bottom <= viewport + threshold;
  }, [anchorRef, threshold]);

  const scrollToLatest = useCallback((behavior: ScrollBehavior = "auto") => {
    const anchor = anchorRef.current;
    if (anchor) revealAnchor(anchor, behavior);
    setFollowing(true);
  }, [anchorRef]);

  useEffect(() => {
    if (!enabled) return;
    const onScroll = () => setFollowing(atLatest());
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [atLatest, enabled]);

  // Reveal new content only when the reader has not deliberately scrolled up.
  useEffect(() => {
    const anchor = anchorRef.current;
    if (!anchor || !enabled || !followingRef.current) return;
    // "auto" rather than smooth: streamed updates land in rapid succession and
    // queued smooth animations fight each other and lag behind the content.
    revealAnchor(anchor, "auto");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anchorRef, enabled, ...dependencies]);

  return { following, scrollToLatest, setFollowing };
}
