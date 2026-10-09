import type { RefObject } from "react";
import { StatusPill } from "./StatusPill";
import { Button, Notice } from "./ui";
import { sessionStateLabel, type RemoteSessionState } from "./remoteSessionState";

/**
 * The one-use desktop session (VD-200, the ConsoleSession board). It fills the
 * window over the dimmed page; Escape and Stop viewing close it. Every failure
 * is said inside it, with Retry session beside it, so the owner never has to
 * close it to try again.
 */
export function ConsoleSessionDialog({
  busy,
  canEnd,
  closeRef,
  dialogRef,
  endError,
  name,
  onEnd,
  onFrameError,
  onRetry,
  onStop,
  retryError,
  state,
  url,
}: {
  busy: string;
  /** Only the host's own desktop can be ended from here, and only by an operator. */
  canEnd: boolean;
  closeRef: RefObject<HTMLButtonElement | null>;
  dialogRef: RefObject<HTMLDivElement | null>;
  endError: string;
  name: string;
  onEnd: () => void;
  /** The frame itself failed to load. */
  onFrameError: () => void;
  onRetry: () => void;
  onStop: () => void;
  retryError: string;
  state: RemoteSessionState;
  url: string;
}) {
  return (
    <div aria-labelledby="remote-console-title" aria-modal="true" className="console-session" ref={dialogRef} role="dialog">
      <section className="console-session__card">
        <header className="console-session__header">
          <div className="ui-card__titles">
            <span className="sys-eyebrow">Protected one-use session</span>
            <h2 id="remote-console-title">{name}</h2>
          </div>
          <div className="console-session__actions">
            <StatusPill
              label={sessionStateLabel(state)}
              tone={state === "connected" ? "success" : state === "failed" ? "danger" : state === "connecting" ? "info" : "neutral"}
            />
            {/*
              * Two controls, because they do different things and one label
              * covering both is what stranded the owner: "Stop viewing" leaves
              * the desktop running; "End desktop" is the recovery route.
              */}
            <Button onClick={onStop} ref={closeRef} type="button" variant="secondary">Stop viewing</Button>
            {canEnd && (
              <Button disabled={busy === "end-desktop"} onClick={onEnd} type="button" variant="quiet">
                {busy === "end-desktop" ? "Ending…" : "End desktop"}
              </Button>
            )}
          </div>
        </header>
        {(endError || state === "failed") && (
          <div className="console-session__notices">
            {endError && <Notice heading="The desktop session was not ended." severity="warning">{endError}</Notice>}
            {state === "failed" && (
              <Notice className="console-session__retry" severity={retryError ? "danger" : "info"}>
                {retryError ? `${retryError}` : "The visible desktop session could not be reached."}
                <Button disabled={Boolean(busy)} onClick={onRetry} type="button" variant="primary">Retry session</Button>
              </Notice>
            )}
          </div>
        )}
        <iframe
          allow="clipboard-read; clipboard-write; fullscreen"
          sandbox="allow-forms allow-same-origin allow-scripts"
          onError={onFrameError}
          src={url}
          title={`${name} remote desktop`}
        />
        <footer className="console-session__footer">
          <span>Stop viewing leaves the desktop running on the appliance.</span>
          {canEnd && <span>End desktop closes it there; the next open starts a fresh one.</span>}
        </footer>
      </section>
    </div>
  );
}
