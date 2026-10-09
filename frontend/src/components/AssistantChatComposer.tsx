import { type FormEvent, type KeyboardEvent, type ReactNode, useEffect, useId, useRef, useState } from "react";
import { Button } from "./ui";

/**
 * Shared "still working" affordance.
 *
 * Chat had elapsed time and a stop control while an appliance check of the
 * same length showed only a changed button label, so the slowest surface was
 * the one with no sign of life. `onCancel` is optional: a check that cannot be
 * interrupted still shows progress rather than pretending to be instant.
 */
export function AssistantResponseStatus({
  active,
  onCancel,
  label,
  startedAt = 0,
}: {
  active: boolean;
  onCancel?: () => void;
  label?: string;
  /**
   * When the request began, as epoch milliseconds.
   *
   * The counter used to start from the moment this component mounted, and this
   * panel is unmounted whenever the reader looks at another Assistant tab: a
   * request at "110s elapsed" came back reading "3s elapsed" after a round
   * trip to History, so the one number telling somebody how long the appliance
   * has really been working was reset by looking away from it. Zero falls back
   * to mount time for a caller that has no start to hand over.
   */
  startedAt?: number;
}) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!active) return;
    const origin = startedAt || Date.now();
    const tick = () => setElapsed(Math.max(0, Math.floor((Date.now() - origin) / 1000)));
    tick();
    const interval = window.setInterval(tick, 1000);
    return () => window.clearInterval(interval);
  }, [active, startedAt]);
  if (!active) return null;
  return (
    <div className="as-working" role="status">
      <span aria-hidden="true" className="as-working__dot" />
      <div className="as-working__copy">
        <strong>{label ?? (elapsed < 3 ? "Reading live appliance data" : "Preparing a concise answer")}</strong>
        <small>{elapsed}s elapsed</small>
      </div>
      {onCancel && <Button className="as-working__stop" onClick={onCancel} type="button">Stop response</Button>}
    </div>
  );
}

export function AssistantChatComposer({
  blocked = false,
  busy,
  children,
  input,
  onChange,
  onSubmit,
  submitLabel = "Ask Vaelor",
}: {
  /**
   * The question as typed cannot be sent yet.
   *
   * Separate from `busy` on purpose. Validation used to be folded into it, so
   * an over-length question disabled the button *and* relabelled it "Thinking…"
   * with nothing in flight — the product claimed to be working on a request it
   * had refused to accept. A blocked composer keeps its own label, which is
   * what tells the reader the send has not happened.
   */
  blocked?: boolean;
  busy: boolean;
  /**
   * What refines the question, inside the question box and above its send
   * button (VD-200, the AssistAnswer board): the armed-check banner and the
   * "Save this as a check I can re-run" disclosure.
   */
  children?: ReactNode;
  input: string;
  onChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  /**
   * What this submission will actually do.
   *
   * With "Keep this as a check I can re-run" ticked the button posted a
   * reviewable appliance check and answered nothing, while still reading "Ask
   * Vaelor" — the label described the other branch.
   */
  submitLabel?: string;
}) {
  const formRef = useRef<HTMLFormElement>(null);
  const canSend = !busy && !blocked && Boolean(input.trim());

  // Enter sends, Shift+Enter (and IME composition) still insert a newline.
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey || event.altKey) return;
    // keyCode 229 covers Windows IMEs that clear isComposing on the
    // composition-confirming Enter.
    if (event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229) return;
    // Only claim the key when it will actually send. Swallowing it while a
    // response streams would stop the user drafting a multi-line follow-up.
    if (!canSend || !formRef.current) return;
    event.preventDefault();
    formRef.current.requestSubmit();
  };

  const hintId = "assistant-question-hint-" + useId().replaceAll(":", "");
  return (
    <form className="as-composer ui-card" onSubmit={onSubmit} ref={formRef}>
      <label className="as-small as-muted" htmlFor="assistant-question">
        Ask a question or describe what you want to do
      </label>
      <textarea
        aria-describedby={hintId}
        className="as-composer__input"
        id="assistant-question"
        maxLength={4000}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Example: Check my cooling and tell me if anything needs attention."
        rows={3}
        value={input}
      />
      {/* The board draws no hint line; the keys are still said, to the reader who needs them. */}
      <span className="sr-only" id={hintId}>Press Enter to send, Shift+Enter for a new line.</span>
      {children}
      <div className="as-composer__foot">
        <span className="as-small as-muted">Live facts are read automatically. Changes always need a separate approval.</span>
        <Button
          disabled={!canSend}
          disabledReason={!busy && !blocked && !input.trim() ? "Type a question first" : undefined}
          type="submit"
          variant="primary"
        >
          {busy ? "Thinking…" : submitLabel}
        </Button>
      </div>
    </form>
  );
}
