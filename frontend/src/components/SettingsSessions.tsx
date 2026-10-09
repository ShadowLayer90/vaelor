import { Icon } from "./Icon";
import { usePagination } from "./PaginatedItems";
import { RecordPager, shortStamp } from "./RecordKit";
import { Button, Card, LoadingLines } from "./ui";
import "../styles/settings.css";

export interface ManagedSession {
  id: string;
  username: string;
  created_at: number;
  expires_at: number;
  last_seen_at: number;
  remote_addr: string;
  user_agent: string;
  current: boolean;
}

/** Signed-in devices a page shows (the SettingsSessions board). */
const SESSIONS_PER_PAGE = 8;

/**
 * Settings › Sessions (VD-200, the SettingsSessions board): every signed-in
 * device, newest activity as recorded, each one endable except your own,
 * which you end by signing out. Session tokens are never shown.
 */
export function SettingsSessions({ busy, onEnd, sessions }: {
  busy: string;
  onEnd: (id: string) => void;
  /** null before the list was read. */
  sessions: ManagedSession[] | null;
}) {
  const pages = usePagination(sessions ?? [], SESSIONS_PER_PAGE);
  return (
    <Card
      actions={sessions ? <span className="status-pill status-pill--neutral stg-count">{sessions.length} session{sessions.length === 1 ? "" : "s"}</span> : undefined}
      as="section"
      className="stg-sessions"
      description="End access you do not recognize. Session tokens are never shown."
      flush
      footer={sessions && sessions.length > 0
        ? <RecordPager label="Signed-in devices" note={`${SESSIONS_PER_PAGE} a page`} page={pages.page} setPage={pages.setPage} totalPages={pages.totalPages} />
        : undefined}
      heading="Signed-in devices"
    >
      {sessions === null && <LoadingLines label="Loading signed-in devices…" />}
      {pages.visible.map((item) => {
        const ending = busy === `session-${item.id}`;
        return (
          <div className="stg-row stg-row--session" key={item.id}>
            <span aria-hidden="true" className={item.current ? "ui-row__icon ui-row__icon--accent" : "ui-row__icon"}>
              <Icon name={item.current ? "shield" : "display"} size={18} />
            </span>
            <div className="stg-row__text">
              <strong>{item.username}{item.current ? " · this device" : ""}</strong>
              <small><span className="record-mono">{item.remote_addr || "Local"}</span> · last active {shortStamp(item.last_seen_at)}</small>
              <small>{item.user_agent || "Unknown client"}</small>
            </div>
            <div className="stg-row__end record-card-button">
              {item.current ? (
                <>
                  <Button className="record-danger-outline" disabled>Current</Button>
                  <small className="ui-muted">Sign out to end this one.</small>
                </>
              ) : (
                <Button
                  aria-label={`End session for ${item.username}`}
                  className="record-danger-outline"
                  disabled={ending}
                  onClick={() => onEnd(item.id)}
                >
                  {ending ? "Ending…" : "End session"}
                </Button>
              )}
            </div>
          </div>
        );
      })}
    </Card>
  );
}
