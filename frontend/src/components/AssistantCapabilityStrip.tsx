import { Icon } from "./Icon";
import { Button } from "./ui";

/**
 * The one-line answer to "what can this AI surface actually see?".
 *
 * Vaelor deliberately ships two AI destinations: the Assistant reads live
 * appliance readings and cannot read your files, AI Chat reads the documents
 * you add and cannot see the readings. Nothing on either screen said so, so the
 * split looked like duplication. This strip is shared rather than duplicated so
 * the two surfaces cannot drift into describing themselves differently.
 *
 * AI Chat renders the identical strip by passing its own `scope`; the memory
 * chip is the same appliance-wide store on both sides.
 *
 * It really is one line now. Four identical chips restated the scope, the
 * scope's qualifier and the answering model — all three already named on the
 * screen — while the only two carrying new information, "Remembers N" and "N
 * skills", were also the only two that could be clicked. Identical styling
 * made that invisible. The restatements are quiet text; the two live items are
 * inline links, and they are the only things here that look clickable.
 */
export interface AssistantCapabilityStripProps {
  /** What this surface can read. The first chip, and the only one that differs. */
  scope: { label: string; detail: string };
  /**
   * Appliance-wide memory. Omit for readers who may not open it: every memory
   * endpoint is administrator-only, so an entry point shown to an operator is
   * a link straight to an error.
   */
  memory?: { count: number; href: string };
  /**
   * Reviewed skills, and the disclosure that reveals them.
   *
   * Named "playbooks" here and skills everywhere else - in the panel this chip
   * opens, in its controls, and in `/assistant/skills` - so the first word the
   * reader met was the one the product never used again.
   */
  skills?: { count: number; expanded: boolean; onToggle: () => void };
  /** The model answering here, named exactly as the engine reports it. */
  model: string;
  label?: string;
}

export function AssistantCapabilityStrip({
  scope,
  memory,
  model,
  skills,
  label = "What this assistant can see",
}: AssistantCapabilityStripProps) {
  return (
    /*
     * The explicit spaces are load-bearing. Without them the headline and its
     * qualifier concatenate in the accessible name - "Remembers 7Shared with AI
     * Chat", "0 skillsReview" - which is what a screen reader announces. The
     * line is a flex row, and a whitespace-only text node is not a flex item,
     * so nothing about the layout changes.
     */
    <div aria-label={label} className="capability-strip" role="group">
      <span className="capability-note">
        <Icon name="cpu" size={15} />
        <strong>{scope.label}</strong>{" "}
        <small>{scope.detail} · {model}</small>
      </span>
      {memory && (
        <a className="capability-chip capability-chip--link" href={memory.href}>
          <Icon name="database" size={15} />
          <strong>Remembers {memory.count}</strong>{" "}
          <small>Shared with AI Chat</small>
        </a>
      )}
      {skills && (
        <Button
          aria-expanded={skills.expanded}
          className="capability-chip capability-chip--action"
          onClick={skills.onToggle}
          type="button"
          variant="quiet"
        >
          <Icon name="shield" size={15} />
          <strong>{skills.count} skill{skills.count === 1 ? "" : "s"}</strong>{" "}
          <small>{skills.expanded ? "Hide" : "Review"}</small>
        </Button>
      )}
    </div>
  );
}
