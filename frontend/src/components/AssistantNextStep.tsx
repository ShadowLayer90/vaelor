import { destinationName } from "../lib/destinations";
import { hashForPage, type NavigationPage } from "../lib/navigation";
import type { AssistantNavigationStep } from "./agentTypes";

/**
 * Suggested next steps that name a place must name a place that exists.
 *
 * The answer engine still writes pre-rename destination names — `Open Workloads
 * to inspect logs...` in `vaelor/deployment_agent.py` and `Open Workloads >
 * Install...` in `vaelor/assistant_policy.py` — while the rail has said "Apps
 * and AI" since the canonical names landed. A reader was being sent to a
 * destination that is not in the navigation, in plain grey text they could not
 * click. Rewriting the mention here fixes both at once and keeps working
 * whatever the server says next.
 *
 * Deliberately conservative: only product-specific terms are matched. "System"
 * and "Home" are ordinary English in this context and linking them would turn
 * prose into a minefield of accidental navigation.
 */
const destinationMentions: ReadonlyArray<{ term: string; page: NavigationPage }> = [
  { term: "Apps and AI", page: "workloads" },
  { term: "Remote console", page: "kvm" },
  { term: "AI Chat", page: "ai-chat" },
  { term: "Workloads", page: "workloads" },
  { term: "Activity", page: "activity" },
  { term: "Cluster", page: "fleet" },
  { term: "Settings", page: "admin" },
];

export interface DestinationMention {
  index: number;
  length: number;
  page: NavigationPage;
}

/** The earliest destination named in a step, longest match winning a tie. */
export function findDestinationMention(text: string): DestinationMention | null {
  let best: DestinationMention | null = null;
  for (const { term, page } of destinationMentions) {
    const index = text.search(new RegExp(`\\b${term.replaceAll(" ", "\\s+")}\\b`, "i"));
    if (index < 0) continue;
    const match = text.slice(index).match(new RegExp(`^${term.replaceAll(" ", "\\s+")}`, "i"));
    const length = match?.[0].length ?? term.length;
    if (!best || index < best.index || (index === best.index && length > best.length)) {
      best = { index, length, page };
    }
  }
  return best;
}

export function AssistantNextStep({ action }: { action: string }) {
  const mention = findDestinationMention(action);
  if (!mention) return <>{action}</>;
  return (
    <>
      {action.slice(0, mention.index)}
      <a href={hashForPage(mention.page)}>{destinationName(mention.page)}</a>
      {action.slice(mention.index + mention.length)}
    </>
  );
}

/**
 * Only a hash route inside this app may be rendered as a destination.
 *
 * `navigation_steps` builds these from a fixed table and never from the
 * model's words, so today nothing else can arrive here. The guard is kept
 * anyway because the answer it travels with *is* partly model-authored, and
 * the cost of an off-site link appearing as "where to go next" inside the
 * appliance's own Assistant is not worth the two lines it saves.
 */
export function isInAppRoute(route: string): boolean {
  return /^#\/[\w./-]*$/.test(String(route ?? "")) || route === "#/";
}

/**
 * Where an answer said to go, as links.
 *
 * The Assistant redirects general-knowledge questions to AI Chat (VD-035), and
 * the backend has emitted `#/ai-chat` alongside that sentence all along — two
 * backend tests pin it. The client printed the sentence and dropped the route,
 * so on the Pi the nearest thing to a link under a redirect went to Memory.
 * Prose that names a place and a route that reaches it are two halves of one
 * answer; this is the second half.
 */
export function AssistantAnswerDestinations({
  steps,
}: {
  steps?: AssistantNavigationStep[];
}) {
  const routes = (steps ?? []).filter((step) => isInAppRoute(step.route));
  if (!routes.length) return null;
  return (
    <nav aria-label="Where to go next" className="assistant-destinations">
      <strong>Go to</strong>
      <ul>
        {routes.map((step) => (
          <li key={`${step.route}-${step.control}-${step.label}`}>
            <a href={step.route}>{step.label || step.control}</a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
