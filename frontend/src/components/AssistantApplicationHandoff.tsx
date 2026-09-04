import type { ApplicationResearchIntent } from "./agentTypes";
import { Button } from "./ui";

export function AssistantApplicationHandoff({ intent }: { intent: ApplicationResearchIntent }) {
  const continueResearch = () => {
    const request = `Deploy an application server ${intent.application_query}`;
    window.sessionStorage.setItem("vaelor.application-research-request", request);
    window.dispatchEvent(new CustomEvent("pironman:navigate", { detail: "workloads" }));
  };

  return <div className="assistant-next-actions assistant-application-handoff">
    <strong>Research before deployment</strong>
    <p>Vaelor will verify the official image, architecture, ports, storage, license, and memory requirements before proposing anything executable.</p>
    <Button onClick={continueResearch} type="button" variant="primary">
      Research and deploy {intent.application_query}
    </Button>
  </div>;
}
