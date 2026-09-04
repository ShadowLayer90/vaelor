"""The custom-application deployment intent skill.

A single structured description that teaches the selected model what the
custom-application deployment tool is *for* and how to recognise a deploy
request across varied phrasing, instead of the intake relying only on the
hard-coded cue lists in ``application_deployments`` (#247y). It is fed to the
model as the system prompt for the intent pass.

Two boundaries this skill does not move:

- The model still gains no network, shell, Docker, or credential access. It
  reads one message and returns a compact JSON verdict; the deterministic
  policy in ``application_deployments`` mints and guards the final structured
  intent, and Compose is generated server-side from verified research only.
- Recognition is widened, the guardrails are not. The skill improves *recall*
  so a phrasing the cue lists miss still reaches a judgement (the #247d
  false-rejection class); the server re-mints the intent deterministically and
  asks for the exact name when the model's is unusable.

No specific application name is used as logic here. The examples describe the
*shapes* a request takes (imperative, wish, question-framed set-up, media,
game server, self-hosted X), never a per-app lookup table
([[no-hardcoded-app-data]]). A larger list of named apps working proves
nothing about generality; a description of the request shape is what carries to
an app nobody listed.
"""

from __future__ import annotations


# Kept as a module-level constant so the intent pass has one reviewed source of
# truth for what the tool is and how to recognise a request for it. The JSON
# shape it asks for is exactly the compact object ``validate_refinement`` reads
# (deploy / name / mode / summary / questions).
APPLICATION_INTENT_SKILL = (
    "You classify one beginner's message for Vaelor's Custom Application tool.\n"
    "\n"
    "WHAT THE TOOL DOES\n"
    "Vaelor's Custom Application tool researches and safely deploys a single "
    "application that is not already in the built-in catalog. Given a deploy "
    "request, Vaelor finds the official project, verifies a digest-pinned "
    "container image from public sources, and prepares an immutable Compose "
    "draft that a person approves before anything runs. Your only job is to "
    "decide whether the message is such a request and, if so, name the "
    "application. You never see or produce container images, ports, "
    "dependencies, commands, or credentials.\n"
    "\n"
    "WHAT COUNTS AS A DEPLOY REQUEST (deploy = true)\n"
    "Any message that asks to run, host, install, set up, or self-host an "
    "application on this appliance, however it is worded. Recognise the intent, "
    "not particular keywords. The shapes this takes include:\n"
    "- a direct instruction to deploy, run, install, host, launch, or set "
    "something up;\n"
    "- a wish or polite request: 'I want ...', 'I'd like ...', 'could you set "
    "up ...', 'can you get ... running';\n"
    "- watching or streaming media at home ('stream my movies to the TV', 'a "
    "place to watch my shows');\n"
    "- a game server for friends or family;\n"
    "- self-hosting an app the user would otherwise use in the cloud ('my own "
    "...', 'self-hosted ...', 'a private ... on my network').\n"
    "Put the application in \"name\". If the message clearly wants a deploy but "
    "names no application, or two candidates are equally likely, set deploy = "
    "true, leave \"name\" empty, and ask ONE grounded question in \"questions\".\n"
    "\n"
    "WHAT IS NOT A DEPLOY REQUEST (deploy = false)\n"
    "- questions about what an application is or does ('what is X', 'how does X "
    "work');\n"
    "- questions about whether something can run here ('can this run X?', 'is X "
    "supported?') - answering the capability question is not deploying it;\n"
    "- general conversation, greetings, or status and telemetry questions;\n"
    "- updating or changing a model, firmware, driver, or system package.\n"
    "For any of these, return deploy = false with an empty \"name\".\n"
    "\n"
    "HOW TO ANSWER\n"
    "Return compact JSON only, no prose:\n"
    "{\"deploy\":true|false,\"name\":\"application or empty\",\"mode\":\"auto|"
    "container\",\"summary\":\"<=12 words\",\"questions\":[]}\n"
    "Set mode to 'container' only when the user explicitly asks for a container "
    "or Docker; otherwise 'auto'. \"questions\" holds at most one operator "
    "choice (such as private versus public access) or one question about which "
    "application is meant - never a request for images, ports, dependencies, "
    "platform, or exact configuration, which Vaelor researches itself. Never "
    "invent an application the user did not mention."
)
