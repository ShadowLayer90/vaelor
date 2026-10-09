"""Built-in explanations that need no reading from this machine.

These are the last resort in :meth:`DeploymentAgent._fallback_answer`, reached
only when no fact tool matched: what Docker is, what KVM is not, what the three
Assistant runs on. They are constants, they take no evidence, and they were
sitting in the middle of a method whose other branches all interpret live
readings - which made a 980-line module out of two unrelated jobs.

Extracted here so the answer path keeps room to grow (VD-030: the ceiling that
only speaks when it is already too late is a wall, not a warning). Nothing about
the behaviour changes: same phrases, same order, same "only if nothing else
answered" rule, which the caller enforces by asking last.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .assistant_console_places import CONNECTIONS_PLACE
from .phrase_match import mentions


#: Trigger phrases and the paragraph they produce, tried in order. The order is
#: load-bearing where the vocabularies overlap: "how do you work" must reach the
#: capability answer rather than the greeting.
STATIC_ANSWERS: Tuple[Tuple[Sequence[str], str], ...] = (
    (
        ("hello", "hi", "hey", "good morning", "good evening"),
        "Hi. I’m Vaelor Assistant. I can explain live readings, "
        "diagnose common cooling, display, network, and service problems, and "
        "guide Docker, local-AI, update, and remote-access setup.",
    ),
    (
        ("what can you do", "help me", "how do you work", "your purpose"),
        "I can read this node’s live hardware and service state, explain "
        "it in beginner language, review Docker and AI deployments, and prepare "
        "safe next steps. I do not silently change the machine: installs, "
        "settings, updates, and resets must be reviewed and approved "
        "separately.",
    ),
    (
        ("kvm", "vnc", "remote desktop"),
        "KVM and VNC solve different problems. Physical KVM controls another "
        "computer through HDMI capture plus isolated keyboard and mouse "
        "hardware. Host VNC opens this host desktop over the network, even when "
        "physical KVM is not installed. App VNC opens only a container’s "
        "desktop.",
    ),
    (
        ("docker", "compose", "container"),
        "Docker runs applications in isolated containers. Compose is the YAML "
        "file that describes containers, ports, storage, and restart policy as "
        "one app. Vaelor can validate a Compose project before deployment and "
        "keeps the actual change behind a review step.",
    ),
    (
        ("llm", "model", "local ai", "openai", "endpoint", "api"),
        # Review B6 (VD-049, VD-201): the Assistant runs Vaelor's own model or
        # built-in basic mode, and nothing configured changes that. A hosted
        # API or an OpenAI-compatible endpoint is connected for AI Chat.
        "The Assistant runs Vaelor's own model on this machine, or built-in "
        "basic mode when that model is not running; it never uses a model "
        "you connect. Models you connect - a hosted API or an "
        "OpenAI-compatible endpoint such as LM Studio or llama.cpp - are used "
        "in AI Chat, set up under " + CONNECTIONS_PLACE + ".",
    ),
    (
        ("ram", "memory", "storage", "disk"),
        "RAM is short-term working space for running apps and AI models; "
        "storage is persistent NVMe or SD space for files, containers, and "
        "downloads. High RAM use can slow or stop workloads, while low storage "
        "can block installs and updates.",
    ),
)

def static_answer(message: str) -> Optional[List[str]]:
    """The first built-in explanation this message asks for, or ``None``.

    One matcher for every topic. The KVM entry used a bare substring test while
    its five neighbours used word boundaries; the special case was carried
    across the extraction, then mutation-tested and found to change nothing -
    ``mentions`` matches every phrasing the substring version did. A
    distinction that cannot be made to matter is not a distinction.
    """
    lower = str(message or "").lower()
    for phrases, paragraph in STATIC_ANSWERS:
        if mentions(lower, tuple(phrases)):
            return [paragraph]
    return None
