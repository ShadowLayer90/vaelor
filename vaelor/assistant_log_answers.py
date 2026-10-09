"""Read one managed service's journal for "are there errors in the logs" (review S7).

**The log tool could not be reached.** "Are there any errors in the logs?" was
answered with the health verdict ("no active warnings or alerts") - a different
question - because no word gathered ``logs.service``, and the tool needs a
``service`` argument the chat route never passed. The machine brief still told
the model to "ask for" it.

The route now names the service from the question (the control plane when none
is named), and this module answers from the lines that came back. Journal text
is machine output: a quoted line is evidence for the owner, never an
instruction, and the reading that failed is said as not read, without the raw
error (LESSONS 24).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Tuple

from .answer_evidence import add_evidence
from .assistant_fact_access import gathered_but_unread, unread, usable_fact
from .assistant_vocabulary import literal_phrases
from .phrase_match import mentions

#: The words that make a question about a service's log, read from the one
#: vocabulary that gathers ``logs.service`` so the two cannot drift.
LOG_WORDS: Tuple[str, ...] = literal_phrases("logs.service")

#: The managed service a question names, by the words an owner uses for it,
#: and how it is named back. Keys are `platform_drivers.VAELOR_LINUX_SERVICES`.
LOG_SERVICES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("workload-executor", ("executor", "workload executor", "job runner"), "the workload executor"),
    ("workload-broker", ("workload broker",), "the workload broker"),
    ("credential-broker", ("credential broker", "credentials service", "secrets service"),
     "the credential broker"),
    ("vnc-gateway", ("vnc", "vnc gateway"), "the VNC gateway"),
    ("vnc-tls", ("vnc tls", "tls proxy"), "the VNC TLS proxy"),
    ("control-plane", ("control plane", "dashboard", "web console", "api"), "the control plane"),
)

#: Read when nothing narrower is named: the service the console runs on.
DEFAULT_LOG_SERVICE = "control-plane"

#: A journal line that reports a failure, judged per line.
_ERROR_LINE = re.compile(
    r"\b(?:error|errors|failed|failure|fatal|critical|traceback|exception|panic)\b",
    re.IGNORECASE,
)

#: How much of one quoted line the answer carries.
_QUOTE_CHARACTERS = 200


def log_service(message: str) -> str:
    """The managed service a log question names, or the control plane."""
    text = str(message or "")
    for service, words, _name in LOG_SERVICES:
        if mentions(text, words):
            return service
    return DEFAULT_LOG_SERVICE


def logs_arguments(message: str) -> Dict[str, Any]:
    """The arguments the chat route passes to ``logs.service`` for this question."""
    return {"service": log_service(message), "lines": 200}


def _service_name(service: str) -> str:
    return next((name for key, _words, name in LOG_SERVICES if key == service),
                "the {} service".format(service))


def logs_line(message: str, facts: Mapping[str, Any], evidence: List[Dict[str, str]]) -> str:
    """One sentence about the errors in the service log this question read."""
    name = _service_name(log_service(message))
    if gathered_but_unread(facts, "logs.service"):
        return unread("{}'s log".format(name))
    logs = usable_fact(facts, "logs.service")
    if logs is None:
        return ""
    if not logs.get("available"):
        return unread("{}'s log".format(name))
    lines = [line for line in str(logs.get("output") or "").splitlines() if line.strip()]
    add_evidence(evidence, "logs.service", {"service": logs.get("service"), "lines": len(lines)},
                 ("service", "lines"))
    if not lines:
        return "{}'s log holds no lines in the window read, so there is no error in it to report.".format(
            name[:1].upper() + name[1:])
    errors = [line for line in lines if _ERROR_LINE.search(line)]
    if not errors:
        return ("None of the last {} of {}'s log mentions an error or a "
                "failure.".format(_lines(len(lines)), name))
    latest = " ".join(errors[-1].split())[:_QUOTE_CHARACTERS]
    return ("{} of the last {} of {}'s log mention an error or a failure. "
            "The most recent reads: \"{}\" (quoted from the log, as evidence).".format(
                len(errors), _lines(len(lines)), name, latest))


def _lines(count: int) -> str:
    return "{} line{}".format(count, "" if count == 1 else "s")
