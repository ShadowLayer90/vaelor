"""Which model credential may serve the Assistant, and which may serve AI Chat.

**One home for both answers** (VD-049, VD-201, VD-202; LESSONS 6). The vault asks
them when a lease is assigned (`CredentialVault.activate`), and `/copilot/setup`
asks the Assistant one when it reports whose model the Assistant is on, so a
refusal and a report can never describe one credential two ways.

- **The Assistant** (``deployment-agent``) runs only on its own fixed install:
  the on-device NPU (FLM) model, or the managed local model a box without an
  NPU runs it on. Not a connection the owner added, and not a model Vaelor runs
  for something else - a cluster or pooled deployment, or the GPU model it
  serves AI Chat. "Vaelor wrote this credential" was the first test and it was
  too wide (VD-202 item 1): a cluster credential has no owner either.
- **AI Chat** (``ai-chat``) never runs on the Assistant's NPU model (VD-007:
  AI Chat runs on the GPU; VD-202 item 2). A single-model box (a Pi) still
  shares its one llama.cpp model between the two, as it always has.

**How each is marked.** Every credential Vaelor writes for a model it runs on
this appliance has an id starting :data:`MANAGED_LOCAL_PREFIX`, and the id is
chosen by the deploy, never by a route. The NPU deploy writes
:data:`ASSISTANT_NPU_PREFIX`; the AI Chat deploys (the GPU fork and the GGUF
deploy for the ``ai-chat`` surface) write :data:`AI_CHAT_MODEL_PREFIX`. A credential written before those two markers
existed is told apart the way the credential sweep always has: an NPU
credential is pinned to the FLM tag it serves (``pins_an_flm_tag``). An
AI Chat model credential from before the marker cannot be told from a Pi's
Assistant model, so it still reads as the Assistant's own until its next
deploy rewrites it - the known gap, written down rather than guessed at.

Leaf module: it imports nothing that imports the broker, so the vault can
import it.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .inference_model_choice import INSTALLED_FLM_MODELS

#: The Assistant's lease, and AI Chat's.
ASSISTANT_PURPOSE = "deployment-agent"
AI_CHAT_PURPOSE_NAME = "ai-chat"

#: Every credential Vaelor writes for a model it runs on this appliance.
MANAGED_LOCAL_PREFIX = "cred_managed_local_"
_NPU_MARKER = "npu_"
_AI_CHAT_MARKER = "chat_"
#: The Assistant's on-device NPU model (written by the flm-real deploy).
ASSISTANT_NPU_PREFIX = MANAGED_LOCAL_PREFIX + _NPU_MARKER
#: A model Vaelor serves AI Chat (the GPU fork and the ``ai-chat`` GGUF deploy).
AI_CHAT_MODEL_PREFIX = MANAGED_LOCAL_PREFIX + _AI_CHAT_MARKER

#: The digest a managed-local deploy ends its id with.
_MANAGED_ID_DIGEST = re.compile(r"[a-f0-9]{12,64}")
#: The markers that may sit between the prefix and the digest, one at most.
_ID_MARKERS = (_NPU_MARKER, _AI_CHAT_MARKER)


def is_managed_local_credential_id(value: Any) -> bool:
    """Whether ``value`` is a whole id a managed-local deploy writes.

    The prefix, at most one marker, and the 12-character (or longer) hex
    digest. The one answer to "is this one of Vaelor's own model credentials",
    built from the prefixes above so a new marker cannot be missed by a reader
    that kept its own copy (LESSONS 6 / 14, VD-202: the Models list's private
    regex lost every marked id). The markers are checked one by one rather
    than as a regex alternation, so each is a line a test can delete.
    """
    text = str(value or "")
    if not text.startswith(MANAGED_LOCAL_PREFIX):
        return False
    rest = text[len(MANAGED_LOCAL_PREFIX):]
    for marker in _ID_MARKERS:
        if rest.startswith(marker):
            rest = rest[len(marker):]
            break
    return bool(_MANAGED_ID_DIGEST.fullmatch(rest))

#: What the vault answers when anything tries to give the Assistant a model that
#: is not its own install (VD-049 / VD-201 / VD-202).
ASSISTANT_TAKES_ONLY_ITS_OWN_MODEL = (
    "Vaelor Assistant runs only on the model Vaelor installs for it, so this "
    "connection cannot be assigned to it. A connection you add, or a cluster "
    "model, serves AI Chat instead."
)

#: What the vault answers when anything tries to put AI Chat on the NPU.
AI_CHAT_NEVER_ON_THE_ASSISTANT_NPU = (
    "AI Chat never runs on the Assistant's on-device (NPU) model; it uses a GPU "
    "model or a connection you add."
)


def _field(credential: Any, name: str) -> str:
    """One field of a vault row or a ``list`` item, ``""`` when absent."""
    try:
        return str(credential[name] or "")
    except (KeyError, IndexError, TypeError):
        return ""


def _vaelor_runs_it_here(credential: Any) -> bool:
    """A credential a Vaelor deploy wrote for a model on this appliance."""
    return (
        _field(credential, "provider") == "openai-compatible"
        and not _field(credential, "owner").strip()
        and _field(credential, "id").startswith(MANAGED_LOCAL_PREFIX)
    )


def carries_the_npu_marker(credential: Any) -> bool:
    """The NPU marker alone: the id, or - written before it - the FLM pin.

    The managed-local tier split (`managed_local_credentials._tiers`) asks only
    this, of ids it already knows are managed-local.
    """
    return (
        _field(credential, "id").startswith(ASSISTANT_NPU_PREFIX)
        or _field(credential, "selected_model") in INSTALLED_FLM_MODELS
    )


def is_the_assistants_npu_model(credential: Any) -> bool:
    """Whether this is the Assistant's on-device NPU model.

    A model Vaelor runs here, marked by its id, or - written before the marker
    - by the FLM tag it is pinned to (``selected_model``), the test the
    credential sweep uses.
    """
    return _vaelor_runs_it_here(credential) and carries_the_npu_marker(credential)


def assistant_may_use(credential: Any) -> bool:
    """Whether the Assistant's lease may name this credential (VD-202 item 1).

    Its own fixed install only: a model Vaelor runs on this appliance that is
    not one it runs for AI Chat. Anything unreadable is refused.
    """
    return _vaelor_runs_it_here(credential) and not _field(
        credential, "id"
    ).startswith(AI_CHAT_MODEL_PREFIX)


def ai_chat_may_use(credential: Any) -> bool:
    """Whether AI Chat's lease may name this credential (VD-202 item 2)."""
    return not is_the_assistants_npu_model(credential)


def ai_chat_may_use_lease(lease: Any, listing: Any) -> bool:
    """Whether AI Chat may send to a resolved ``ai-chat`` lease (VD-210).

    A lease from before VD-202 can still name the Assistant's NPU model; the
    vault resolves it though it refuses to assign it. Asked of
    :func:`ai_chat_may_use` over the listed row for the lease's credential.
    A credential missing from ``listing`` is judged on the lease itself
    (its id, provider and model), so an unlisted NPU lease still fails.
    """
    credential_id = _field(lease, "credential_id")
    row = next((item for item in listing or () if _field(item, "id") == credential_id), None)
    if row is None:
        row = {
            "id": credential_id,
            "provider": _field(lease, "provider"),
            "selected_model": _field(lease, "model"),
        }
    return ai_chat_may_use(row)


def role_refusal(purpose: str, credential: Mapping[str, Any]) -> str:
    """The sentence refusing ``purpose`` for ``credential``, or ``""``."""
    if purpose == ASSISTANT_PURPOSE and not assistant_may_use(credential):
        return ASSISTANT_TAKES_ONLY_ITS_OWN_MODEL
    if purpose == AI_CHAT_PURPOSE_NAME and not ai_chat_may_use(credential):
        return AI_CHAT_NEVER_ON_THE_ASSISTANT_NPU
    return ""
