"""What a deployed cluster agent's attached skills hand it: scopes and guidance.

Moved out of :mod:`vaelor.agent_pool_operations`, which sat at the line
ceiling, so the deploy, the relaunch, the reconcile's change check and the
review screen's preview all read ONE derivation (LESSONS pattern 6).

A skill attached to a deployed agent does two things, and both are reported:

* **It adds read scopes.** The skill's grants, intersected with the read-only
  vocabulary (:data:`~vaelor.custom_agents.ALLOWED_SCOPES`), are unioned into
  the agent's own scopes. An acting permission or a declared broker credential
  refuses the deploy instead (a deployed agent is read-only).
* **It sends guidance.** The skill's name, description and how-to body are
  appended to the agent's instructions under :data:`SKILL_BLOCK_HEADER`,
  bounded so the block cannot overflow the model window.

The bound used to cut text and drop whole skills with no word to anyone, while
a dropped skill's scopes stayed granted (ACC-138). Now every attached skill
gets a guidance row saying exactly what was sent - ``sent`` whole,
``shortened`` (with the characters sent out of the total), or ``not-sent``
with the reason - and a skill whose guidance is not sent grants NO scope: an
agent never holds a permission for a skill it was never told about.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .custom_agents import ALLOWED_PERMISSIONS, ALLOWED_SCOPES
from .skills_library import _allowed_grants

#: At most this many attached skills are sent; later ones are reported
#: ``not-sent`` and grant nothing.
MAX_SKILLS_IN_INSTRUCTIONS = 8
#: Per-skill budgets for the one-line heading.
MAX_SKILL_NAME_CHARS = 80
MAX_SKILL_DESCRIPTION_CHARS = 160
#: A skill's how-to body is sent up to this many characters; a longer body is
#: sent shortened and reported so.
MAX_SKILL_INSTRUCTIONS_CHARS = 1200
#: The whole attached-skills block, header included, stays under this.
MAX_SKILL_BLOCK_CHARS = 4000
SKILL_BLOCK_HEADER = "Attached skills:"

#: What happened to one skill's guidance. Written into the deploy result and
#: the review preview; the console renders each.
GUIDANCE_SENT = "sent"
GUIDANCE_SHORTENED = "shortened"
GUIDANCE_NOT_SENT = "not-sent"

#: Why a skill's guidance was not sent - one sentence per cause, each saying
#: that its scopes were withheld too.
_OVER_SKILL_COUNT = (
    "Only the first {} attached skills are sent to the agent, so this one is "
    "not sent and its read scopes are not granted."
)
_OVER_BLOCK_BUDGET = (
    "The attached-skill guidance is limited to {} characters and this skill "
    "does not fit, so it is not sent and its read scopes are not granted."
)

#: Deploy-time skill refusals (fail closed), phrased distinctly from one
#: another and from the definition refusals so no two share a literal.
SKILL_MISSING = (
    "The attached skill {!r} is missing from the library or has been disabled."
)
SKILL_ACTING_REFUSED = (
    "A read-only cluster agent cannot take on attached skill {!r}, whose grant "
    "{} is an acting capability."
)
SKILL_CREDENTIALS_REFUSED = (
    "The attached skill {!r} declares broker credentials, which no read-only "
    "agent surface consumes, so it is refused."
)
SKILL_UNSAFE_TEXT = (
    "The attached skill {!r} carries a name, description, or instructions that "
    "failed the instruction safety screen."
)


class SkillRefused(ValueError):
    """A strict (deploy-time) resolve met an attached skill it must refuse."""


def pinned_skill_ids(skills: Any) -> List[str]:
    """The de-duplicated skill ids from a pinned ``[{skill_id, version}]`` list.

    The pinned ``version`` is intentionally ignored: the skills library is
    unversioned, so a skill is always resolved against its live manifest.
    """
    ids: List[str] = []
    for entry in skills or []:
        if not isinstance(entry, Mapping):
            continue
        skill_id = str(entry.get("skill_id") or "").strip()
        if skill_id and skill_id not in ids:
            ids.append(skill_id)
    return ids


def resolve_skill(library: Any, skill_id: str) -> Optional[Mapping[str, Any]]:
    """The live library manifest for one pinned skill id, or ``None``.

    A library never injected, or any lookup trouble, is reported as not-found -
    the fail-safe direction.
    """
    if library is None:
        return None
    try:
        record = library.get(str(skill_id))
    except Exception:  # noqa: BLE001 - a lookup failure is a not-found skill
        return None
    return record if isinstance(record, Mapping) else None


def classify_skill(
    record: Optional[Mapping[str, Any]], skill_id: str, allowed: Any,
) -> Tuple[Optional[str], List[str]]:
    """``(reason, read_scopes)`` for one resolved skill.

    ``reason`` is ``None`` when the skill is read-only-safe to attach, and
    ``read_scopes`` is then its grants restricted to the read vocabulary. A
    non-``None`` reason is one of the fail-closed refusals.
    """
    if record is None or not record.get("enabled", False):
        return SKILL_MISSING.format(skill_id), []
    kept = [
        grant
        for grant in (str(item).strip() for item in record.get("grants") or [])
        if grant in allowed
    ]
    acting = sorted(grant for grant in kept if grant in ALLOWED_PERMISSIONS)
    if acting:
        return SKILL_ACTING_REFUSED.format(skill_id, ", ".join(acting)), []
    if record.get("credentials"):
        return SKILL_CREDENTIALS_REFUSED.format(skill_id), []
    name = str(record.get("name") or "")
    description = str(record.get("description") or "")
    instructions = str(record.get("instructions") or "")
    # The register-time screens, reused (defense in depth). Imported here so
    # this adds no instrumented compiled-pattern binding at module level.
    from .custom_agents import UNSAFE_INSTRUCTIONS, UNSAFE_INSTRUCTIONS_BODY

    if (UNSAFE_INSTRUCTIONS.search(name) or UNSAFE_INSTRUCTIONS.search(description)
            or UNSAFE_INSTRUCTIONS_BODY.search(instructions)):
        return SKILL_UNSAFE_TEXT.format(skill_id), []
    return None, sorted(grant for grant in kept if grant in ALLOWED_SCOPES)


def _entry_text(item: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """One skill's block entry and the facts about what it cut."""
    raw_name = " ".join(str(item.get("name") or "").split())
    raw_description = " ".join(str(item.get("description") or "").split())
    raw_body = str(item.get("instructions") or "").rstrip()
    name = raw_name[:MAX_SKILL_NAME_CHARS]
    description = raw_description[:MAX_SKILL_DESCRIPTION_CHARS]
    body = raw_body[:MAX_SKILL_INSTRUCTIONS_CHARS]
    line = "- " + name
    if description:
        line += ": " + description
    # The how-to body follows its heading with markdown newlines intact.
    entry = line + "\n" + body if body else line
    facts = {
        "total_chars": len(raw_name) + len(raw_description) + len(raw_body),
        "sent_chars": len(name) + len(description) + len(body),
        "shortened": (
            len(name) < len(raw_name) or len(description) < len(raw_description)
            or len(body) < len(raw_body)
        ),
    }
    return entry, facts


def guidance_block(accepted: List[Mapping[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Render the bounded block and one guidance row per accepted skill.

    Skills are taken in their pinned order. A skill past
    :data:`MAX_SKILLS_IN_INSTRUCTIONS`, or whose entry would carry the block
    past :data:`MAX_SKILL_BLOCK_CHARS`, is ``not-sent`` with its reason; a
    later, smaller skill may still fit. A sent skill whose name, description
    or body was cut to its budget is ``shortened``. Each row carries the
    scopes the skill grants - none for a skill that was not sent.
    """
    lines = [SKILL_BLOCK_HEADER]
    total = len(SKILL_BLOCK_HEADER)
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(accepted):
        entry, facts = _entry_text(item)
        row: Dict[str, Any] = {
            "skill_id": str(item.get("skill_id") or ""),
            "name": str(item.get("name") or ""),
            "total_chars": facts["total_chars"],
            "sent_chars": 0,
            "reason": "",
            "scopes": [],
        }
        if index >= MAX_SKILLS_IN_INSTRUCTIONS:
            row.update(status=GUIDANCE_NOT_SENT,
                       reason=_OVER_SKILL_COUNT.format(MAX_SKILLS_IN_INSTRUCTIONS))
        elif total + 1 + len(entry) > MAX_SKILL_BLOCK_CHARS:
            row.update(status=GUIDANCE_NOT_SENT,
                       reason=_OVER_BLOCK_BUDGET.format(MAX_SKILL_BLOCK_CHARS))
        else:
            lines.append(entry)
            total += 1 + len(entry)
            row.update(
                status=GUIDANCE_SHORTENED if facts["shortened"] else GUIDANCE_SENT,
                sent_chars=facts["sent_chars"],
                scopes=list(item.get("scopes") or []),
            )
        rows.append(row)
    block = "\n".join(lines) if len(lines) > 1 else ""
    return block, rows


def skill_surface(library: Any, skills: Any, *, strict: bool) -> Dict[str, Any]:
    """Resolve pinned skills to granted read scopes, the block, and the report.

    The one derivation behind the deploy, the relaunch, the reconcile's change
    check and the review preview. ``strict`` decides the unsafe case: on deploy
    the first unsafe skill raises :class:`SkillRefused` with its reason; on
    relaunch it is dropped into ``refusals`` so a reboot never strands a
    healthy agent. ``scopes`` is the union over skills whose guidance was sent
    (whole or shortened) only.
    """
    allowed = _allowed_grants()
    accepted: List[Dict[str, Any]] = []
    refusals: List[Dict[str, str]] = []
    for skill_id in pinned_skill_ids(skills):
        record = resolve_skill(library, skill_id)
        reason, read_scopes = classify_skill(record, skill_id, allowed)
        if reason is not None or record is None:
            if strict:
                raise SkillRefused(reason)
            refusals.append({"skill_id": skill_id, "reason": str(reason)})
            continue
        accepted.append({
            "skill_id": skill_id,
            "name": str(record.get("name") or ""),
            "description": str(record.get("description") or ""),
            "instructions": str(record.get("instructions") or ""),
            "scopes": read_scopes,
        })
    block, guidance = guidance_block(accepted)
    scopes = sorted({scope for row in guidance for scope in row["scopes"]})
    return {
        "scopes": scopes,
        "instructions": block,
        "accepted": accepted,
        "refusals": refusals,
        "guidance": guidance,
    }


def append_skill_block(instructions: Any, block: Any) -> str:
    """Join the pre-bounded "Attached skills:" block onto base instructions.

    The base instructions are capped at custom-agent registration and the block
    by :func:`guidance_block`, so this only joins the two with a blank line.
    """
    base = str(instructions or "")
    extra = str(block or "")
    if not extra:
        return base
    if not base:
        return extra
    return base + "\n\n" + extra
