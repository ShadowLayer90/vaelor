"""A worker's software state against its controller, and the card's words (VD-194 P1).

Design §3a.7 and §6. Pure: a stored record (the last reading and the last
attempt), the controller's expectation and the time in; one state and every
sentence the Fleet card shows out. The card draws these verbatim (VD-173) and
holds no word of its own.

**Every state is dated and has an exit** (LESSONS 22, LESSONS 8). Nothing reads
``current`` without the time it was read (design G9), a reading older than
:data:`STALE_AFTER_SECONDS` says so, and :data:`STATES` names, for each state,
the owner action that leaves it.

The full appliance outranks the profile. A worker that still runs the twelve
appliance services is described by that first, because it is what will decide
the next step once conversion (design §4) is built - it is not built yet, and
the card says so - and its ownership differences are the appliance's own.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

from . import worker_profile as wp
from .api_host_settings_routes import STALE_AFTER_SECONDS
from .worker_profile_compare import DIFFERS, MATCHES, NO_STATE, UNREAD, compare

#: Said wherever the full appliance is on a worker: today nothing converts it.
#: Found on the Z2, 2026-10-04 (LESSONS 10): the exit read "Conversion" while
#: only the read-only measurement was built.
NOT_CONVERTIBLE_YET = "Converting this machine to the worker profile is not available yet"

#: Each state: its label, its tone, what it means, what it allows, and its exit.
#: The card draws ``label``, ``tone`` and the exit (:func:`exit_words`, as the
#: pill's hover and screen-reader text); ``meaning`` travels in the payload.
#: Every word here is read by an operator, so none names a design reference
#: (LESSONS 10 / 24, guarded in ``tests/test_worker_profile_state.py``).
#:
#: Reachable today: current, behind and unknown from a reading alone; the
#: full-appliance states from the probe; behind-queued and update-failed from
#: the newest profile job (P2, `worker_profile_job`). Two are not reachable
#: yet, and their words say what they will mean: behind-held-serving needs a
#: profile change marked disruptive (none is), and ahead needs this
#: controller's history of the profiles it shipped (not kept yet).
STATES: Dict[str, Dict[str, str]] = {
    "current": {
        "label": "Up to date", "tone": "success",
        "meaning": "Measured matches this controller's profile, read within 15 minutes.",
        "allowed": "Everything.", "exit": "A profile change or a hand edit; Recheck re-reads it."},
    "current-reading-old": {
        "label": "Up to date at last check", "tone": "neutral",
        "meaning": "Matched at its last reading, which is more than 15 minutes old.",
        "allowed": "Everything; the reading is shown with its age.",
        "exit": "Recheck."},
    "behind": {
        "label": "Behind", "tone": "warning",
        "meaning": "Differs from this controller's profile, and no update of it is queued.",
        "allowed": "Serving continues; new deploys allowed.",
        # Recheck queues an update when a part the profile owns differs
        # (cluster_worker_profile.recheck_worker_profile); the 15-minute check
        # does too while the machine answers, until SWEEP_TRIES updates for the
        # same profile have failed (worker_profile_job).
        "exit": ("Recheck reads this machine again and queues an update of the parts that differ. "
                 "While the machine answers, the 15-minute check queues one too, until three "
                 "updates of the same change have failed.")},
    "behind-unreachable": {
        "label": "Behind, not reachable", "tone": "warning",
        "meaning": "Differs, and the machine did not answer at the last attempt.",
        "allowed": "Its deployments show as they are.",
        "exit": ("The 15-minute check queues the update once the machine answers; "
                 "Recheck does it at once.")},
    "behind-queued": {
        "label": "Updating", "tone": "info",
        "meaning": "Differs; a profile update for it is queued or running.",
        "allowed": "Serving continues, and new deploys are allowed while it updates.",
        "exit": "The update finishes and reads the machine again."},
    "behind-held-serving": {
        "label": "Behind, held while serving", "tone": "warning",
        "meaning": "Only changes that would interrupt serving differ, and a loaded deployment names it.",
        "allowed": "Serving continues; the held change waits.",
        "exit": "Unload or remove that deployment, or choose to update now."},
    "update-failed": {
        "label": "Update failed", "tone": "danger",
        "meaning": "The last profile update failed; the line below says why.",
        "allowed": "Serving continues; deploy checks treat it as behind.",
        "exit": "Recheck tries again, after any fix the line below names."},
    "ahead": {
        "label": "Ahead of this controller", "tone": "warning",
        "meaning": "The worker records a profile this controller does not know.",
        "allowed": "As for behind.",
        "exit": "The controller re-applies its own profile."},
    # Reachable today: the full appliance is read from the probe alone. Nothing
    # converts, stops or removes it yet, so none of these offers that.
    "full-appliance-installed": {
        "label": "Full appliance installed", "tone": "warning",
        "meaning": "The full appliance's services are on this machine and running.",
        "allowed": "Serving continues as it is; the worker profile is laid down beside the appliance.",
        "exit": NOT_CONVERTIBLE_YET + "; until it is, this machine keeps serving as it is."},
    "full-appliance-stopped": {
        "label": "Appliance stopped", "tone": "info",
        "meaning": "Every appliance service is present, stopped and disabled.",
        "allowed": "Serving continues; the stopped services stay as they are.",
        "exit": ("Recheck after the appliance's services are changed. "
                 + NOT_CONVERTIBLE_YET + ".")},
    "full-appliance-partial": {
        "label": "Partly installed", "tone": "warning",
        "meaning": "Some appliance services are present, or they are in mixed states.",
        "allowed": "Serving continues; nothing is done automatically.",
        "exit": ("Recheck once the appliance's services are put right by hand. "
                 + NOT_CONVERTIBLE_YET + ".")},
    "unknown": {
        "label": "Unknown", "tone": "neutral",
        "meaning": "Never measured, or not readable.",
        "allowed": "Nothing claims current; serving and deploys carry on unaffected.",
        "exit": "Recheck."},
}

NEVER_CHECKED = "never checked"
STALE_SENTENCE = "reading is old — Recheck"
#: The same, for a viewer who cannot press Recheck: it is administrator-only
#: (VD-189 / LESSONS 19, frontend review FE-2).
STALE_SENTENCE_OTHERS = "reading is old — an administrator can Recheck"
NOT_CHECKED_SENTENCE = "Not checked yet. Recheck reads it."
NOT_CHECKED_SENTENCE_OTHERS = "Not checked yet. An administrator can Recheck it."
#: Added to a state's exit for a viewer who cannot press Recheck (round 2 F2).
ADMINISTRATOR_ONLY = "Only an administrator can Recheck."
NO_REASON = "no reason was given"
#: What the card says when it cannot show a reading for a reason that is not a
#: failed check (round 4 N3). Fixed words: the cause is in the log, never raw
#: exception text in front of an operator (LESSONS 24, N6).
NOT_SHOWN_INTERNAL = "Vaelor couldn't show this machine's software (an internal error, logged)."
NOT_CHECKED_INTERNAL = "Vaelor couldn't check this machine's software (an internal error, logged)."
NOT_CHECKED_STORE = "Vaelor couldn't check this machine's software: its records could not be read (logged)."
NOT_SAVED = "Vaelor read this machine but couldn't save the reading (logged). Recheck tries again."
STORED_UNREADABLE = "This machine's stored reading could not be read (logged). Recheck replaces it."
NOT_ENROLLED = "This machine is no longer enrolled."
#: Its exit (P1-R4-5, LESSONS 22): a Recheck of a machine that is gone is a
#: 404, so the way back is to add it again, which is administrator-only.
NOT_ENROLLED_EXIT = {"administrator": "Add this machine again to check its software.",
                     "other": "An administrator can add this machine again to check its software."}
#: A node row whose own fields will not decode (P1-R4-1). Recheck reads that
#: same row and fails on it, so it is not offered. PH-R2: a machine still
#: awaiting its join can be removed under Setup, which tolerates the record;
#: a joined one cannot be drained through a plan that reads the same row.
NODE_RECORD_UNREADABLE = "This machine's enrolment record could not be read (logged)."
_RECORD_BY_HAND = ("Otherwise the record has to be repaired or removed on this controller by hand; "
                   "the log names its node id.")
NODE_RECORD_EXIT = {
    "administrator": ("If it is listed under Setup as awaiting its join, remove it there; Vaelor deletes "
                      "the record and its sign-in. " + _RECORD_BY_HAND),
    "other": ("If it is listed under Setup as awaiting its join, an administrator can remove it there. "
              + _RECORD_BY_HAND),
}
#: A stored reading that carries no time (RW-L5): it is not taken as the state.
UNDATED_READING = "This reading has no time recorded, so it is not taken as this machine's state."
#: The pill for a view that shows no reading for one of the reasons above.
NOT_SHOWN_LABEL = "Not shown"
#: A check that ran and failed, with its reason (round 2 F1).
CHECK_FAILED = "The check failed: {}."

#: How the worker compares with this controller's profile, for the states
#: whose own sentence is about something else (owner check 2026-10-04, review
#: round 2). Read from :func:`profile_verdict` - the comparison
#: :func:`derive_state` makes - never from a second one (LESSONS 6), and empty
#: where the sentence already says it: one fact, said once.
PROFILE_SUMMARY = {
    "appliance-matches": "Beside the appliance, every part this controller expects is in place.",
    "appliance-differs": "Beside the appliance, these differ from this controller's profile: {}.",
    "appliance-unread": "Beside the appliance, some parts could not be read: {}.",
    "appliance-check-unread": ("This machine is not compared with this controller's profile until "
                               "the appliance check can be read."),
}
#: An unknown reading whose parts could not all be read (review round 2).
NOT_COMPARED_UNREAD = "Not compared with this controller's profile, because some parts could not be read: {}."
PROFILE_CODES_SUMMARY = "Profile codes"
#: Each code's label, and what it reads when there is no code to show.
PROFILE_CODE_LABELS = {
    "measured": "Read from this machine",
    "expected": "This controller expects for this machine",
    "release": "This release, for any worker, before per-machine settings",
}
MEASURED_UNREAD = "not worked out: some parts could not be read"
MEASURED_NEVER = "not checked yet"
CODE_NOT_WORKED_OUT = "could not be worked out"

#: How a reading's age is worded: below ``below`` seconds (``None``: always),
#: the age divided by ``unit`` fills ``{n}``. The backend's own words and the
#: card's live ones both come from this one table (round 2 F4, LESSONS 6).
AGE_WORDS_TABLE = (
    {"below": 60, "unit": 0, "words": "checked just now"},
    {"below": 2 * 3600, "unit": 60, "words": "checked {n} min ago"},
    {"below": 2 * 86400, "unit": 3600, "words": "checked {n} h ago"},
    {"below": None, "unit": 86400, "words": "checked {n} days ago"},
)

_RUNNING_STATES = ("active", "activating", "reloading", "deactivating")


def age_words(checked_at: Optional[float], now: float) -> str:
    """How long ago a reading was taken, in the card's words."""
    if not _dated(checked_at):
        return NEVER_CHECKED
    seconds = max(0.0, float(now) - float(checked_at))
    step = next(row for row in AGE_WORDS_TABLE if row["below"] is None or seconds < row["below"])
    if not step["unit"]:
        return step["words"]
    return step["words"].replace("{n}", str(int(seconds // step["unit"])))


def undated_words(record: Mapping[str, Any], now: float) -> Dict[str, Any]:
    """What the card draws for a reading that carries no time (design G9).

    Never the state's own label or tone, which could claim "Up to date"
    (FE-1). A first check that failed is a check: it says so, when, and why
    (round 2 F1, LESSONS 8) - not "not checked".
    """
    unshown = record.get("unshown")
    if isinstance(unshown, dict):
        return {"label": unshown.get("label") or NOT_SHOWN_LABEL, "tone": "warning",
                "checked": age_words(unshown.get("at"), now) if unshown.get("at") else "",
                "sentence": {"administrator": unshown["sentence"], "other": unshown["sentence"]}}
    attempt = record.get("attempt") or {}
    if attempt.get("ok") is False:
        failed = CHECK_FAILED.format(attempt.get("reason") or NO_REASON)
        return {"label": "Check failed", "tone": "warning", "checked": age_words(attempt.get("at"), now),
                "sentence": {"administrator": failed, "other": failed}}
    return {"label": "Not checked", "tone": "neutral", "checked": NEVER_CHECKED,
            "sentence": {"administrator": NOT_CHECKED_SENTENCE, "other": NOT_CHECKED_SENTENCE_OTHERS}}


def _state_words(state: Mapping[str, str], disabled: bool, reload_needed: bool) -> str:
    """One appliance unit's state in words (round 2 F5): failed is failed, unread is unread."""
    active = state.get("ActiveState")
    if not active:
        lead = "state not read"
    elif active in _RUNNING_STATES:
        lead = "running"
    elif active == "failed":
        lead = "failed"
    else:
        lead = "stopped"
    return ", ".join(word for word, holds in (
        (lead, True), ("disabled", disabled), ("reload pending", reload_needed)) if holds)


def _dated(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _unit_words(name: str, state: Mapping[str, str]) -> str:
    words = "{} ({}, {})".format(name, state.get("ActiveState") or NO_STATE,
                                 state.get("UnitFileState") or NO_STATE)
    if state.get("NeedDaemonReload") == "yes":
        words += ", reload pending"
    return words


def appliance_reading(appliance: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Whether the full appliance is on the worker: none, running, stopped, partial or unread.

    Counted from the probe's answer for each unit name the controller's wheel
    ships. An answer with no units at all is unread, never "none" - an empty
    list is what a failed read looks like (LESSONS 8).
    """
    appliance = appliance or {}
    units = appliance.get("units") if isinstance(appliance.get("units"), dict) else {}
    if appliance.get("error") or not units:
        why = appliance.get("error") or "the machine reported no appliance services"
        return {"reading": "unread", "units": [], "present": 0, "running": 0,
                "sentence": "Whether the full appliance is on this machine could not be read: {}.".format(
                    why.rstrip("."))}
    rows = []
    for name in sorted(units):
        state = units[name]
        present = state.get("LoadState") not in ("not-found", "", None)
        rows.append({"name": name, "present": present,
                     "running": present and state.get("ActiveState") in _RUNNING_STATES,
                     "disabled": state.get("UnitFileState") in ("disabled", "masked"),
                     "reload_needed": state.get("NeedDaemonReload") == "yes",
                     "words": _unit_words(name, state)})
        row = rows[-1]
        row["state_words"] = _state_words(state, row["disabled"], row["reload_needed"])
    present = [row for row in rows if row["present"]]
    running = [row for row in present if row["running"]]
    total = len(rows)
    if not present:
        reading = "none"
        sentence = "None of the appliance's {} services is on this machine.".format(total)
    elif len(present) == total and len(running) == total:
        reading = "running"
        sentence = "Full Vaelor appliance still installed on this machine ({} services found).".format(total)
    elif len(present) == total and not running and all(row["disabled"] for row in present):
        reading = "stopped"
        sentence = "Full appliance stopped on this machine ({} services present, none running).".format(total)
    else:
        reading = "partial"
        absent = total - len(present)
        sentence = "Partly installed: {}{}.".format(
            "; ".join(row["words"] for row in present),
            "; {} not present".format(absent) if absent else "")
    return {"reading": reading, "units": [{key: row[key] for key in
                                           ("name", "present", "running", "disabled", "reload_needed",
                                            "state_words")}
                                          for row in rows],
            "present": len(present), "running": len(running), "sentence": sentence}


_APPLIANCE_STATES = {"running": "full-appliance-installed", "stopped": "full-appliance-stopped",
                     "partial": "full-appliance-partial"}


def appliance_present(appliance: Mapping[str, Any]) -> bool:
    """Whether any of the full appliance is on the worker, by :func:`appliance_reading`."""
    return appliance.get("reading") in _APPLIANCE_STATES


def profile_verdict(rows: List[Mapping[str, Any]], measured_digest: Optional[str],
                    expected_digest: str) -> str:
    """The one answer to "does this worker match this controller's profile?".

    ``unread`` while any part is unread, ``differs`` when any row differs or
    the digests do, else ``matches``. Rows count as well as digests because
    the fingerprint leaves some fields out (a file's type), so a row can
    differ under equal digests (review round 2, LESSONS 6). :func:`derive_state`
    and the card's profile line both read this; nothing compares again.
    """
    if measured_digest is None or any(row["status"] == UNREAD for row in rows):
        return UNREAD
    if any(row["status"] == DIFFERS for row in rows) or measured_digest != expected_digest:
        return DIFFERS
    return MATCHES


#: What a list of parts reads as when no row names one: a difference only in
#: the digest, or a reading with nothing in it.
_NO_LABEL = {DIFFERS: "its digest", UNREAD: "the reading"}


def _labels(rows: List[Mapping[str, Any]], status: str) -> str:
    return ", ".join(row["label"] for row in rows if row["status"] == status) or _NO_LABEL[status]


def derive_state(*, record: Mapping[str, Any], rows: List[Mapping[str, Any]],
                 appliance: Mapping[str, Any], measured_digest: Optional[str],
                 expected_digest: str, now: float, job: Optional[Mapping[str, Any]] = None,
                 serving_hold: bool = False, history: Iterable[str] = (),
                 marker_digest: str = "") -> str:
    """The one state, by the precedence of design §3a.7.

    ``job`` is the newest profile job for this worker (`worker_profile_job.card_job`);
    ``serving_hold`` is the planner's verdict that every difference would
    interrupt a loaded deployment (P3); ``history`` is the profile digests this
    controller has shipped, which is what tells "ahead" from "behind".
    """
    if not record.get("reading") or not _dated(record.get("checked_at")):
        return "unknown"
    if appliance["reading"] == "unread":
        return "unknown"
    if appliance["reading"] in _APPLIANCE_STATES:
        return _APPLIANCE_STATES[appliance["reading"]]
    verdict = profile_verdict(rows, measured_digest, expected_digest)
    if verdict == UNREAD:
        return "unknown"
    if verdict == DIFFERS:
        job_state = str((job or {}).get("state") or "")
        attempt = record.get("attempt") or {}
        if job_state in ("queued", "running"):
            return "behind-queued"
        if job_state == "failed":
            return "update-failed"
        if (attempt.get("ok") is False and attempt.get("unreachable")
                and float(attempt.get("at") or 0) >= float(record["checked_at"])):
            return "behind-unreachable"
        if serving_hold:
            return "behind-held-serving"
        known = set(history)
        if known and marker_digest and marker_digest != expected_digest and marker_digest not in known:
            return "ahead"
        return "behind"
    if float(now) - float(record["checked_at"]) > STALE_AFTER_SECONDS:
        return "current-reading-old"
    return "current"


def _sentence(state: str, rows: List[Mapping[str, Any]], appliance: Mapping[str, Any],
              record: Mapping[str, Any], job: Optional[Mapping[str, Any]]) -> str:
    attempt = record.get("attempt") or {}
    if state in _APPLIANCE_STATES.values():
        return appliance["sentence"]
    differing = _labels(rows, DIFFERS)
    if state == "unknown":
        if record.get("reading") and not _dated(record.get("checked_at")):
            return UNDATED_READING
        if not record.get("reading"):
            if isinstance(record.get("unshown"), dict):
                return record["unshown"]["sentence"]
            if attempt.get("ok") is False:
                return CHECK_FAILED.format(attempt.get("reason") or NO_REASON)
            return NOT_CHECKED_SENTENCE
        if appliance["reading"] == "unread":
            return appliance["sentence"]
        return NOT_COMPARED_UNREAD.format(_labels(rows, UNREAD))
    if state in ("current", "current-reading-old"):
        return "Worker software matches this controller's profile."
    if state == "update-failed":
        # The job's own sentence names the part and the reason (P2).
        return str((job or {}).get("message") or "The last update did not finish.")
    if state == "behind-unreachable":
        # Owner check 2026-10-04: a label such as "Telemetry agent (Telegraf)"
        # carries brackets, so the list is never wrapped in brackets of its own.
        return "Differs from this controller's profile: {}. The machine did not answer: {}.".format(
            differing, attempt.get("reason") or NO_REASON)
    return "Differs from this controller's profile: {}.".format(differing)


def worker_software_view(*, node_id: str, record: Optional[Mapping[str, Any]],
                         expected: Mapping[str, Any], release_digest: str,
                         now: Optional[float] = None, job: Optional[Mapping[str, Any]] = None,
                         serving_hold: bool = False, history: Iterable[str] = (),
                         update_note: str = "") -> Dict[str, Any]:
    """Everything the card's "Worker software" row shows, in words; no raw reading."""
    now = time.time() if now is None else float(now)
    record = dict(record or {})
    reading = record.get("reading") if isinstance(record.get("reading"), dict) else None
    appliance = appliance_reading(reading.get("appliance") if reading else {"error": "never read"})
    kept = record.get("amd_smi") if isinstance(record.get("amd_smi"), dict) else None
    comparison = compare(expected, reading, appliance["reading"] in _APPLIANCE_STATES, kept) if reading else {
        "components": [], "measured_digest": None}
    items = (reading or {}).get("items") or {}
    marker = items.get("marker") if isinstance(items.get("marker"), dict) else {}
    state = derive_state(
        record=record, rows=comparison["components"], appliance=appliance,
        measured_digest=comparison["measured_digest"], expected_digest=expected["digest"],
        now=now, job=job, serving_hold=serving_hold, history=history,
        marker_digest=str(marker.get("profile_digest") or ""),
    )
    checked_at = record.get("checked_at") if _dated(record.get("checked_at")) else None
    stale = checked_at is not None and now - float(checked_at) > STALE_AFTER_SECONDS
    words = STATES[state]
    attempt = record.get("attempt") or {}
    # P1-R4-5 (LESSONS 22): a view that shows no reading may carry an exit of
    # its own, where the state's Recheck would not leave it.
    unshown = record.get("unshown") if isinstance(record.get("unshown"), dict) else {}
    exits = dict(unshown["exit_words"]) if isinstance(unshown.get("exit_words"), dict) else exit_words(state)
    return {
        "node_id": node_id, "state": state, "label": words["label"], "tone": words["tone"],
        "meaning": words["meaning"], "exit": exits["administrator"],
        "sentence": _sentence(state, comparison["components"], appliance, record, job),
        "checked_at": int(checked_at) if checked_at is not None else None,
        "checked": age_words(checked_at, now), "stale": stale,
        "stale_sentence": STALE_SENTENCE if stale else "",
        # The card also works staleness out itself from checked_at against
        # this window (FE-5), so a payload held in a cache still ages.
        "stale_after_seconds": STALE_AFTER_SECONDS,
        "old_reading_words": {"administrator": STALE_SENTENCE, "other": STALE_SENTENCE_OTHERS},
        "undated": undated_words(record, now),
        "exit_words": exits,
        # Round 3 R3-4: what a current reading turns into once the card has
        # held it past the window - the backend's own words, so the card never
        # shows "Up to date" beside "reading is old".
        "when_old": ({"label": STATES["current-reading-old"]["label"],
                      "tone": STATES["current-reading-old"]["tone"],
                      "exit_words": exit_words("current-reading-old")} if state == "current" else None),
        # The card measures a reading's age on this box's clock (round 2 F4):
        # (served_at - checked_at) plus the time since the payload arrived.
        "served_at": int(now),
        "age_words_table": [dict(row) for row in AGE_WORDS_TABLE],
        "profile_summary": profile_summary(state, comparison["components"], appliance,
                                           comparison["measured_digest"], expected["digest"]),
        "profile_codes": profile_codes(reading, comparison["measured_digest"], expected["digest"],
                                       release_digest),
        "components": [{key: row[key] for key in ("id", "label", "status", "word", "why", "note",
                                                   "appliance_owned", "measured", "words")}
                       for row in comparison["components"]],
        "appliance": {"reading": appliance["reading"], "sentence": appliance["sentence"],
                      "units": appliance["units"]},
        "attempt_note": _attempt_note(attempt, now) if reading else "",
        # P2: the newest profile update, queued, running or finished, in the
        # job's own words (VD-173); "" when there has been none.
        "update_note": update_note,
    }


def profile_summary(state: str, rows: List[Mapping[str, Any]], appliance: Mapping[str, Any],
                    measured_digest: Optional[str], expected_digest: str) -> str:
    """How the worker compares with this controller's profile, where its sentence does not say.

    Derived from the state and :func:`profile_verdict`, so it cannot contradict
    the pill (review round 2, LESSONS 6): "matches" is only ever the current
    states' own sentence, an unknown reading claims nothing, and a full
    appliance is described beside the appliance, never as up to date.
    """
    if state in _APPLIANCE_STATES.values():
        verdict = profile_verdict(rows, measured_digest, expected_digest)
        if verdict == UNREAD:
            return PROFILE_SUMMARY["appliance-unread"].format(_labels(rows, UNREAD))
        if verdict == DIFFERS:
            return PROFILE_SUMMARY["appliance-differs"].format(_labels(rows, DIFFERS))
        return PROFILE_SUMMARY["appliance-matches"]
    if state == "unknown" and rows and appliance["reading"] == "unread":
        return PROFILE_SUMMARY["appliance-check-unread"]
    return ""


def profile_codes(reading: Optional[Mapping[str, Any]], measured_digest: Optional[str],
                  expected_digest: str, release_digest: str) -> Dict[str, Any]:
    """The three profile codes, each labelled; the card keeps them under a disclosure.

    The release code is the whole release's profile with placeholders where
    each machine's own settings go, so it never equals a worker's code; it says
    which release a controller runs, not whether a worker matches (LESSONS 5).
    """
    if measured_digest:
        measured = wp.short(measured_digest)
    else:
        measured = MEASURED_UNREAD if reading else MEASURED_NEVER
    values = {"measured": measured,
              "expected": wp.short(expected_digest) or CODE_NOT_WORKED_OUT,
              "release": wp.short(release_digest) or CODE_NOT_WORKED_OUT}
    return {"summary": PROFILE_CODES_SUMMARY,
            "rows": [{"id": key, "label": PROFILE_CODE_LABELS[key], "value": values[key]}
                     for key in ("measured", "expected", "release")]}


def exit_words(state: str) -> Dict[str, str]:
    """A state's exit for whoever may press Recheck, and for everyone else (VD-189)."""
    exit_text = STATES[state]["exit"]
    other = exit_text if "Recheck" not in exit_text else "{} {}".format(exit_text, ADMINISTRATOR_ONLY)
    return {"administrator": exit_text, "other": other}


def _attempt_note(attempt: Mapping[str, Any], now: float) -> str:
    """Why the newest check failed, beside the older reading still shown."""
    if attempt.get("ok") is not False:
        return ""
    when = age_words(attempt.get("at"), now)
    when = when[len("checked "):] if when.startswith("checked ") else "at an unknown time"
    return "The last check ({}) failed: {}.".format(when, attempt.get("reason") or NO_REASON)
