"""The GPU memory pool: what it is now, what it may be set to, and what that leaves.

On a GPU that shares system memory (the Strix Halo appliances) the memory a
model can be placed in is the kernel's GTT aperture, and the aperture is the
TTM ``pages_limit`` byte for byte (VD-055, measured). The kernel sets that
limit to about half of the memory the operating system can see. Raising it is
what let a 27B and a 30B model load on the lab pair, and until VD-161 it was a
hand edit on each box.

This module is the ONE reading of that setting, for both kinds of machine
(LESSONS 6). Its input is a small record of facts a machine reports about
itself, and everything a screen, a job or the root side says about the pool is
derived from that record by :func:`pool_status`:

* The controller gathers the facts itself (:func:`read_local_facts`).
* An enrolled worker gathers them with :data:`REMOTE_FACTS_PROGRAM` over the
  SSH channel it was enrolled on, at enrolment and at every Recheck.

Both run the SAME gathering source (:data:`_FACTS_SOURCE`), so a worker and the
controller cannot describe a pool differently.

**The gatherer reports facts, not files.** It returns the running limit, the
memory total, the lines of Vaelor's own file, and - for every other modprobe
file - only its NAME and the lines that set this limit, plus which kernel
command-line overrides are present. A worker's whole ``modprobe.d``, its
command line and its memory table never leave the machine, are never stored,
and are never served (review S4): they can hold a Wi-Fi regulatory choice, a
disk-encryption hint or a serial console someone would not expect a fleet
console to keep.

**What is never done here.** Nothing in this module writes a file or restarts
a machine. The root side that applies a size is
:mod:`vaelor.gpu_memory_pool_apply`; the worker path is
:mod:`vaelor.gpu_memory_pool_nodes`. Both take their bounds from
:func:`require_size_gib`, so a size refused on the screen is refused at the
privileged boundary by the same rule.

**The ceiling is the fit engine's, not a second reserve.** The largest size on
offer is `cluster_gpu_sizing.achievable_gtt_ceiling` - system memory less the
share kept for the operating system, Docker and the control plane - which is
the figure the fit already prints as "the ceiling can reach". Two ceilings
would let the fit suggest a size this setting then refuses.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .cluster_gpu_sizing import VERDICT_WONT_FIT, achievable_gtt_ceiling
from .cluster_gpu_ways_out import POOL_SETTING_NAME

GIB = 1024 ** 3

#: What the setting is called wherever a person reads it, and where it lives.
#: The name's one home is the fit's own ways out (`cluster_gpu_ways_out`, merged
#: from w3-serving), so the fit's "a larger pool would make this fit" and the
#: screen that changes the pool cannot name it differently.
SETTING_NAME = POOL_SETTING_NAME
SETTING_PLACE = "Cluster > Setup > Machine settings"

#: The one file Vaelor writes for this setting, and the one line it holds.
#: Spelled here for the readers; the root side that writes it
#: (`gpu_memory_pool_apply`) imports both, so reader and writer agree.
CONFIG_PATH = "/etc/modprobe.d/vaelor-gpu-memory.conf"
CONFIG_TEMPLATE = "options ttm pages_limit={pages} page_pool_size={pages}\n"
CONFIG_HEADER = (
    "# Written by Vaelor (Cluster > Setup > Machine settings). The size of the\n"
    "# memory pool this machine's GPU may use. Change it there, not here.\n"
)

#: How far the running limit may sit from half of visible memory and still be
#: read as "the kernel's own default". The kernel takes its half when the
#: graphics memory module loads, early in boot, and the memory it can see grows
#: a little afterwards (the boot image's own memory is handed back), so the two
#: are close and not equal. Two percent is far below the smallest change this
#: setting can make (one GiB) on any machine it applies to. An UPPER BOUND on
#: that drift, not a measurement of it.
DEFAULT_TOLERANCE = 0.02

#: What any loose size is told, at every boundary that checks one.
WHOLE_NUMBER_REQUIRED = "Choose the GPU memory pool size as a whole number of GiB."

#: How this controller is named on the settings surface and in its job's
#: outcome; a worker is named by its own enrolled name.
CONTROLLER_NAME = "This controller"

OVERRIDE_ABSENT = "absent"
OVERRIDE_VAELOR = "vaelor"
OVERRIDE_UNRECOGNISED = "unrecognised"
OVERRIDE_UNREADABLE = "unreadable"

#: The kernel command-line parameters that set the GPU's memory directly, in
#: the gatherer's spelling, and what a person is told each one is.
_CMDLINE_NAMES = {
    "ttm.pages_limit": "the kernel command line (ttm.pages_limit)",
    "amdgpu.gttsize": "the kernel command line (amdgpu.gttsize)",
}

#: The gathering source both machines run. ``root`` exists so a test can point
#: it at a directory; production calls ``gather()`` bare.
#:
#: It reports FACTS: a file that is absent is ``None``/``absent``, a file that
#: exists and could not be read is ``False``/``unreadable`` (the two are never
#: merged - a file nobody could read must never be "restored" by deleting it),
#: and for a modprobe file that is not Vaelor's own only its name and the
#: lines that set this limit leave the machine. Vaelor's own file is read by
#: its one path, never through the directory listing, so a crowded directory
#: cannot hide it.
_FACTS_SOURCE = (
    "import glob, json, os, re\n"
    "def gather(root=''):\n"
    "    own = '/etc/modprobe.d/vaelor-gpu-memory.conf'\n"
    "    def read(name, limit=65536):\n"
    "        try:\n"
    "            with open(root + name, encoding='utf-8', errors='replace') as handle:\n"
    "                return handle.read(limit)\n"
    "        except FileNotFoundError:\n"
    "            return None\n"
    "        except OSError:\n"
    "            return False\n"
    "    def lines(text):\n"
    "        return [line.strip()[:200] for line in text.splitlines()\n"
    "                if line.strip() and not line.lstrip().startswith('#')]\n"
    "    sets_limit = re.compile(\n"
    "        r'options\\s+(?:ttm\\s(?:.*\\s)?pages_limit=|amdgpu\\s(?:.*\\s)?gttsize=)')\n"
    "    facts = {'others': [], 'unreadable': [], 'truncated': False}\n"
    "    text = read('/sys/module/ttm/parameters/pages_limit', 64)\n"
    "    facts['pages_limit'] = text.strip() if text else None\n"
    "    facts['mem_total_kb'] = None\n"
    "    for line in (read('/proc/meminfo') or '').splitlines():\n"
    "        if line.startswith('MemTotal:'):\n"
    "            fields = line.split()\n"
    "            facts['mem_total_kb'] = fields[1] if len(fields) > 1 else None\n"
    "            break\n"
    "    text = read(own)\n"
    "    facts['config'] = {\n"
    "        'state': 'absent' if text is None else 'unreadable' if text is False else 'present',\n"
    "        'lines': lines(text)[:16] if text else [],\n"
    "    }\n"
    "    seen = set()\n"
    "    for folder in ('/etc/modprobe.d', '/run/modprobe.d', '/usr/lib/modprobe.d', '/lib/modprobe.d'):\n"
    "        found = sorted(glob.glob(root + folder + '/*.conf'))\n"
    "        if len(found) > 256:\n"
    "            facts['truncated'] = True\n"
    "        for path in found[:256]:\n"
    "            name = path[len(root):].replace(os.sep, '/')\n"
    "            real = os.path.realpath(path)\n"
    "            if name == own or real in seen:\n"
    "                continue\n"
    "            seen.add(real)\n"
    "            text = read(name)\n"
    "            if text is False:\n"
    "                facts['unreadable'].append(name)\n"
    "                continue\n"
    "            hits = [line for line in lines(text or '') if sets_limit.match(line)]\n"
    "            if hits:\n"
    "                facts['others'].append({'path': name, 'lines': hits[:8]})\n"
    "    text = read('/proc/cmdline', 8192)\n"
    "    facts['cmdline'] = None if not text else [\n"
    "        name for name in ('ttm.pages_limit', 'amdgpu.gttsize')\n"
    "        if re.search(r'(?:^|\\s)' + re.escape(name) + '=', text)]\n"
    "    try:\n"
    "        facts['page_size'] = os.sysconf('SC_PAGE_SIZE')\n"
    "    except (AttributeError, OSError, ValueError):\n"
    "        facts['page_size'] = None\n"
    "    return facts\n"
)

#: What an enrolled worker runs (``python3 -c``) to report its own facts.
REMOTE_FACTS_PROGRAM = _FACTS_SOURCE + "print(json.dumps(gather()))\n"


def _gatherer() -> Callable[..., Dict[str, Any]]:
    namespace: Dict[str, Any] = {}
    exec(_FACTS_SOURCE, namespace)  # noqa: S102 - a module constant, never input
    return namespace["gather"]


def read_local_facts(root: str = "") -> Dict[str, Any]:
    """This machine's own facts, gathered by the source a worker runs."""
    return _gatherer()(root)


def parse_remote_facts(text: Any) -> Optional[Dict[str, Any]]:
    """A worker's answer to :data:`REMOTE_FACTS_PROGRAM`, or ``None`` if it is not one."""
    try:
        facts = json.loads(str(text))
    except (TypeError, ValueError):
        return None
    if not isinstance(facts, dict) or not isinstance(facts.get("config"), dict):
        return None
    return facts


def _positive_int(text: Any) -> Optional[int]:
    """``text`` as a positive whole number, or ``None``.

    ASCII digits only, one to twelve of them. ``int()`` alone takes ``+5``,
    ``1_000`` and digits from other scripts, and a value that reaches a root
    boundary must mean exactly one thing to everyone who reads it (review S5).
    """
    value = str(text).strip(" \t\r\n") if text is not None else ""
    if re.fullmatch(r"[0-9]{1,12}", value) is None:
        return None
    number = int(value)
    return number if number > 0 else None


def parse_config(config: Any) -> Dict[str, Any]:
    """What Vaelor's own file asks for, from the gatherer's ``config`` fact.

    ``vaelor`` is exactly one line, ``options ttm`` naming ``pages_limit`` and
    ``page_pool_size`` with the same positive number and nothing else. That is
    also the line the lab boxes carry from the hand edit, which is why an
    upgrade adopts it as the current setting instead of replacing it. Any
    other content is ``unrecognised``: reported, never reinterpreted. A file
    that exists and could not be read is ``unreadable``, which is not
    ``absent``.
    """
    record = config if isinstance(config, dict) else {}
    state = record.get("state")
    if state == "absent":
        return {"state": OVERRIDE_ABSENT, "pages": None}
    if state != "present":
        return {"state": OVERRIDE_UNREADABLE, "pages": None}
    lines = [str(line) for line in record.get("lines") or []]
    if len(lines) == 1:
        match = re.fullmatch(r"options\s+ttm\s+(.+)", lines[0])
        if match is not None:
            pairs: Dict[str, str] = {}
            for token in match.group(1).split():
                name, _, value = token.partition("=")
                pairs[name] = value
            pages = _positive_int(pairs.get("pages_limit"))
            if (
                set(pairs) == {"pages_limit", "page_pool_size"}
                and pages is not None
                and pairs["page_pool_size"] == pairs["pages_limit"]
            ):
                return {"state": OVERRIDE_VAELOR, "pages": pages}
    return {"state": OVERRIDE_UNRECOGNISED, "pages": None}


def render_config(pages: int) -> str:
    """The whole text of Vaelor's file for ``pages``: a fixed header, one fixed line."""
    if type(pages) is not int or pages <= 0:
        raise ValueError("The GPU memory pool size must be a positive whole number.")
    return CONFIG_HEADER + CONFIG_TEMPLATE.format(pages=pages)


def _other_sources(facts: Dict[str, Any]) -> List[str]:
    """Anything besides Vaelor's file that also sets the GPU's memory, by name."""
    sources = [
        str(item.get("path", "")) for item in facts.get("others") or []
        if isinstance(item, dict) and item.get("path")
    ]
    for name in facts.get("cmdline") or []:
        if name in _CMDLINE_NAMES:
            sources.append(_CMDLINE_NAMES[name])
    return sources


def _unread_sources(facts: Dict[str, Any]) -> List[str]:
    """What could not be read, so whether it sets the limit is not known."""
    unread = [str(name) for name in facts.get("unreadable") or []]
    if facts.get("truncated"):
        unread.append("some of this machine's many modprobe files")
    if facts.get("cmdline") is None:
        unread.append("the kernel command line")
    return unread


def _unsupported(reason: str) -> Dict[str, Any]:
    return {
        "supported": False, "reason": reason, "can_change": False,
        "blocked_reason": reason,
    }


def _gpu_refusal(gpu: Any) -> str:
    """Why this machine's GPU has no pool to size, or ``""`` when it has one."""
    facts = gpu if isinstance(gpu, dict) else {}
    if not facts.get("present"):
        return (
            "No GPU that Vaelor can serve models on was found on this machine, "
            "so there is no GPU memory pool to size."
        )
    unified = facts.get("unified_memory")
    if unified is False:
        return (
            "This machine's GPU has its own dedicated memory. The pool is a "
            "setting only for a GPU that shares the system's memory."
        )
    if unified is not True:
        return (
            "Vaelor could not tell whether this machine's GPU shares the "
            "system's memory. Recheck the machine, then look again."
        )
    return ""


def _gib_text(bytes_value: Optional[int]) -> str:
    return "{:.1f}".format((bytes_value or 0) / GIB)


def pool_status(facts: Any, gpu: Any) -> Dict[str, Any]:
    """Everything a surface may say about one machine's GPU memory pool.

    ``facts`` is the gathered record; ``gpu`` is the machine's GPU record in
    the probe's shape (``present``, ``unified_memory``). Capability comes from
    discovery: a machine gets the setting only when its kernel exposes the
    limit AND its GPU was found to share system memory. Every other machine
    gets ``supported: False`` with the reason in plain words.

    ``restart_pending`` compares what the files ask for with what the kernel is
    running. It is ``None`` - unknown, with the reason - when something other
    than Vaelor's file also sets the limit, or when a file that might could not
    be read: Vaelor cannot say which setting the next start will honour, and
    will not change one it cannot see the whole of.
    """
    refusal = _gpu_refusal(gpu)
    if refusal:
        return _unsupported(refusal)
    if not isinstance(facts, dict) or not isinstance(facts.get("config"), dict):
        return _unsupported(
            "This machine did not report its memory settings. Recheck it, then "
            "look again."
        )
    page_size = facts.get("page_size")
    current_pages = _positive_int(facts.get("pages_limit"))
    if current_pages is None or type(page_size) is not int or page_size <= 0:
        return _unsupported(
            "This machine's kernel does not report a GPU memory pool limit, so "
            "there is nothing here to change."
        )
    ram_kb = _positive_int(facts.get("mem_total_kb"))
    if ram_kb is None:
        return _unsupported(
            "This machine's total memory could not be read, so a safe range "
            "for the GPU memory pool cannot be worked out."
        )
    ram_bytes = ram_kb * 1024
    default_bytes = (ram_bytes // page_size // 2) * page_size
    ceiling = achievable_gtt_ceiling(ram_bytes) or 0
    min_gib = -(-default_bytes // GIB)
    max_gib = ceiling // GIB
    current_bytes = current_pages * page_size
    override = parse_config(facts.get("config"))
    override_bytes = (
        override["pages"] * page_size if override["pages"] is not None else None
    )
    others = _other_sources(facts)
    unread = _unread_sources(facts)
    status: Dict[str, Any] = {
        "supported": True,
        "reason": "",
        "page_size": page_size,
        "ram_bytes": ram_bytes,
        "current_bytes": current_bytes,
        "default_bytes": default_bytes,
        "system_left_bytes": max(0, ram_bytes - current_bytes),
        "override": {"state": override["state"], "bytes": override_bytes},
        "other_sources": others,
        "min_gib": min_gib,
        "max_gib": max_gib,
        "can_change": True,
        "blocked_reason": "",
    }
    at_default = abs(current_bytes - default_bytes) <= default_bytes * DEFAULT_TOLERANCE
    if others:
        status["restart_pending"] = None
        status["restart_reason"] = (
            "Something other than Vaelor also sets this limit ({}), so Vaelor "
            "cannot say which one the next restart will use."
        ).format(", ".join(others))
        status["can_change"] = False
        status["blocked_reason"] = (
            "{} also sets this limit. Remove that setting on the machine first; "
            "Vaelor will not overwrite or outvote it."
        ).format(", ".join(others))
    elif unread or override["state"] == OVERRIDE_UNREADABLE:
        named = ", ".join(
            unread + ([CONFIG_PATH] if override["state"] == OVERRIDE_UNREADABLE else [])
        )
        status["restart_pending"] = None
        status["restart_reason"] = (
            "{} could not be read, so Vaelor cannot say what the next restart "
            "will use.".format(named)
        )
        status["can_change"] = False
        status["blocked_reason"] = (
            "{} could not be read on this machine, so Vaelor cannot tell "
            "whether it also sets this limit and will not change the pool "
            "until it can.".format(named)
        )
    elif override["state"] == OVERRIDE_VAELOR:
        pending = override_bytes != current_bytes
        status["restart_pending"] = pending
        status["restart_reason"] = (
            "The pool is set to {} GiB and takes that size when this machine "
            "restarts.".format(_gib_text(override_bytes)) if pending else ""
        )
    elif override["state"] == OVERRIDE_ABSENT:
        status["restart_pending"] = not at_default
        status["restart_reason"] = (
            "" if at_default else
            "The pool goes back to the kernel's own size, about {} GiB, when "
            "this machine restarts.".format(_gib_text(default_bytes))
        )
    else:
        status["restart_pending"] = None
        status["restart_reason"] = (
            "The settings file on this machine was not written in Vaelor's "
            "form, so Vaelor cannot say what the next restart will use. "
            "Choosing a size here replaces that file."
        )
    # Nothing above the kernel's own size fits under the ceiling: a range that
    # holds only the default (or nothing) is not a setting worth offering.
    if max_gib * GIB <= default_bytes and status["can_change"]:
        status["can_change"] = False
        status["blocked_reason"] = (
            "This machine has too little memory to raise the pool above the "
            "kernel's own size and still leave enough for the system."
        )
    return status


#: What a machine's card is given. The status holds more than a card draws
#: (the page size, the list of other sources by path); the sentences already
#: say what the reader needs from those, so they stay off the wire.
_CARD_KEYS = (
    "supported", "reason", "can_change", "blocked_reason", "ram_bytes",
    "current_bytes", "default_bytes", "system_left_bytes", "override",
    "min_gib", "max_gib", "restart_pending", "restart_reason",
)


def card_view(status: Any) -> Dict[str, Any]:
    """The part of a status a settings card needs, and nothing else."""
    record = status if isinstance(status, dict) else {}
    return {key: record[key] for key in _CARD_KEYS if key in record}


def file_now_holds(size_gib: Optional[int]) -> str:
    """What Vaelor's file holds after a change that could not be undone.

    The words a refusal puts after "the settings file on <machine>": the size
    it now asks for, or that it has been removed (``None``). One home, because
    the controller's program and the worker job both say it.
    """
    if size_gib is None:
        return "has been removed"
    return "now asks for {} GiB".format(size_gib)


def unknown_after_change(name: str) -> Dict[str, Any]:
    """The stored status for a machine whose setting could not be read back.

    Written when a change was attempted and the machine then stopped
    answering: what it holds is not known, and the card must say so rather
    than keep showing the reading from before the attempt.
    """
    return _unsupported(
        "A change to the GPU memory pool was attempted on {} and its setting "
        "could not be read afterwards. Recheck it to see what it holds "
        "now.".format(name)
    )


def pool_left_note(inventory: Any) -> str:
    """What removing a worker leaves behind of this setting, or ``""``.

    Removing a worker deletes the only sign-in Vaelor has for it, so a pool
    file Vaelor wrote there stays, with nothing left to manage it (review S9).
    The removal plan and the finished removal both say so, and how to take it
    off - before, from the console; after, by hand.
    """
    status = (inventory or {}).get("gpu_memory_pool") if isinstance(inventory, dict) else None
    state = ((status or {}).get("override") or {}).get("state") if isinstance(status, dict) else None
    if state not in (OVERRIDE_VAELOR, OVERRIDE_UNRECOGNISED):
        return ""
    return (
        " Its GPU memory pool setting ({}) also stays on the machine. To take "
        "it off first, use Remove setting under {} before removing the "
        "machine; afterwards, delete that file on the machine and run 'sudo "
        "update-initramfs -u' there."
    ).format(CONFIG_PATH, SETTING_PLACE)


def _refuse_unsupported(status: Dict[str, Any]) -> None:
    if not status.get("supported"):
        raise ValueError(
            str(status.get("reason") or "This machine has no GPU memory pool setting.")
        )


def require_size_gib(value: Any, status: Dict[str, Any]) -> int:
    """``value`` as the number of pages to write, or a plain refusal.

    The one bounds rule: the screen's range, the job's check and the root
    side's check are all this call over a status read where the check runs.
    A whole number only - ``True``, ``"44"`` and ``44.0`` are refused, because
    the privileged side must never have to decide what a loose value meant.
    """
    _refuse_unsupported(status)
    if not status.get("can_change"):
        raise ValueError(str(status.get("blocked_reason")))
    if type(value) is not int:
        raise ValueError(WHOLE_NUMBER_REQUIRED)
    if not status["min_gib"] <= value <= status["max_gib"]:
        raise ValueError(
            "Choose a GPU memory pool between {} and {} GiB on this machine. "
            "Below that is smaller than the kernel's own size; above it leaves "
            "too little memory for the system.".format(
                status["min_gib"], status["max_gib"]
            )
        )
    return value * GIB // status["page_size"]


def require_revert(status: Dict[str, Any]) -> None:
    """Refuse a revert on a machine with nothing of Vaelor's to remove.

    A file that could not be READ is refused too: removing a file nobody
    could read is not putting anything back.
    """
    _refuse_unsupported(status)
    state = status["override"]["state"]
    if state == OVERRIDE_ABSENT:
        raise ValueError(
            "This machine has no Vaelor GPU memory pool setting to remove; it "
            "is already on the kernel's own size."
        )
    if state == OVERRIDE_UNREADABLE:
        raise ValueError(
            "{} could not be read on this machine, so Vaelor will not remove "
            "it. Check the file there first.".format(CONFIG_PATH)
        )


# --- the fit: would a bigger pool make this model fit ---------------------

def pool_way_out(machines: List[Dict[str, Any]]) -> str:
    """The ONE sentence that tells a reader a larger pool would make a model fit.

    ``machines`` is ``[{name, size_gib}]``. Public so that whichever part of
    the product says this - the fit's own way-out list or the route's hint -
    says it in these words, naming the setting and its place from
    :data:`SETTING_NAME` and :data:`SETTING_PLACE`.
    """
    named = ", ".join(
        "{} to {} GiB".format(item.get("name") or item.get("node_id"), item["size_gib"])
        for item in machines
    )
    return (
        "This model would fit with a larger {}: {}. Change it in {}. Each "
        "machine has to restart before the new size counts."
    ).format(SETTING_NAME, named, SETTING_PLACE)


def _raised_ledger(ledger: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """The ledger with each shared-memory GPU's pool at its largest allowed size.

    Returns ``(ledger, raised)`` where ``raised`` lists the machines whose pool
    could grow and to how many GiB. Only a node the ledger calls ``unified``
    and whose system memory was read is touched; nothing is invented for a
    node that reported neither.
    """
    raised_ledger = copy.deepcopy(ledger)
    raised: List[Dict[str, Any]] = []
    for node in raised_ledger.get("nodes", []) or []:
        gpu = (node.get("capacity") or {}).get("gpu") or {}
        if not gpu.get("present") or gpu.get("memory_model") != "unified":
            continue
        ceiling = achievable_gtt_ceiling(gpu.get("system_ram_bytes"))
        if ceiling is None:
            continue
        target = (ceiling // GIB) * GIB
        current = int(gpu.get("gtt_total_bytes", 0) or 0)
        if target <= current:
            continue
        gain = target - current
        gpu["gtt_total_bytes"] = target
        gpu["addressable_bytes"] = int(gpu.get("addressable_bytes", 0) or 0) + gain
        free = node.setdefault("free", {})
        free["gpu_memory_bytes"] = int(free.get("gpu_memory_bytes", 0) or 0) + gain
        raised.append({
            "node_id": str(node.get("node_id", "")),
            "name": str(node.get("name", "")),
            "size_gib": target // GIB,
        })
    return raised_ledger, raised


def pool_fit_hint(
    decision: Dict[str, Any], ledger: Dict[str, Any],
    rerun: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """For a "will not fit": whether the largest allowed pools would change it.

    ``rerun`` is the SAME fit call over a different ledger, so the hint cannot
    promise a fit the engine would refuse. ``None`` when the model already
    fits, when no machine's pool can grow, or when it still would not fit -
    the hint exists only when it is true.
    """
    if decision.get("verdict") != VERDICT_WONT_FIT:
        return None
    raised_ledger, raised = _raised_ledger(ledger)
    if not raised:
        return None
    if rerun(raised_ledger).get("verdict") == VERDICT_WONT_FIT:
        return None
    return {"would_fit": True, "machines": raised, "sentence": pool_way_out(raised)}
