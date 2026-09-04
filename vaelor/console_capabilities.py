"""The remote-console ladder: advertised ≠ reachable ≠ provisioned.

Task #54. ``vaelor/kvm.py`` answers with three states — ``ready``,
``not_configured``, ``unavailable`` — computed from Raspberry Pi hardware. That
is truthful for the Pi's capture path and stays. It has no rung for *"the
hardware is there and talking, but nothing answers"*, and no rung for
*"advertised in firmware and never provisioned"*, so on an HP Z2 Mini G1a those
two collapse into the same word as *"this machine has no such hardware"*.

Measured on the Z2 on 2026-08-07 from a second machine on the same segment
(VD-009a):

* The HP Remote System Controller is **not fitted** — all five detection
  attributes read ``No``, no SMBIOS type 42, no type 38, and HP's QuickSpecs
  say the Integrated variant is incompatible with this platform. It is an
  external appliance to buy, not an empty socket to populate.
* ``AMD DASH = Enable`` in firmware, and TCP 623/664 are **silently swallowed**
  by the NIC's management engine while every other port on the machine answers
  RST. Advertised, present and armed, unreachable, unprovisioned — four
  different facts in one capability.

Two rules make the ladder hold up, and both are enforced here rather than left
to callers:

1. **A rung is claimed only by the probe that can prove it.** A firmware
   attribute proves ``advertised`` and nothing above it. Only a transport probe
   may claim ``reachable``; only an authenticated call may claim
   ``provisioned``.
2. **Every state carries who can move it.** That is the ``actor`` field, and it
   is the reason ``not_configured`` misleads: it renders as "Needs setup", which
   promises a path Vaelor can walk, when the next step is a person pressing F3
   at POST on a screen Vaelor does not have.

**The constraint that is easy to get wrong.** The identical probe run *on* the
Z2 returns the opposite verdict — RST, i.e. "no DASH hardware" — because Linux
routes the host's own address over ``lo``::

    $ ip route get 192.168.4.58
    local 192.168.4.58 dev lo src 192.168.4.58

The packet never reaches the network controller, so a self-probe does not
merely fail to detect the management engine: it produces a confident ``absent``
for a machine that demonstrably has one. Vantage point is part of the
measurement, which is why :func:`out_of_band_reachability` takes it as a
required argument and answers ``unknown`` — never ``absent`` — when the probe
ran on the machine it was probing. This mirrors what
:mod:`vaelor.model_reachability` already does with its ``unknown`` flag, and
reuses its ``checked_at`` vocabulary, because someone at the machine can
provision the engine between two page loads.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


#: The five rungs, strictly ordered. A capability reports the highest rung it
#: has positive evidence for, and never a rung above the evidence.
STATES = ("absent", "advertised", "present", "reachable", "provisioned")

#: Who can move a capability up. This is the field the three-state model
#: lacked, and the one that stops the interface implying Vaelor can do
#: something only a person standing at the machine can.
ACTOR_LABELS: Dict[str, str] = {
    "vaelor": "Vaelor can do this",
    "operator-remote": "Someone can do this from here",
    "operator-onsite": "Requires someone at the machine",
    "hardware": "Requires hardware",
    "none": "Nothing will change this on this platform",
}

#: Where DASH / AMD PRO Manageability speaks, per HP's own whitepaper: *"By
#: default, the Realtek Ethernet controller will use these ports."*
OUT_OF_BAND_PORTS = (623, 664)

#: Firmware attributes that advertise out-of-band management, and the values
#: that count as an advertisement. Reading these is a local sysfs read, so it
#: is the one rung the appliance can establish about itself.
OUT_OF_BAND_FIRMWARE_ATTRIBUTES = ("amd dash", "remote device management")

#: The five attributes that report whether the HP Remote System Controller is
#: physically fitted. All five read ``No`` on the Z2.
REMOTE_CONTROLLER_ATTRIBUTES = (
    "hp remote system controller mainboard connection detected",
    "hp remote system controller usb connection detected",
    "hp remote system controller vdisplay connection detected",
    "hp remote system controller microsd card detected",
    "hp remote system controller microsd card mounted",
)

_ENABLED = {"enable", "enabled", "on", "yes", "1", "true"}


def rung(state: str) -> int:
    """The ladder position of a state name; an unknown name is the bottom."""
    try:
        return STATES.index(str(state))
    except ValueError:
        return 0


def highest(*states: str) -> str:
    """The highest rung among several pieces of evidence for one capability."""
    return max(states, key=rung) if states else STATES[0]


def capability(
    identifier: str,
    title: str,
    state: str,
    actor: str,
    detail: str,
    **extra: Any,
) -> Dict[str, Any]:
    """One ladder row, with ``actionable`` derived rather than passed in.

    Only rung 4 may render an actionable console, and only when Vaelor is the
    party that can act. Deriving it here is the point: a row cannot report
    ``unreachable`` and still carry a control, because no caller gets to decide
    that separately from the state it just reported.
    """
    name = str(actor if actor in ACTOR_LABELS else "none")
    value = str(state if state in STATES else "absent")
    row = dict(extra)
    # Written last, so nothing a caller passes can raise a row past the state
    # it just reported.
    row.update({
        "id": str(identifier),
        "title": str(title),
        "state": value,
        "rung": rung(value),
        "actor": name,
        "actor_label": ACTOR_LABELS[name],
        "detail": str(detail),
        "actionable": value == "provisioned" and name == "vaelor",
    })
    return row


def probe_vantage(target: str, local_addresses: Optional[Iterable[str]] = None) -> str:
    """``"self"`` when a probe to ``target`` would never leave this machine.

    A connection to the host's own configured address is short-circuited onto
    ``lo`` before the interface is consulted — ``SO_BINDTODEVICE`` does not
    change this — so the host's own stack answers, and it answers RST. Naming
    that case is the whole difference between a measurement and its opposite.
    """
    address = str(target or "").strip().lower()
    if not address or address in {"127.0.0.1", "::1", "localhost", "loopback"}:
        return "self"
    known = {str(item or "").strip().lower() for item in (local_addresses or ())}
    return "self" if address in known else "peer"


def out_of_band_reachability(
    observations: Optional[Mapping[int, str]],
    control: str,
    *,
    vantage: str,
    checked_at: Optional[float] = None,
) -> Dict[str, Any]:
    """What a port sweep of the management ports actually established.

    ``observations`` maps port to one of ``rst``, ``timeout``, ``open`` or
    ``authenticated``. ``control`` is the same observation for a port in the
    same pass that is known to have nothing listening — it is **mandatory**,
    because without it "timeout" and "closed" are indistinguishable and a host
    firewall turns every result into a false ``absent``.

    The middle outcome is the one a port scanner throws away: management ports
    that time out while the control port answers RST means an engine is
    intercepting them in hardware, ahead of the host. That is ``present``, and
    it is the Z2's actual condition.

    ``unknown`` is a real answer and is returned whenever the measurement could
    not be made — including, unconditionally, when ``vantage`` is ``"self"``.
    """
    moment = time.time() if checked_at is None else float(checked_at)
    result: Dict[str, Any] = {
        "state": "", "unknown": True, "vantage": str(vantage),
        "detail": "", "checked_at": moment,
        "observations": {int(port): str(value) for port, value in (observations or {}).items()},
        "control": str(control or ""),
    }
    if str(vantage) != "peer":
        # Not a gap in coverage — the wrong answer. Off-box this machine reads
        # TIMEOUT/TIMEOUT against an RST control; on itself the identical probe
        # reads RST/RST and concludes there is no management engine at all.
        result["detail"] = (
            "This check ran on the machine it was checking, so the packets went "
            "over the loopback interface and never passed the network "
            "controller. A host cannot see its own management engine; another "
            "machine on the same network has to look."
        )
        return result
    seen = [str(value).lower() for value in (observations or {}).values()]
    if not seen:
        result["detail"] = "No management port was probed."
        return result
    if str(control or "").lower() != "rst":
        result["detail"] = (
            "The control port did not answer either, so a timeout on the "
            "management ports proves nothing about them."
        )
        return result
    result["unknown"] = False
    if "authenticated" in seen:
        result.update({
            "state": "provisioned",
            "detail": "An authenticated management call was answered.",
        })
    elif "open" in seen:
        result.update({
            "state": "reachable",
            "detail": (
                "A management port completed a connection, so the engine is "
                "serving and is waiting for credentials."
            ),
        })
    elif "timeout" in seen:
        result.update({
            "state": "present",
            "detail": (
                "The management ports are silently swallowed while a port with "
                "nothing behind it answers RST. Something is intercepting them "
                "in front of the host: the engine is powered and on the wire, "
                "and it is not serving."
            ),
        })
    elif set(seen) == {"rst"}:
        result.update({
            "state": "absent",
            "detail": (
                "The management ports answer RST exactly as an unused port "
                "does, so nothing is claiming them."
            ),
        })
    else:
        result.update({
            "unknown": True,
            "detail": "The probe returned an outcome that was not recognised.",
        })
    return result


def unchecked_reachability(*, checked_at: Optional[float] = None) -> Dict[str, Any]:
    """The honest answer where Vaelor has no off-box vantage — which is today.

    Option 3 of the three ways to get a vantage: do not probe, report
    ``advertised``, and say the rest is unknown. Least effort, entirely
    truthful, and strictly better than a confident ``unavailable``. Option 1 —
    a peer in the fleet opens the connection and reports the signature — is the
    upgrade, and substitutes a real :func:`out_of_band_reachability` here.
    """
    return {
        "state": "", "unknown": True, "vantage": "none",
        "observations": {}, "control": "",
        "checked_at": time.time() if checked_at is None else float(checked_at),
        "detail": (
            "no machine other than this one has checked, and this machine "
            "cannot check itself — a connection to its own address never "
            "reaches the network controller."
        ),
    }


def _attribute_values(root: Path) -> Dict[str, str]:
    """Current values of a ``firmware-attributes`` class directory, lowercased."""
    values: Dict[str, str] = {}
    try:
        entries = sorted(root.iterdir())[:512]
    except OSError:
        return values
    for entry in entries:
        try:
            values[entry.name.strip().lower()] = (
                (entry / "current_value")
                .read_text(encoding="utf-8", errors="replace")
                .strip("\x00\n ")
                .lower()
            )
        except OSError:
            continue
    return values


def firmware_out_of_band(
    attributes_root: str = "/sys/class/firmware-attributes/hp-bioscfg/attributes",
) -> Dict[str, Any]:
    """What this machine's own firmware claims, which is rung 1 and no higher.

    ``AMD DASH = Enable`` says the feature is armed. It says nothing about
    whether anything answers, and nothing at all about whether an account
    exists — HP removed the default credentials on 2020-and-later platforms
    under California SB 327, so provisioning is a person at POST. Reading this
    is deliberately separated from claiming anything above it.
    """
    values = _attribute_values(Path(attributes_root))
    advertised = any(
        values.get(name, "") in _ENABLED
        for name in OUT_OF_BAND_FIRMWARE_ATTRIBUTES
    )
    controller = [
        name for name in REMOTE_CONTROLLER_ATTRIBUTES
        if values.get(name, "") in _ENABLED
    ]
    return {
        "advertised": advertised,
        "attributes_read": len(values),
        "remote_controller_fitted": bool(controller),
        "remote_controller_evidence": controller,
    }


def video_capability(video: Mapping[str, Any], machine_class: str) -> Dict[str, Any]:
    """Seeing the screen. Never inherits a rung from out-of-band power."""
    if video.get("stream_ready"):
        return capability(
            "console.video", "Remote screen", "provisioned", "vaelor",
            "A protected capture stream is running and can be opened from here.",
        )
    if video.get("capture_detected"):
        return capability(
            "console.video", "Remote screen", "present", "operator-onsite",
            "A capture device is attached, but its protected stream has not "
            "been commissioned on the machine.",
        )
    if str(machine_class) == "workstation":
        return capability(
            "console.video", "Remote screen", "absent", "hardware",
            "This workstation has no video capture path. HP's Remote System "
            "Controller (part 7K6E4AA), an external appliance, is the supported "
            "option for this platform; it is not fitted.",
        )
    return capability(
        "console.video", "Remote screen", "absent", "hardware",
        "No capture adapter was detected. A supported USB HDMI capture adapter "
        "is what this rung needs.",
    )


def input_capability(hid: Mapping[str, Any], machine_class: str) -> Dict[str, Any]:
    """Keyboard and mouse, which emulate a USB *device* and so need a UDC.

    On an x86 workstation there is no rung above ``absent`` and no actor: the
    USB controllers are host-only and no peripheral turns them into a device
    port. That is why the actor here is ``none`` rather than ``hardware`` — a
    purchase would not help, and the copy this replaces implied one would.
    """
    if hid.get("input_ready"):
        return capability(
            "console.input", "Keyboard and mouse", "provisioned", "vaelor",
            "The isolated USB input bridge is commissioned.",
        )
    if hid.get("configured"):
        return capability(
            "console.input", "Keyboard and mouse", "present", "operator-onsite",
            "A USB HID gadget exists; its guarded input bridge has not been "
            "commissioned on the machine.",
        )
    if hid.get("controller_available"):
        return capability(
            "console.input", "Keyboard and mouse", "present", "operator-onsite",
            "A USB device controller is available, so this machine can emulate "
            "a keyboard once the HID gadget is commissioned on it.",
        )
    if str(machine_class) == "workstation":
        return capability(
            "console.input", "Keyboard and mouse", "absent", "none",
            "This machine has no USB device controller. A workstation's USB "
            "ports are host-only, and no capture or adapter hardware adds one, "
            "so keyboard emulation cannot be reached on this platform.",
        )
    return capability(
        "console.input", "Keyboard and mouse", "absent", "hardware",
        "No USB device controller was detected, so nothing here can emulate a "
        "keyboard.",
    )


def out_of_band_power_capability(
    firmware: Optional[Mapping[str, Any]] = None,
    reachability: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Out-of-band power and inventory — the DASH row, and never the screen.

    This is the concrete answer to *"those two must not be promised together"*.
    A ``power.oob`` that one day reaches ``provisioned`` must not light up a
    console: the screen is ``console.video``, it is a separate row, and nothing
    about DASH moves it.
    """
    facts = dict(firmware or {})
    probe = dict(reachability or {})
    advertised = bool(facts.get("advertised"))
    measured = "" if probe.get("unknown", True) else str(probe.get("state") or "")
    state = highest("advertised" if advertised else "absent", measured or "absent")
    unknown = bool(probe.get("unknown", True))
    if state == "absent":
        return capability(
            "power.oob", "Out-of-band power", "absent", "none",
            "No out-of-band management firmware was found on this machine.",
            reachable=False if not unknown else None,
            checked_at=probe.get("checked_at", 0.0),
            vantage=probe.get("vantage", ""),
        )
    if state == "provisioned":
        return capability(
            "power.oob", "Out-of-band power", "provisioned", "vaelor",
            "The management engine accepted an authenticated call, so power and "
            "inventory are available independently of the host.",
            reachable=True, checked_at=probe.get("checked_at", 0.0),
            vantage=probe.get("vantage", ""),
        )
    if state == "reachable":
        return capability(
            "power.oob", "Out-of-band power", "reachable", "operator-remote",
            "The management engine is answering and is waiting for credentials. "
            "Vaelor has none for it.",
            reachable=True, checked_at=probe.get("checked_at", 0.0),
            vantage=probe.get("vantage", ""),
        )
    if state == "present":
        return capability(
            "power.oob", "Out-of-band power", "present", "operator-onsite",
            "This machine's network controller has management firmware and is "
            "responding on the network, but it has no account and refuses "
            "connections. Provisioning happens at the machine during startup "
            "and cannot be done from here.",
            reachable=False, checked_at=probe.get("checked_at", 0.0),
            vantage=probe.get("vantage", ""),
        )
    return capability(
        "power.oob", "Out-of-band power", "advertised", "operator-onsite",
        "Firmware advertises out-of-band management. Nothing has contacted it: "
        "{} Provisioning happens at the machine during startup.".format(
            probe.get("detail")
            or "no reachability check has been made from another machine."
        ),
        reachable=None, checked_at=probe.get("checked_at", 0.0),
        vantage=probe.get("vantage", ""),
    )


def console_ladder(
    snapshot: Mapping[str, Any],
    machine_class: str,
    *,
    firmware: Optional[Mapping[str, Any]] = None,
    reachability: Optional[Mapping[str, Any]] = None,
) -> Sequence[Dict[str, Any]]:
    """``remote_console`` split into three siblings, discovered independently.

    They never inherit each other's rung. Seeing the screen, driving the
    keyboard, and powering the machine out of band fail for different reasons,
    are fixed by different actions, and belong on different rows.
    """
    video = snapshot.get("video") if isinstance(snapshot.get("video"), Mapping) else {}
    hid = snapshot.get("hid") if isinstance(snapshot.get("hid"), Mapping) else {}
    return [
        video_capability(video, machine_class),
        input_capability(hid, machine_class),
        out_of_band_power_capability(firmware, reachability),
    ]
