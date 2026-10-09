"""Whether a cluster link is the controller's shared LAN card, said where it is chosen (B8).

The cluster link (VD-162) can be any of this controller's own links, its LAN
card included. A split on the LAN card is not refused: it is fenced as a
SHARED network card (ACC-187, `gpu_ray_plane.render_ruleset`), which guards
only the split's own connections and leaves the card open to everything else,
so the LAN stays reachable. The confirmation that sets the link used to say
none of that, and an owner choosing the LAN card could not tell it from a
dedicated cable.

**One rule decides "shared", and it is the split's own.** `prepare_split`
(`gpu_ray_plane`) fences a machine as shared when its split address is the
one Vaelor reaches it on, or its split interface is the one it enrolled on.
:func:`rides_shared_card` is that rule, written once here so the console's
sentence and the fence cannot answer the question differently (LESSONS 6).

**The words are the backend's** (the browser holds none, VD-147 pass 3): the
host-settings surface carries each link's ``shared_note`` and the console
shows it in the confirmation as written.

Runs in the control plane only, over values it already holds: the discovered
links, the controller's advertise address and its enrolled interface. Nothing
is probed.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

#: The confirmation's sentence for the shared LAN card (B8). It says what the
#: choice means: how a split is fenced there, and that the LAN stays reachable.
SHARED_CARD_NOTE = (
    "{link} is the network card other machines reach this controller on "
    "({address}), so it is shared with your LAN. A split on it is fenced as a "
    "shared network card: Vaelor guards only the split's own connections, "
    "your LAN still reaches this controller through {link}, and the split "
    "shares the card's bandwidth with everything else on it."
)


def rides_shared_card(
    *, link_name: Any, link_address: Any, advertise_address: Any, enrolled_interface: Any,
) -> bool:
    """Whether a split on this link is fenced as a shared card (`prepare_split`'s rule)."""
    address = str(link_address or "")
    name = str(link_name or "")
    return bool(
        (address and address == str(advertise_address or ""))
        or (name and name == str(enrolled_interface or ""))
    )


def shared_card_note(
    link: Mapping[str, Any], *, advertise_address: Any, enrolled_interface: Any,
) -> str:
    """The confirmation's sentence for ``link``, or ``""`` for a link of its own."""
    if not rides_shared_card(
        link_name=link.get("name"), link_address=link.get("address"),
        advertise_address=advertise_address, enrolled_interface=enrolled_interface,
    ):
        return ""
    # The address other machines reach this controller on is the advertised
    # one; a card matched by its name may report a second address of its own.
    return SHARED_CARD_NOTE.format(
        link=link.get("name", ""), address=advertise_address or link.get("address") or "",
    )


def with_shared_notes(
    links: Iterable[Mapping[str, Any]], *, advertise_address: Any, enrolled_interface: Any,
) -> List[Dict[str, Any]]:
    """Each discovered link with its ``shared_note`` (``""`` for a link of its own)."""
    return [
        {**link, "shared_note": shared_card_note(
            link, advertise_address=advertise_address, enrolled_interface=enrolled_interface,
        )}
        for link in links
    ]
