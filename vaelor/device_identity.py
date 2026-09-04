"""Durable, dashboard-owned Pironman enclosure selection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .runtime_paths import env_value, state_path as vaelor_state_path
from .version import __version__

DEFAULT_STATE_PATH = vaelor_state_path("device-identity.json")


def appliance_identity(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The appliance's identity, with one answer to "which version".

    ``device_info`` is the enclosure hardware runtime's own record, and its
    ``version`` key is SunFounder's ``pironman5`` package version — ``1.3.18``
    on the appliance this was measured on. Every reader of that key was asking
    a different question. ``GET /api/v2/device`` already knew, and overwrote
    the value at the route; the ``system.identity`` tool did not, and told the
    Assistant the machine was "running Vaelor 1.3.18" while the machine brief
    in the same prompt said 2.1.0a48 (#185, `FOUNDATION_EVIDENCE.md` line 71).

    Correcting it at one of two readers is the repair that does not stay fixed,
    so both read this. ``version`` is Vaelor's, from one constant;
    ``enclosure_runtime_version`` is the vendor package's, under a name that
    says which runtime it is, because it is the only report this appliance has
    of the code driving the OLED, the fans and the RGB. An appliance with no
    enclosure has no such runtime and gets an empty string rather than an
    invented "unknown".
    """
    identity = dict(raw)
    identity["enclosure_runtime_version"] = str(raw.get("version") or "").strip()
    identity["version"] = __version__
    return identity


def state_path() -> Path:
    return Path(env_value(
        "VAELOR_DEVICE_IDENTITY_PATH", "PM_DEVICE_IDENTITY_PATH",
        DEFAULT_STATE_PATH,
    ))


def read_override() -> Optional[str]:
    path = state_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    value = payload.get("device_variant_override") if isinstance(payload, dict) else None
    return value if isinstance(value, str) and value else None


def write_override(model: str) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"device_variant_override": model}, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def selected_product(
    model: Optional[str],
    products: Mapping[str, Mapping[str, Any]],
    candidates: Optional[Sequence[str]] = None,
) -> Optional[dict[str, Any]]:
    """Render a manually chosen enclosure, and say how well it is evidenced.

    ``confident`` used to be an unconditional ``True``: whatever an operator
    picked, the appliance then asserted. On a Pi that turned "a Pironman 5 Mini
    can be told it is a Pro Max" into a settled fact, complete with an OLED, a
    third fan and a touchscreen the case does not have.

    ``candidates`` is what discovery left open (see
    :func:`vaelor.platforms.plausible_products`). The choice is only *confident*
    when discovery had already narrowed to this one enclosure and the operator
    agreed with it. Picking between two cases the bridge cannot tell apart is a
    disambiguation, and a disambiguation is a guess with good manners — it is
    reported as one, so the interface keeps offering the picker rather than
    settling on an answer nothing measured.

    Omitting ``candidates`` means the caller has no discovery evidence to hand,
    which is not the same as evidence for the choice: it is reported as not
    confident rather than defaulting to the claim.
    """
    if model not in products:
        return None
    allowed = None if candidates is None else [str(item) for item in candidates]
    product = dict(products[model])
    product.update(
        {
            "id": model,
            "detected_from": "manual",
            "confident": allowed == [model],
            "disambiguated_from": allowed or [],
        }
    )
    return product
