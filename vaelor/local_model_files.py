"""Filesystem validation and host budgeting for managed local GGUF models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from .copilot_setup import hardware_inventory


def model_file(value: str, models_root: Path) -> Path:
    model = Path(value).resolve()
    if (
        models_root not in model.parents
        or not model.is_file()
        or model.suffix.lower() != ".gguf"
    ):
        raise ValueError("Choose a downloaded managed GGUF model.")
    return model


def model_hardware_budget(
    models_root: Path, workloads_root: Path
) -> Dict[str, Any]:
    """Count managed-model RAM as reclaimable during a model switch."""
    hardware = hardware_inventory(str(models_root))
    compose_file = workloads_root / "model-assistant" / "compose.yaml"
    reclaimable = 0
    try:
        match = re.search(
            r"^\s*memory:\s*(\d+)M\s*$",
            compose_file.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        reclaimable = int(match.group(1)) * 1024 ** 2 if match else 0
    except (OSError, ValueError):
        pass
    total = int(hardware.get("memory_total_bytes", 0) or 0)
    available = int(hardware.get("memory_available_bytes", 0) or 0)
    hardware["memory_available_bytes"] = (
        min(total, available + reclaimable) if total else available
    )
    hardware["managed_model_reclaimable_bytes"] = reclaimable
    return hardware


def remove_model(payload: Dict[str, Any], models_root: Path, workloads_root: Path):
    model = model_file(str(payload.get("path", "")), models_root)
    if payload.get("confirm") != "remove-model":
        raise ValueError("Confirm removal before deleting this model.")
    compose_file = workloads_root / "model-assistant" / "compose.yaml"
    try:
        active_config = compose_file.read_text(encoding="utf-8")
    except OSError:
        active_config = ""
    if str(model.parent) in active_config and model.name in active_config:
        raise ValueError(
            "This model is active. Switch the Assistant to another model before removing it."
        )
    size = model.stat().st_size
    model.unlink()
    try:
        model.parent.rmdir()
    except OSError:
        pass
    return {"path": str(model), "reclaimed_bytes": size}
