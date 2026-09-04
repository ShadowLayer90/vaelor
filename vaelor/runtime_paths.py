"""Vaelor-owned runtime paths with one-window Pironman compatibility aliases."""

from __future__ import annotations

import os
from pathlib import Path


def env_value(primary: str, legacy: str, default: str) -> str:
    """Prefer the Vaelor setting, then its documented legacy alias."""
    return os.environ.get(primary, "").strip() or os.environ.get(
        legacy, ""
    ).strip() or default


STATE_ROOT = Path(env_value(
    "VAELOR_STATE_ROOT", "PM_STATE_ROOT", "/var/lib/vaelor"
))
DATA_ROOT = Path(env_value(
    "VAELOR_DATA_ROOT", "PM_DATA_ROOT", "/var/lib/vaelor"
))
APP_ROOT = Path(env_value(
    "VAELOR_APP_ROOT", "PM_APP_ROOT", "/opt/vaelor"
))
RUN_ROOT = Path(env_value(
    "VAELOR_RUN_ROOT", "PM_RUN_ROOT", "/run/vaelor"
))
LOG_ROOT = Path(env_value(
    "VAELOR_LOG_ROOT", "PM_LOG_ROOT", "/var/log/vaelor"
))


def state_path(relative: str) -> str:
    return str(STATE_ROOT / relative)


def data_path(relative: str) -> str:
    return str(DATA_ROOT / relative)


def app_path(relative: str) -> str:
    return str(APP_ROOT / relative)


def run_path(relative: str) -> str:
    return str(RUN_ROOT / relative)


def jobs_group_id(group_module):
    if group_module is None:
        return None
    for name in ("vaelor-jobs", "pironman-jobs"):
        try:
            return group_module.getgrnam(name).gr_gid
        except KeyError:
            continue
    return None
