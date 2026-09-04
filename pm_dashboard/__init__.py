"""Deprecated aliases for the former ``pm_dashboard`` package name.

Vaelor owns every implementation. This compatibility namespace exists for
one documented deprecation window and must never grow product logic of its own.
"""

from __future__ import annotations

from vaelor import __version__, main

__all__ = ["__version__", "main"]
