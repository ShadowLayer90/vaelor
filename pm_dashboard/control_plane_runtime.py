"""Deprecated alias for :mod:`vaelor.control_plane_runtime`."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("vaelor.control_plane_runtime")
