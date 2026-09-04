"""Deprecated alias for :mod:`vaelor.runtime_paths`."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("vaelor.runtime_paths")
