"""Deprecated alias for :mod:`vaelor.pm_dashboard`."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("vaelor.pm_dashboard")
