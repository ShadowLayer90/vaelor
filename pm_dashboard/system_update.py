"""Deprecated alias for :mod:`vaelor.system_update`."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("vaelor.system_update")
