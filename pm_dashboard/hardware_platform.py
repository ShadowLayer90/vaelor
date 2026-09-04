"""Deprecated alias for :mod:`vaelor.hardware_platform`."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("vaelor.hardware_platform")
