"""Deprecated alias for :mod:`vaelor.api_v2`."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("vaelor.api_v2")
