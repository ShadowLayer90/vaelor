"""Deprecated alias for :mod:`vaelor.security`."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("vaelor.security")
