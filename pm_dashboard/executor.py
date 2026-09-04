"""Deprecated alias for :mod:`vaelor.executor`."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("vaelor.executor")
