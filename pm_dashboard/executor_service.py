"""Deprecated alias for :mod:`vaelor.executor_service`."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("vaelor.executor_service")
