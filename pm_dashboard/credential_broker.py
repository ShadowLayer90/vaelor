"""Deprecated alias for :mod:`vaelor.credential_broker`."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("vaelor.credential_broker")
