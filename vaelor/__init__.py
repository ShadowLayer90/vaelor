"""Vaelor edge-compute control plane."""

from .version import __version__


def main():
    from .server import main as server_main

    return server_main()

__all__ = ["__version__", "main"]
