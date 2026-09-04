"""Public CLI for encrypted Vaelor state transfer."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

from .portable_state_core import (
    MAX_ARCHIVE_BYTES,
    PortableState,
    PortableStateError,
)

__all__ = [
    "MAX_ARCHIVE_BYTES",
    "PortableState",
    "PortableStateError",
    "main",
]


def _passphrase() -> str:
    value = os.environ.get("VAELOR_TRANSFER_PASSPHRASE", "")
    return value or getpass.getpass("Transfer passphrase: ")


def main() -> int:
    parser = argparse.ArgumentParser(prog="vaelor-state")
    subcommands = parser.add_subparsers(dest="action", required=True)
    subcommands.add_parser("scope")
    export = subcommands.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    inspect = subcommands.add_parser("inspect")
    inspect.add_argument("archive", type=Path)
    restore = subcommands.add_parser("import")
    restore.add_argument("archive", type=Path)
    restore.add_argument("--replace", action="store_true")
    restore.add_argument("--confirm", default="")
    args = parser.parse_args()
    state = PortableState()
    try:
        if args.action == "scope":
            result = state.scope()
        elif args.action == "export":
            result = state.export(args.output, _passphrase())
        elif args.action == "inspect":
            result = state.inspect(args.archive, _passphrase())
        else:
            if args.confirm != "IMPORT VAELOR STATE":
                raise PortableStateError(
                    "Type IMPORT VAELOR STATE exactly before importing."
                )
            result = state.import_archive(
                args.archive,
                _passphrase(),
                replace=args.replace,
            )
            result.pop("files", None)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, PortableStateError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
