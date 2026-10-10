"""Install what a console-upgraded release newly depends on (R19).

The console upgrade reinstalls the wheel with ``--no-deps``
(``appliance_upgrade.REINSTALL_FLAGS``) so it never re-resolves the whole
dependency graph on the box. The cost was that a dependency a release ADDS
never arrived: PyYAML once, and VD-212's ``segno``. ``install-vaelor.sh`` has no
such gap - it runs a resolving install before its own ``--no-deps`` reinstall -
so this is the console path's half of the same rule.

After the reinstall, the venv's own interpreter lists every runtime requirement
the INSTALLED release declares (its ``Requires-Dist``, markers evaluated, extras
excluded) that the venv does not satisfy: missing, or installed outside the
declared range. Only those are handed to pip, with
``--upgrade-strategy only-if-needed``, so nothing already satisfied is upgraded
and a venv with nothing unmet never reaches pip or the network.

**A requirement that cannot be installed fails the upgrade** (it rolls back to
the previous release). The new code imports what it declares, so carrying on
would start services that crash-loop on an import, or serve a feature that
cannot work - and nothing here knows which feature a module backs, so
"degraded" could not be said truthfully. The message names the requirement and
pip's own reason (offline, most likely).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

#: Run by the venv's python with the distribution name as ``argv[1]``; prints one
#: unmet requirement per line. ``packaging`` is a declared dependency of Vaelor,
#: so it is present wherever this runs.
CHECK_PROGRAM = """
import sys
from importlib import metadata
from packaging.requirements import Requirement
for text in metadata.requires(sys.argv[1]) or []:
    requirement = Requirement(text)
    if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
        continue
    extras = "[" + ",".join(sorted(requirement.extras)) + "]" if requirement.extras else ""
    wanted = requirement.name + extras + str(requirement.specifier)
    try:
        installed = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        print(wanted)
        continue
    if not requirement.specifier.contains(installed, prereleases=True):
        print(wanted)
"""


def check_argv(python: str, distribution: str = "vaelor-control-plane") -> List[str]:
    """The check, run isolated: root runs it, so cwd and ``PYTHON*`` stay out."""
    return [python, "-I", "-c", CHECK_PROGRAM, distribution]


class DependencyFailure(RuntimeError):
    """A declared requirement is unmet and could not be installed."""


def unmet_requirements(run: Callable[[List[str]], str], python: str) -> List[str]:
    """The declared requirements the venv does not satisfy, via ``run(argv)``."""
    return [line.strip() for line in run(check_argv(python)).splitlines()
            if line.strip()]


def install_unmet(
    run: Callable[[List[str]], str],
    python: str,
    pip: str,
    *,
    normalize: Optional[Sequence[str]] = None,
) -> List[str]:
    """Install every unmet declared requirement; return what was installed.

    ``run`` executes an argv and raises on a non-zero exit (the broker's
    ``_run``). Raises :class:`DependencyFailure` when pip cannot install them
    or they are still unmet afterwards.

    ``normalize`` is the broker's own venv permission fix
    (``NORMALIZE_PERMS`` + the venv): this install runs under the broker's
    ``UMask=0007``, after the reinstall's fix already ran, so whatever it adds
    would otherwise be group-only and group-writable - unreadable by the
    services outside ``vaelor-jobs`` (R23, #178 / VD-102). It runs whenever
    pip ran, before the re-check and before any restart.
    """
    unmet = unmet_requirements(run, python)
    if not unmet:
        return []
    try:
        run([pip, "install", "--upgrade-strategy", "only-if-needed", *unmet])
    except Exception as error:  # noqa: BLE001 - re-raised with what was needed
        raise DependencyFailure(
            "This release needs {} and it could not be installed, so the "
            "upgrade was not applied: {}".format(", ".join(unmet), error)
        ) from error
    finally:
        # Also after a FAILED install: pip may already have replaced a shared
        # package (cryptography, say) before failing, and the rollback's
        # services must still be able to read it.
        if normalize:
            run(list(normalize))
    still = unmet_requirements(run, python)
    if still:
        raise DependencyFailure(
            "This release needs {}, still unmet after installing it, so the "
            "upgrade was not applied.".format(", ".join(still))
        )
    return unmet
