"""The executor's half of an appliance upgrade: check, fetch, verify, stage.

The privileged apply - the ``pip`` reinstall, the restart of the control plane,
and the readback that proves which code is actually running - is a broker job,
because the executor can neither write ``/opt/vaelor/venv`` nor restart the
control plane it is a peer of. This mixin owns everything that comes *before*
that handoff and can be proven offline:

* **check** - is a pinned release offered, and is this appliance eligible for
  it? Eligibility compares the release's ``min_from_version`` against the
  *current* running version, which at check time is honestly this process's own
  ``__version__`` because nothing has changed yet.
* **fetch** - copy the manifest's wheel into a staging directory.
* **verify-artifact** - recompute the SHA-256 and byte length and require both
  to equal the manifest. A mismatch deletes the file and fails the job with
  nothing installed (#130).
* **stage** - hand the verified wheel to the broker as a one-use plan, then ask
  the broker to apply it.

The job finishes as *accepted*: it does not, and must not, claim the version
changed. Whether the new code is actually running is the broker's readback to
answer, recorded in its result file and surfaced by ``GET /api/v2/upgrade`` -
never stamped here from this process's imported constant (#185/#171).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

# The pre-upgrade running version. Honest here and only here: at check time this
# process and the control plane are the same installed package, so this is the
# true "from". It is never used to describe the state *after* an apply - that is
# the broker's live readback (#185/#171).
from .version import __version__ as CURRENT_VERSION
from .appliance_upgrade import (
    CONFIRMATION,
    ApplianceUpgradeClient,
    UpgradeApplyPlans,
)
from .release_source import StubReleaseSource, upgrade_eligibility
from .runtime_paths import state_path


class ExecutorApplianceUpgradeMixin:
    """Implement check -> fetch -> verify-artifact -> stage for ``JobExecutor``."""

    def _release_source(self):
        source = getattr(self, "release_source", None)
        if source is None:
            source = StubReleaseSource()
            self.release_source = source
        return source

    def _upgrade_plans(self) -> UpgradeApplyPlans:
        plans = getattr(self, "upgrade_plans", None)
        if plans is None:
            plans = UpgradeApplyPlans()
            self.upgrade_plans = plans
        return plans

    def _upgrade_client(self) -> ApplianceUpgradeClient:
        client = getattr(self, "appliance_upgrade", None)
        if client is None:
            client = ApplianceUpgradeClient()
            self.appliance_upgrade = client
        return client

    def _run_appliance_upgrade(self, job: Dict[str, Any]) -> Dict[str, Any]:
        payload = job["payload"] if isinstance(job.get("payload"), dict) else {}
        if payload.get("confirmation") != CONFIRMATION:
            raise ValueError("Confirm the reviewed Vaelor upgrade before it runs.")

        # check
        self._checkpoint(15, "Checking for a pinned Vaelor release", "downloading")
        manifest = self._release_source().latest_manifest()
        if manifest is None:
            raise ValueError("No pinned Vaelor release is offered; there is nothing to upgrade to.")
        if payload.get("to_version") != manifest.version:
            raise ValueError(
                "The confirmed target no longer matches the offered release."
            )
        eligibility = upgrade_eligibility(CURRENT_VERSION, manifest)
        if not eligibility["eligible"]:
            raise ValueError(eligibility["reason"])

        # fetch
        staging = Path(state_path("upgrade/staged"))
        staging.mkdir(parents=True, exist_ok=True)
        staged = (staging / manifest.wheel_name).resolve()
        if staging.resolve() not in staged.parents:
            raise ValueError("The staged wheel path is invalid.")
        self._checkpoint(40, "Fetching the pinned release wheel", "downloading")
        self._release_source().fetch(manifest, staged)

        # verify-artifact (#130): only the manifest-pinned bytes may be staged.
        self._checkpoint(70, "Verifying the release wheel", "validating")
        actual_bytes = staged.stat().st_size
        digest = hashlib.sha256()
        with staged.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_bytes != manifest.bytes or actual_sha != manifest.sha256:
            staged.unlink(missing_ok=True)
            raise ValueError(
                "The fetched wheel did not match the pinned length or digest; "
                "nothing was installed."
            )

        # stage the one-use plan and hand off to the broker.
        self._checkpoint(
            88, "Staging the verified wheel for the upgrade broker", "starting"
        )
        plan = self._upgrade_plans().stage(
            job.get("actor", ""), staged, CURRENT_VERSION, manifest.version,
            manifest.sha256, manifest.bytes,
        )
        result = self._upgrade_client().apply(plan["id"], CONFIRMATION)

        return self.store.finish(
            job["id"],
            state="completed",
            message=(
                "Upgrade to {} accepted. The control plane will restart, verify "
                "the running version, and refresh deployed models."
                .format(manifest.version)
            ),
            result={
                "from_version": CURRENT_VERSION,
                "to_version": manifest.version,
                "sha256": manifest.sha256,
                "bytes": manifest.bytes,
                "staged_wheel": str(staged),
                "broker": result,
                # Deliberately no running_version here: this process cannot know
                # it, and claiming it from this process's own constant is the
                # #185 defect. The broker's readback owns that fact.
            },
        )
