"""The executor's release-model install: a fine-tuned NPU model from its pinned release.

Moved out of :mod:`vaelor.executor` unchanged, as a mixin :class:`vaelor.executor.JobExecutor`
composes, so the executor keeps headroom under the 1,000-line ceiling. It is
self-contained: its collaborators (the catalog, the root bridge, the on-disk
model check) are imported when it runs, and it reaches the executor only
through ``self._checkpoint`` and ``self._deploy_npu_assistant``.
"""

from __future__ import annotations

from typing import Any, Dict


class ExecutorReleaseModelMixin:
    """``_install_release_model`` for :class:`vaelor.executor.JobExecutor`."""

    def _install_release_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Install and serve a fine-tuned NPU model from its pinned release.

        Unlike a Hugging Face GGUF (download then deploy), a fine-tune arrives as
        a release: the job carries only the flm tag, and the catalog holds the
        pinned source URL and sha256 - the trust anchor is the code shipped in the
        wheel, never the job payload. The download, sha-verify and unpack into the
        snap paths flm serves from happen as root behind the hardware bridge
        (:meth:`HardwareBridgeClient.flm_install_release`), then the shared NPU-
        assistant deploy the reconcile uses brings the model up and health-checks
        it - here driven by an explicit plan built from the known tag rather than
        the fit ladder, which does not stock release models. A long client timeout
        covers the multi-GB download; the bridge serialises it under its flm lock.
        """
        from .flm_service import npu_model_present
        from .hardware_bridge import HardwareBridgeClient
        from .model_catalog import catalog_release_for_tag

        release = catalog_release_for_tag(str(payload.get("tag") or ""))
        if release is None:
            raise ValueError(
                "No release-installable on-device model matches this request."
            )
        # Report progress. The download is a single multi-GB blocking call into
        # the root bridge, so this cannot tick byte-by-byte, but a person watching
        # the setup screen must see that a large download is under way and roughly
        # where it is - not a silent gap that reads as "nothing is happening"
        # (the setup UX defect). The states are the ones UpdateJobStatus renders.
        self._checkpoint(
            5, "Preparing the on-device model install.", state="running")
        # Skip the multi-GB download when the model is already installed on disk.
        # `fetch-npu-model.sh` installs the NPU model into
        # `/var/lib/vaelor/flm/models` on a clean box, and the first-boot
        # auto-enable enqueues this deploy to serve+pin it. Re-downloading it
        # would be pointless (and would fail on a box with no release source), so
        # a present model dir goes straight to the serve+pin path below.
        if not npu_model_present():
            self._checkpoint(
                10,
                "Downloading and verifying the on-device model (about 3.4 GiB). "
                "This can take a few minutes on a slow connection.",
                state="downloading",
            )
            HardwareBridgeClient(timeout=1200).flm_install_release(
                release["source_url"], release["sha256"]
            )
        else:
            # Stay below the deploy's 45%: this is not the final progress, the
            # serve stages below are.
            self._checkpoint(
                15, "The on-device model is already on disk.", state="running")
        # The install placed OUR flm-real + model in the snap paths, so serve the
        # known tag directly. A pathless `_deploy_model` would consult the fit
        # ladder (`should_serve_on_npu`), which stocks only Hugging Face GGUFs and
        # cannot pick a release model on a fresh box - it would fall through to the
        # llama.cpp path and fail with "Choose a downloaded managed GGUF model".
        # We know the tag, so we build the plan `_deploy_npu_assistant` needs and
        # call it: launch flm-real, health-check, pin + activate the managed
        # credential on the served tag - the same path the reconcile uses.
        #
        # No checkpoint here: `_deploy_npu_assistant` below reports "Starting the
        # neural processor model server" at 45% and climbs to 95%, so its first
        # step is the next thing the progress bar shows - keeping it monotonic
        # rather than jumping to 90% and then back to 45%.
        plan = {
            "flm_tag": release["tag"],
            "context_tokens": 16384,
            "model": release["name"],
            "context_reason": "",
        }
        return self._deploy_npu_assistant({"surface": "assistant"}, {"plan": plan})
