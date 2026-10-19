"""The NPU Assistant's serving pair: one FastFlowLM runtime and one model package (VD-219).

The Assistant on the neural processor is two downloads that only work together:
the FastFlowLM portable runtime (``flm`` + ``flm-real`` + its libraries) and the
model package FastFlowLM serves (the q4nx weights with the tokenizer, config and
chat template that runtime expects). A package records the runtime it was
converted for: FastFlowLM 1.0.3 moved Qwen3.5 to Q4_K and refuses an older
package ("version 1.0.2 < 1.0.3"), and an older runtime cannot read a newer
package. So the two are pinned **as a pair** here, and nothing installs one
without the other (LESSONS 6: a runtime pin in the installer and a model pin in
the catalog drifted apart once already, VD-218).

:data:`CURRENT_PAIR` is what a 1.5.3 box serves: FastFlowLM 1.0.2 and the
ff-4b-h package published with 1.5.3. :data:`TARGET_PAIR` is the pair a box is
moved to, by the installer and the console update alike (VD-217), with an
automatic return to the previous pair when the new one fails its health check
(`npu_runtime_switch`). It was ``None`` until a package passed the owner's gate
(VD-219); it now names VD-219 round 3, pass 2 (``ff-4b-v219r3p2-flm1.0.9``,
owner: "Ship it now", 2026-10-19). **Shipping a later package is the same one
edit: set** :data:`TARGET_PAIR` **to** ``ServingPair(FLM_1_0_9, <its Package>)``,
with the record ``tools/npu_model_package.py`` prints for the published parts
and the probe recorded from one served run on the NPU.

Everything else reads the pair from here: the model catalog's pin
(`model_catalog.NPU_RELEASE_MODELS`), the part names the release install
accepts (`flm_npu_release`), the installer's runtime fetch
(`npu_runtime_files`), and ``deploy/fetch-npu-model.sh`` through
``python -I -m vaelor.npu_runtime_pin shell``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence


class Runtime(NamedTuple):
    """One FastFlowLM portable Linux release, verified by its SHA-256."""

    version: str
    url: str
    sha256: str
    size_bytes: int


class Package(NamedTuple):
    """One NPU model package: a tar of one model folder, published as split release parts."""

    #: The package's own name. A package made by ``tools/npu_model_package.py``
    #: carries it inside the folder (:data:`PACKAGE_MARKER`); the 1.5.3 package
    #: predates that and is recognised by :data:`LEGACY_PACKAGE_ID`.
    package_id: str
    #: The folder FastFlowLM's model list names for :attr:`tag`.
    folder: str
    #: The FastFlowLM tag the Assistant serves.
    tag: str
    #: The release asset names: ``<part_prefix>NN``.
    part_prefix: str
    #: The reassembled tar's SHA-256: the only trust anchor (never the release JSON).
    sha256: str
    size_bytes: int
    #: The FastFlowLM version the package was converted for.
    runtime_version: str
    #: A fixed question and the answer this package gives it at temperature 0,
    #: recorded from one served run when the package is released
    #: (``tools/npu_model_package.py --probe-prompt/--probe-answer``). The health
    #: check after a move asks it and compares (`npu_runtime_switch.serves`):
    #: answering is not proof of serving *this* model (VD-219 window 2: FastFlowLM
    #: 1.0.9 replaced a package it judged outdated with the stock model, which
    #: answered). Empty on the 1.5.3 package, which predates the check.
    probe_prompt: str = ""
    probe_answer: str = ""


class ServingPair(NamedTuple):
    runtime: Runtime
    package: Package

    @property
    def key(self) -> str:
        """How the pair is named on disk and in the journal: ``<runtime>+<package>``."""
        return "{}+{}".format(self.runtime.version, self.package.package_id)


#: A file inside a package's model folder naming the package (JSON: ``package_id``,
#: ``runtime_version``). FastFlowLM ignores it.
PACKAGE_MARKER = "vaelor-package.json"

#: The 1.5.3 package has no :data:`PACKAGE_MARKER`; a box holding its folder
#: under FastFlowLM 1.0.2 holds this package.
LEGACY_PACKAGE_ID = "ff-4b-h-flm1.0.2"

FLM_1_0_2 = Runtime(
    "1.0.2",
    "https://github.com/ROCm/FastFlowLM/releases/download/v1.0.2/fastflowlm_1.0.2_linux.tar.gz",
    "7a6f8a30db3ce832cc546c89ce6bc6e686289f93a75f457fae575cfa33a4604a",
    40_929_563,
)

#: FastFlowLM 1.0.9 (2026-10-15). Its own ``flm`` wrapper puts the XRT 2.25
#: multiarch mirror (``lib/x86_64-linux-gnu``) first on ``LD_LIBRARY_PATH``;
#: Vaelor launches through that wrapper (`flm_service.FLM_BINARY`), so the
#: "No NPU device" seen when ``flm-real`` is run bare (kb N172) does not apply.
FLM_1_0_9 = Runtime(
    "1.0.9",
    "https://github.com/ROCm/FastFlowLM/releases/download/v1.0.9/fastflowlm_1.0.9_linux.tar.gz",
    "6ecd27d59f1f17650faaedc11787c4b755912a1725508ef15a772a6cc98a582e",
    48_082_541,
)

FF_4B_H_FLM_1_0_2 = Package(
    LEGACY_PACKAGE_ID,
    "Qwen3.5-4B-NPU2",
    "qwen3.5:4b",
    "qwen35-4b-npu2.tar.part",
    "80401bba2fc26896a7cac8ef23629ee495b20b1b853bfb9ce5202562772ca0e4",
    3_597_588_480,
    "1.0.2",
)

CURRENT_PAIR = ServingPair(FLM_1_0_2, FF_4B_H_FLM_1_0_2)

#: VD-219 round 3, pass 2: packaged on the Z2 (~/r3pkg/p2, ``tools/npu_model_package.py``
#: at 4b1dce86). The probe is question B of the round-3 NPU window (2026-10-18, kb D369):
#: the answer :func:`npu_runtime_switch.ask` got (32 tokens, temperature 0, penalty 0),
#: byte-identical over three FastFlowLM 1.0.9 starts. Question A ("closer to its limit")
#: is answered wrongly by this package, so it is not the known answer.
FF_4B_V219R3P2_FLM_1_0_9 = Package(
    "ff-4b-v219r3p2-flm1.0.9",
    "Qwen3.5-4B-NPU2",
    "qwen3.5:4b",
    "ff-4b-v219r3p2-flm1.0.9.tar.part",
    "e6756e276ae021954ba393524c3b99ed5e9873618dd4626c5594b1c0024bc632",
    3_430_471_680,
    "1.0.9",
    probe_prompt=("Readings: memory 12.4 of 16.0 GB used (78%). Root disk 410 of 500 GB used "
                  "(82%). Which is fuller, the memory or the root disk? Answer in one sentence."),
    probe_answer="The root disk is fuller, at 410 of 500 GB (82%), compared to the memory at 12.4 of",
)

#: The pair boxes move to; ``None`` would keep every box on :data:`CURRENT_PAIR`.
TARGET_PAIR: Optional[ServingPair] = ServingPair(FLM_1_0_9, FF_4B_V219R3P2_FLM_1_0_9)


def serving_pair() -> ServingPair:
    """The pair this release installs: the target once one is pinned, else today's."""
    return TARGET_PAIR if TARGET_PAIR is not None else CURRENT_PAIR


def release_source(environ: Optional[Dict[str, str]] = None) -> str:
    """The GitHub release index the package's parts are listed in.

    Every Vaelor release re-hosts the package's parts beside its wheel, so the
    release is this version's unless ``VAELOR_RELEASE_TAG`` names another (as
    the installer's wheel download reads it), and ``VAELOR_NPU_RELEASE_SOURCE``
    replaces the whole URL (a mirror). The model catalog spells the same
    default (`model_catalog.NPU_RELEASE_SOURCE`).
    """
    from .model_catalog import NPU_RELEASE_SOURCE

    if environ is None:
        return NPU_RELEASE_SOURCE
    override = environ.get("VAELOR_NPU_RELEASE_SOURCE")
    if override:
        return override
    from .version import __version__

    return "https://api.github.com/repos/{}/releases/tags/{}".format(
        environ.get("VAELOR_REPO", "ShadowLayer90/vaelor"),
        environ.get("VAELOR_RELEASE_TAG", "v" + __version__),
    )


def pair_problems(pair: ServingPair) -> List[str]:
    """Why a pair cannot be installed as written; empty when it can."""
    problems = []
    if pair.package.runtime_version != pair.runtime.version:
        problems.append("the package was converted for FastFlowLM {}, not {}".format(
            pair.package.runtime_version, pair.runtime.version))
    for name, digest in (("runtime", pair.runtime.sha256), ("package", pair.package.sha256)):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            problems.append("the {} SHA-256 is not a hex digest".format(name))
    if not pair.runtime.url.startswith("https://"):
        problems.append("the runtime is not fetched over HTTPS")
    for value in (pair.package.package_id, pair.package.folder, pair.runtime.version):
        if not value or "/" in value or value.startswith(".") or not value.isprintable():
            problems.append("{!r} is not a plain name".format(value))
    return problems


def pair_from_record(record: Dict[str, object]) -> ServingPair:
    """A pair from its JSON form (``{"runtime": {...}, "package": {...}}``), for bench copies."""
    runtime = record["runtime"]
    package = record["package"]
    if not isinstance(runtime, dict) or not isinstance(package, dict):
        raise ValueError("a pair record holds a runtime object and a package object")
    return ServingPair(Runtime(**runtime), Package(**package))


def pair_record(pair: ServingPair) -> Dict[str, Dict[str, object]]:
    return {"runtime": dict(pair.runtime._asdict()), "package": dict(pair.package._asdict())}


def shell_lines(pair: Optional[ServingPair] = None) -> List[str]:
    """``KEY=value`` lines for ``deploy/fetch-npu-model.sh``; every value is a plain token."""
    pair = pair or serving_pair()
    values = {
        "NPU_PACKAGE_ID": pair.package.package_id,
        "NPU_MODEL_NAME": pair.package.folder,
        "NPU_PART_PREFIX": pair.package.part_prefix,
        "NPU_PACKAGE_SHA": pair.package.sha256,
        "NPU_RUNTIME_VERSION": pair.runtime.version,
        "NPU_PACKAGE_BYTES": str(pair.package.size_bytes),
    }
    lines = []
    for key, value in values.items():
        if not value or any(not (c.isalnum() or c in "._:+-") for c in value):
            raise ValueError("{} is not a plain token: {!r}".format(key, value))
        lines.append("{}={}".format(key, value))
    return lines


def main(argv: Sequence[str]) -> int:
    """``shell``: the serving pair as ``KEY=value`` lines; ``json``: the whole pair."""
    verb = list(argv[1:])
    if verb == ["shell"]:
        print("\n".join(shell_lines()))
        return 0
    if verb == ["json"]:
        print(json.dumps(pair_record(serving_pair()), indent=2, sort_keys=True))
        return 0
    sys.stderr.write("usage: python -I -m vaelor.npu_runtime_pin shell|json\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
