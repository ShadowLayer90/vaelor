"""The backend benchmark catalogue: numbers, and where each one was measured.

Split out of ``inference_tuning`` (#206) because that module reached the
1,000-line ceiling and because these constants are one cohesive thing: the
throughput matrix and its provenance. ``inference_tuning`` re-exports every name
here, so it stays the one public module for tuning facts.

**Everything in this file was measured on one machine, and that machine is
named on every number that leaves it.** ``MEASURED_ON`` is the reference bench
box; the figures below describe *its* accelerators. On any other machine they
are reference provenance and nothing else, so a caller that surfaces them must
carry ``benchmarked_on`` (or say ``measured: false``) rather than present them
as this machine's own reading (#205, LESSONS 5/6). Keeping the provenance on the
constant, not in a distant docstring, is task #57's rule.
"""

from __future__ import annotations


#: Where every number in this module was measured. Printed in reasons so an
#: operator can tell a measurement from an assumption without reading the code.
#: On any machine that is not this box it is *provenance*, never this machine's
#: own status - carried as ``benchmarked_on``, not ``measured_on`` (#205).
MEASURED_ON = "HP Z2 Mini G1a (Ryzen AI Max, 16 GiB VRAM carve-out + 22.78 GiB GTT)"


#: The build every number below was taken on. Recorded because the previous set
#: was not: those figures came from snap revision 360 (v11.5.1), the machine
#: auto-refreshed to 378 mid-benchmark, and reasoning from the superseded
#: numbers is what produced the wrong default.
LEMONADE_SNAP_REVISION = 378
LEMONADE_VERSION = "v11.5.2"
SUPERSEDED_SNAP_REVISION = 360
SUPERSEDED_VERSION = "v11.5.1"

#: Fraction of the VRAM carve-out a Vulkan deployment may plan to occupy.
#: ``Qwen3.6-27B`` on Vulkan was measured at 16,268 MB of 16,384 — 99.3%. No
#: longer a selector: ROCm is the default and draws on the larger pool. It is
#: kept because an operator who *overrides* to Vulkan is entitled to be told
#: when the model they picked will sit on that ceiling.
VULKAN_CARVE_OUT_HEADROOM_FRACTION = 0.90

#: The observation the headroom fraction was set from.
#:
#: **The two figures come from different runs and are now labelled as such.**
#: The memory numbers are corroborated twice and are sound. The temperature was
#: not: 95 °C came from the benchmark matrix, while a dedicated probe of the
#: *same configuration* recorded 82 °C. Quoting the higher number without
#: saying where it came from is the kind of silent worst-case that turns a
#: measurement into an argument. The dedicated probe is the better instrument,
#: so it is what an operator is shown; the matrix figure is kept beside it
#: rather than discarded.
VULKAN_CARVE_OUT_CEILING_OBSERVED = {
    "model": "Qwen3.6-27B",
    "used_mib": 16268,
    "carve_out_mib": 16384,
    "utilisation": 0.993,
    "memory_source": "benchmark matrix, corroborated by a dedicated probe",
    "peak_temperature_c": 82,
    "peak_temperature_source": "dedicated probe of the same configuration",
    "matrix_peak_temperature_c": 95,
    "matrix_peak_temperature_source": "benchmark matrix run",
}

#: Measured throughput per backend on snap revision 378 (v11.5.2).
#:
#: **These reverse the previous conclusion.** On revision 360 ROCm prefill was
#: far behind Vulkan and Vulkan was the obvious default. The box auto-refreshed
#: to 378 mid-benchmark; the validation pass found ROCm prefill up by as much as
#: +262% while Vulkan and the NPU were bit-for-bit unchanged. On 378 ROCm wins
#: prefill on 7 of 8 models and Vulkan wins decode on 7 of 8.
#:
#: A patch release does not normally move prefill that far, so the gain is
#: almost certainly a newer bundled llama.cpp / ``therock`` gfx1151 runtime
#: rather than anything in Lemonade's own code. That matters for whoever reads
#: this next: the number to re-measure against is the runtime, not the snap
#: version, and a future refresh could move it again in either direction.
#: **Provenance correction, not a conclusion change.** The headline pair was
#: 2445.1 ROCm against 2334.0 Vulkan, but those came from two different
#: harnesses: the ROCm figure was taken through Lemonade and the Vulkan figure
#: in a native container. The native harness reads 15-29% in Vulkan's favour,
#: so the published comparison *understated* ROCm's lead. Like for like, both
#: through Lemonade, it is 2445.1 against 2022.4 - ROCm ahead by 20.9% rather
#: than 4.8%. The conclusion survives and is stronger; only the citation was
#: wrong. ``lemonade_prefill_tps`` is the like-for-like number and is what the
#: decision reason quotes.
#:
#: This does not reopen the default. ``DEFAULT_GPU_BACKEND`` is ROCm by owner
#: decision (VD-003), taken twice months apart and once before any benchmark
#: existed. Do not "fix" it from these numbers in either direction.
BACKEND_MEASUREMENTS = {
    "llama-3.2-3b": {
        "vulkan": {
            "prefill_tps": 2334.0,
            "harness": "native-container",
            # Same model, same revision, measured through the same harness as
            # the ROCm figure below. The only number the two may be compared on.
            "lemonade_prefill_tps": 2022.4,
            "lemonade_harness": "lemonade",
            "snap_revision": LEMONADE_SNAP_REVISION,
        },
        "rocm": {
            "prefill_tps": 2445.1,
            "harness": "lemonade",
            "lemonade_prefill_tps": 2445.1,
            "lemonade_harness": "lemonade",
            "snap_revision": LEMONADE_SNAP_REVISION,
            # The single largest movement in the refresh, and the one the
            # ordering-flip detector caught: this model went from 6th to 1st in
            # the ROCm ranking between runs.
            "superseded_prefill_tps": 675.8,
            "superseded_snap_revision": SUPERSEDED_SNAP_REVISION,
            "prefill_change": 2.618,
        },
    },
    "27b": {
        # Vulkan was unchanged by the refresh, so these still describe 378.
        "vulkan": {
            "prefill_tps": 356.7, "ttft_seconds": 13.1,
            "snap_revision": LEMONADE_SNAP_REVISION,
        },
        # ROCm on this model has not been re-measured on 378. The revision-360
        # figures are kept for provenance and explicitly *not* presented as
        # current, because ROCm is exactly what the refresh changed.
        "rocm": {
            "prefill_tps": None,
            "snap_revision": None,
            "superseded_prefill_tps": 233.6,
            "superseded_ttft_seconds": 23.1,
            "superseded_snap_revision": SUPERSEDED_SNAP_REVISION,
        },
    },
}

#: How the two backends split the eight measured models on revision 378.
BACKEND_WINS = {
    "prefill": {"rocm": 7, "vulkan": 1, "models": 8},
    "decode": {"vulkan": 7, "rocm": 1, "models": 8},
}

#: Measured KV-quantization win, 3 B at 32 k context, ``q8_0`` K+V against f16.
#:
#: **This was published as a ROCm result and it is not one.** The underlying
#: cells are tagged ``[vulkan]``, so the backend is recorded here explicitly
#: rather than left to be inferred from whatever backend the reader's host
#: happens to run. Decode was also rounded to +21% from a measured +19.9%.
#: Neither correction changes the recommendation - quantizing the KV cache is
#: still a win on the model and window measured - but a constant in this module
#: naming the wrong backend is a defect in the one property the module exists
#: to guarantee.
#:
#: It is *not* a universal win: see ``MODEL_FLAG_OVERRIDES`` for a model that
#: loses a quarter of its prefill to the same setting on ROCm.
KV_QUANTIZATION_MEASUREMENT = {
    "model": "3B",
    "backend": "vulkan",
    "context_tokens": 32768,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "vram_change": -0.29,
    "prefill_change": 0.16,
    "decode_change": 0.199,
    "ttft_change": -0.14,
}

#: Measurements behind the per-model overrides, kept beside them so a reviewer
#: can check the claim without leaving the file.
MODEL_FLAG_MEASUREMENTS = {
    "gpt-oss-20b": {
        "flash_attention_forced_on_prefill_tps": 848.0,
        "flash_attention_auto_prefill_tps": 1544.0,
        "backend": "vulkan",
        # Revision 360, never re-validated on 378. Named `superseded_` for the
        # same reason the revision-360 backend figures are: ROCm is exactly
        # what the refresh changed, so a ROCm number from 360 describes a
        # build this appliance no longer runs.
        "superseded_ubatch_1024_prefill_change": 0.189,
        "superseded_ubatch_1024_ttft_change": -0.158,
        "superseded_ubatch_backend": "rocm",
        "superseded_ubatch_snap_revision": SUPERSEDED_SNAP_REVISION,
    },
    "gemma-4-26b-a4b": {
        "backend": "rocm",
        "kv_q8_0_prefill_change": -0.26,
        "reproduced_in_passes": 2,
        "snap_revision": LEMONADE_SNAP_REVISION,
    },
}
