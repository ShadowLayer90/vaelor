"""What a local model costs to run, measured on the machine that runs it.

The catalog records what an artifact *weighs* - `download_bytes`, digest-verified
against the publishing registry. This module records what it *costs*, which is a
different and much larger number: Qwen3 1.7B is a 1,107,409,472 byte download and
occupies 3,157 MB resident at steady state on a Pi. A budget built from the
download size understates by roughly three times.

**Steady state, not startup.** The figure that matters is not observable when a
model finishes loading. Measured on the Pi, Qwen3 1.7B at 2,048 tokens and one
slot:

    prompts   RSS_MB   anon_MB   file_MB
    0           2387      1329      1058
    20          3111      2053      1058
    40          3157      2099      1058
    60          3157      2099      1058
    80          3157      2099      1058

llama.cpp allocates working buffers lazily and reaches a ceiling ~32% above the
startup reading, plateauing around 40 prompts. Every container limit set on
2026-08-08 was derived from a startup number, and every OOM kill that day follows
from that (#117). There is no leak; there was a reading taken too early.

**This module refuses to extrapolate, and that is its main feature.** Two
predictions were made during the same session from `weights + KV + fixed
overhead`, and both were low - the 4B by 21%, the 3B by 18%. Memory does not
scale with parameter count on this hardware: Llama-3.2-3B costs *more* than
Qwen3 4B despite 455 MiB fewer weights. A combination that has not been measured
returns :data:`UNMEASURED` and the caller must decide what to do about it. An
estimate that reads like a measurement is how a limit gets set 336 MB too low.

**Nothing here is measured on the Z2** (VD-071 is Pi-scoped). Its runtime is
flm-real on the NPU rather than llama.cpp on GGUF, it does no prefix caching
(VD-008), and its memory comes from GTT at a driver default. Its curve is not
assumed to look like this one.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

MIB = 1024 ** 2

#: Returned for any combination that has not been measured. Deliberately not a
#: number: a caller that wants to size something must handle not knowing, rather
#: than receive a guess it cannot distinguish from evidence.
UNMEASURED = "unmeasured"

#: A footprint observed after the process reached plateau.
MEASURED_STEADY = "measured-steady-state"

#: A footprint observed at load, before warm-up. Recorded when that is all that
#: exists, and never usable as a reservation - it is a lower bound and the real
#: figure is materially higher.
MEASURED_STARTUP_ONLY = "measured-startup-only"

#: A footprint observed under sustained load that **had not stopped rising**
#: when the observation ended.
#:
#: Distinct from :data:`MEASURED_STEADY`, and the distinction is the point. The
#: 1.7B plateaus by 40 prompts and this module was written expecting that of
#: everything. The 4B at 4,096 does not: measured 2026-08-10, anonymous memory
#: was still climbing ~13 MB per prompt at prompt 40 with no flattening.
#:
#: It is a **floor**, not a ceiling - a real observation of this model in this
#: configuration, so it outranks an estimate, but the true steady state is
#: higher by an unknown amount. Sizing from it is deliberate and the margin is
#: doing real work. Replace it with a :data:`MEASURED_STEADY` record when
#: somebody runs it long enough to flatten.
MEASURED_RISING = "measured-still-rising"

#: **How the number was counted.** RSS and the cgroup's own `memory.current`
#: are not comparable, and the gap is large: RSS counts reclaimable file-backed
#: pages the kernel drops under pressure, so the same running model reads about
#: 1.7 GB higher under RSS - 5,383 MB against 3,632 MiB for the model the Pi
#: ships. A container limit is enforced against the cgroup figure, so that is
#: the one a limit may be derived from.
#:
#: The three older rows below predate the distinction and are RSS; re-measuring
#: them on the cgroup basis is owed work, and until it is done **this** field is
#: what separates them. It was claimed for one release that `source` did that.
#: It does not and cannot: all four rows are legitimately `MEASURED_STEADY`,
#: because how long you waited and how you counted are different questions.
RSS_BASIS = "rss"
CGROUP_BASIS = "cgroup-memory-current"

#: How many distinct prompts the Pi needed before memory stopped rising. Used by
#: anything that measures a new model: read the footprint before this and the
#: number is wrong in the dangerous direction.
WARM_UP_PROMPTS = 40


def _record(
    startup_bytes: int,
    steady_state_bytes: Optional[int],
    seconds_per_answer: float,
    cold_start_seconds: float,
    source: str,
    provenance: str,
    basis: str = RSS_BASIS,
) -> Dict[str, Any]:
    # `basis` defaults to RSS because every row that predates the field is one,
    # and a default that silently claimed the newer basis would be the same
    # mistake in the other direction.
    return {
        "startup_bytes": startup_bytes,
        "steady_state_bytes": steady_state_bytes,
        "seconds_per_answer": seconds_per_answer,
        "cold_start_seconds": cold_start_seconds,
        "source": source,
        "basis": basis,
        "provenance": provenance,
    }


#: Keyed by (catalog id, platform, context tokens, slots). The key carries the
#: serving configuration because the footprint depends on it: the same 1.7B at
#: 4,096 tokens exceeded a 3,584 MB limit that it lives inside at 2,048.
FOOTPRINTS: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {
    ("qwen3-1.7b-q4", "arm64", 2048, 1): _record(
        startup_bytes=2387 * MIB,
        steady_state_bytes=3157 * MIB,
        seconds_per_answer=10.0,
        cold_start_seconds=6.0,
        source=MEASURED_STEADY,
        provenance=(
            "Pi 192.168.4.62, 2026-08-08. RSS sampled from /proc/<pid>/status "
            "at 0/20/40/60/80 prompts: 2387, 3111, 3157, 3157, 3157 MB. "
            "Anonymous memory 1329 -> 2099 MB; file-backed constant at 1058 MB."
        ),
    ),
    # Measured when 8,192 was the recommended window, on the reasoning that it
    # was the smallest power of two above a 4,668-token agent prompt. That
    # prompt measures 1,344 and the recommendation is now 4,096 (VD-075), so
    # **this row is no longer the shipped configuration** - it is a measurement
    # of a window nothing deploys at by default. It stays because it is real,
    # and because 8,192 remains reachable by an explicit request. There is no
    # row at 4,096, which is why the shipped Pi deploy currently sizes from the
    # estimate; see the same-named test in test_model_sizing_during_replace.
    ("qwen3-4b-q4", "arm64", 8192, 1): _record(
        startup_bytes=5355 * MIB,
        steady_state_bytes=5542 * MIB,
        seconds_per_answer=20.0,
        cold_start_seconds=37.0,
        source=MEASURED_STEADY,
        provenance=(
            "Pi 192.168.4.62, 2026-08-08, limit 6656 MB. RSS at 0/20/40 "
            "prompts: 5355, 5542, 5425 MB; anonymous 3541 -> 4228 -> 4228, "
            "plateaued. File-backed *fell* 1814 -> 1197 MB as the cgroup "
            "reclaimed page cache, so RSS overstates the hard requirement and "
            "the anonymous figure is the floor. Zero kernel OOM events; "
            "1,740 MB left available to the rest of the machine."
        ),
    ),
    # The configuration the Pi actually ships (VD-076, VD-077): the 2507
    # refresh at Q4_0, 4,096 tokens, one slot. Filling the gap
    # `test_the_shipped_window_has_no_measured_footprint_and_says_so` was
    # holding open - the shipped deploy had been sizing from the arithmetic
    # estimate that runs ~21% low and OOM-killed containers (#117).
    #
    # **This is a floor, not a plateau.** Read the source field before using
    # the number: at prompt 40 anonymous memory was still climbing and the
    # curve had not turned over.
    ("qwen3-4b-instruct-2507-q4-0", "arm64", 4096, 1): _record(
        # **cgroup `memory.current`, not RSS.** The rows above record RSS, and
        # on this row that would be 5,383 MB - which exceeds what the Pi can
        # spare and makes the sizing path *refuse a configuration that
        # demonstrably runs*. RSS counts file-backed pages the cgroup reclaims
        # under pressure (2,260 MB of them here); `memory.current` is what the
        # limit is enforced against, so it is what a limit must be derived
        # from. The two older rows are therefore on a different basis and are
        # not comparable with this one - re-expressing them is owed work, and
        # until it is done the `basis` field below is what distinguishes them.
        # This comment named `source` for one release, which was wrong: all
        # four rows are `MEASURED_STEADY`, and rightly so, because that field
        # records how long the reading waited rather than how it counted.
        startup_bytes=3338 * MIB,
        steady_state_bytes=3632 * MIB,
        seconds_per_answer=29.2,
        cold_start_seconds=41.0,
        source=MEASURED_STEADY,
        basis=CGROUP_BASIS,
        provenance=(
            "Pi 192.168.4.62, 2026-08-10, limit 5120 MB, engine "
            "llama.cpp build 10335 (74ce15741) at "
            "sha256:2a8440d3aa0be70bf1d1824d2721fc5001616d46be4348b1c1b38fa30af4fe1c, "
            "**--cache-ram 256**. 80 distinct prompts, 96 max_tokens, the "
            "appliance's ordinary workloads running. RSS at 0/10/20/30/40/50/"
            "60/70/80: 5091, 5240, 5351, 5351, 5351, 5368, 5368, 5368, 5383 "
            "MB. Anonymous 2831 -> 3091 by prompt 20 and flat thereafter, "
            "3122 at prompt 80 - a residual 0.5 MB per prompt. cgroup "
            "memory.current steady at 3601-3632 MB, 71% of the limit. Zero "
            "OOM events. "
            "THE SAME RUN AT THE ENGINE DEFAULT of --cache-ram 8192 did not "
            "plateau at all: anon 2831 -> 3430 by prompt 40, +13 MB per "
            "prompt, cgroup 87% of the limit and climbing. The default is "
            "larger than the Pi's whole 7,931 MB, so nothing ever evicted. "
            "Bounding it cost nothing - 29.2 s per answer against 30.3 s. "
            "The cgroup figure is the hard requirement; RSS overstates it by "
            "the file-backed pages the cgroup reclaims under pressure. "
            "VERIFIED END TO END: this 3,632 MB figure derives a 4,352 MiB "
            "limit and a 217 MiB cache, and that exact rendered configuration "
            "was then deployed and re-run - 40 prompts, plateau at prompt 20, "
            "cgroup steady at 3,146-3,162 MB (73% of limit, 27% headroom), "
            "28.4 s per answer, restarts=0, OOMKilled=false. The recorded "
            "figure and the configuration it produces are a fixed point, "
            "which is deliberate: sizing from the 3,162 MB seen at 217 MiB of "
            "cache would derive a smaller limit that nothing has run."
        ),
    ),
    ("qwen3-4b-q4", "arm64", 2048, 1): _record(
        startup_bytes=5055 * MIB,
        steady_state_bytes=5100 * MIB,
        seconds_per_answer=20.0,
        cold_start_seconds=45.0,
        source=MEASURED_STEADY,
        provenance=(
            "Pi 192.168.4.62, 2026-08-08. Sized to its cap at startup rather "
            "than warming into it: 5055 MB immediately, +11 MB over five "
            "prompts, cgroup memory.current 5106 MB against a 5120 MB limit. "
            "The steady-state figure is therefore close to the startup one "
            "*for this limit*, and is not evidence about a smaller cap."
        ),
    ),
}


#: `uname -m` to the name this table is keyed by. The sizing path receives a
#: hardware snapshot carrying the kernel's architecture, and a lookup that
#: silently misses because one side says `aarch64` and the other `arm64` would
#: report "unmeasured" for a model that *is* measured - a false absence, which
#: is the failure class that cost the most time on 2026-08-08.
PLATFORM_ALIASES = {
    "aarch64": "arm64",
    "arm64": "arm64",
    "x86_64": "amd64",
    "amd64": "amd64",
}


def normalise_platform(value: Optional[str]) -> Optional[str]:
    return PLATFORM_ALIASES.get(str(value or "").strip().lower())


def identify(repo: Optional[str], file: Optional[str]) -> Optional[str]:
    """The catalog id for an artifact, by the repository and file that name it.

    Resolved through the catalog rather than by a byte count, because two
    publishers' Q4_K_M builds of the same model share a filename and sit 1,056
    bytes apart (VD-065). The pair is what the download path already verifies.
    """
    from .model_catalog import LOCAL_MODEL_CATALOG

    for entry in LOCAL_MODEL_CATALOG:
        if entry["repo"] == repo and entry["file"] == file:
            return entry["id"]
    return None


def footprint(
    model_id: str, platform: str, context_tokens: int, slots: int
) -> Optional[Dict[str, Any]]:
    """What this model costs in this configuration, or ``None`` if unmeasured."""
    record = FOOTPRINTS.get((model_id, platform, context_tokens, slots))
    return dict(record) if record else None


def reservation_bytes(
    model_id: str, platform: str, context_tokens: int, slots: int
) -> Dict[str, Any]:
    """The memory to reserve, or a statement that it is not known.

    Never returns a number derived from anything but an observation of this
    model in this configuration on this platform. The alternative - scaling from
    a nearby measurement - was tried twice by hand during the session that
    produced this module and was 18% and 21% low both times.
    """
    record = footprint(model_id, platform, context_tokens, slots)
    if record is None:
        return {
            "bytes": None,
            "source": UNMEASURED,
            "reason": (
                "No footprint has been measured for {} on {} at {} tokens with "
                "{} slot(s). Measure it - serving at least {} distinct prompts "
                "before reading - rather than scaling from another model: "
                "memory does not track parameter count here, and a 3B has been "
                "measured costing more than a 4B."
            ).format(model_id, platform, context_tokens, slots, WARM_UP_PROMPTS),
        }
    if record["source"] == MEASURED_RISING:
        # Without this branch the constant was decorative: declared, documented
        # at length, produced by no row and read by nothing, so a record that
        # ever carried it would have been sized from exactly like a settled
        # one. A floor used as a limit is the OOM kill it was named to prevent.
        return {
            "bytes": None,
            "source": MEASURED_RISING,
            "startup_bytes": record["startup_bytes"],
            "basis": record["basis"],
            "reason": (
                "The reading for {} was still climbing when it was taken, so it "
                "is a floor and not a plateau, and must not be used as a limit. "
                "Re-measure with the prompt cache bounded, serving at least {} "
                "distinct prompts, and record the value it settles at."
            ).format(model_id, WARM_UP_PROMPTS),
        }
    if record["source"] == MEASURED_STARTUP_ONLY or record["steady_state_bytes"] is None:
        return {
            "bytes": None,
            "source": MEASURED_STARTUP_ONLY,
            "startup_bytes": record["startup_bytes"],
            "reason": (
                "Only a startup reading exists for {} ({} bytes). Steady state "
                "runs ~32% higher on this platform, so this is a lower bound "
                "and must not be used as a limit."
            ).format(model_id, record["startup_bytes"]),
        }
    return {
        "bytes": record["steady_state_bytes"],
        "source": record["source"],
        # Carried out to the caller rather than left in the table, because a
        # figure that travels without saying how it was counted is what let a
        # cgroup row and an RSS row be compared as though they meant the same
        # thing (VD-040's rule, applied to memory rather than to storage).
        "basis": record["basis"],
        "provenance": record["provenance"],
    }


def fits(
    model_id: str, platform: str, context_tokens: int, slots: int, available_bytes: int
) -> Dict[str, Any]:
    """Whether this model fits, refusing to answer when the cost is unknown.

    A `False` here means measured-and-too-large. It never means "probably not":
    an unmeasured combination reports `None`, because telling an owner a model
    does not fit when nobody has checked is the same defect as telling them it
    does.
    """
    reservation = reservation_bytes(model_id, platform, context_tokens, slots)
    if reservation["bytes"] is None:
        return {
            "fits": None,
            "source": reservation["source"],
            "reason": reservation["reason"],
        }
    return {
        "fits": reservation["bytes"] <= available_bytes,
        "required_bytes": reservation["bytes"],
        "available_bytes": available_bytes,
        "source": reservation["source"],
    }


def identify_by_file(file: Optional[str]) -> Optional[str]:
    """The catalog id for an artifact known only by its filename.

    A deployed Compose file records `LLAMA_ARG_MODEL` and nothing about where
    the artifact came from, so this is all the running configuration knows.

    **Ambiguity returns None.** VD-065: two publishers' Q4_K_M builds of the
    same model share a filename and differ by 1,056 bytes, which is why
    :func:`identify` insists on the repository as well. Guessing between them
    here would put one publisher's measurements against another's artifact -
    the exact defect the pair exists to prevent - so a filename that more than
    one catalog entry claims resolves to nothing at all.
    """
    from .model_catalog import LOCAL_MODEL_CATALOG

    if not file:
        return None
    matches = {
        entry["id"] for entry in LOCAL_MODEL_CATALOG if entry["file"] == file
    }
    return matches.pop() if len(matches) == 1 else None
