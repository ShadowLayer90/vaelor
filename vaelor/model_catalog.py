"""The local models Vaelor can actually install, and nothing else.

First run on an HP Z2 Mini G1a, 48.9 GB of RAM, verbatim:

    Recommended
    **Install Qwen3 32B**
    Private local answers · about 1.1 GB · reviewed before download

and the catalog header beneath it, *"Vaelor recommends 32B for 48.9 GB RAM."*
The card actually badged RECOMMENDED FOR THIS HARDWARE was **Qwen3 1.7B**, and
following the flow through installed ``Qwen3-1.7B-Q4_K_M.gguf`` at
1,107,409,472 bytes. Three statements about one action, all different: the name
said 32B, the size said 1.1 GB, the install gave the 1.7B. A 32B at Q4 is about
20 GB, so **the size was the only honest figure of the three.**

There was no 32B anywhere in the catalog. The largest was Qwen3 4B.

The name came from a tier ladder that mapped a memory budget to a parameter
class and then looked the class up in a table of *names* - ``14B``, ``32B``,
``70B`` - which carried a search string and no artifact. Nothing downstream
could install any of them, and nothing checked that it could. Same shape as the
phantom NPU model names and the ``2.0.4`` left in the deploy profiles: a string
that outlived the thing it referred to.

So the recommendation now derives from this module, and this module holds only
entries with a resolvable repository *and* file - the ones a download can
actually be started from. Naming a model the catalog does not contain is no
longer something the recommendation can express, because it has no names of its
own to say.

**What this deliberately does not do is invent artifacts to fill the ladder.**
Adding ``Qwen/Qwen3-32B-GGUF`` here from memory would make the sentence true by
asserting a repository nobody in this tree has verified - the same defect in the
other direction. A machine that can hold more than the catalog offers is told
so, in :func:`headroom_note`, and pointed at the search box that already
exists. That is a smaller promise and a true one.

Having a repository and a file is not the same as resolving
--------------------------------------------------------

The first version of this module was guarded by a test asserting that ``repo``
and ``file`` were non-empty strings, which both were, of an entry naming
``Qwen/Qwen3-1.7B-GGUF`` / ``Qwen3-1.7B-Q4_K_M.gguf``. **That file is not in
that repository.** ``Qwen/Qwen3-1.7B-GGUF`` publishes one GGUF, ``Q8_0``, at
1,834,426,016 bytes; the Q4_K_M this appliance downloaded came from
``unsloth/Qwen3-1.7B-GGUF``, and 1,107,409,472 is *unsloth's* byte count. So the
entry paired one publisher's repository with another publisher's artifact and
that publisher's size, and every assertion the suite made about it passed. A
test that two strings are non-empty cannot tell a real artifact from a plausible
one, which is the same shape as asserting a value is under a cap while the cap
itself drifts.

Two things follow, and only the second is a fix:

* Both entries now name ``unsloth``, which is what the rest of this tree records
  the appliance actually fetching - ``/var/lib/vaelor/models/unsloth--Qwen3-4B-
  GGUF/Qwen3-4B-Q4_K_M.gguf`` on disk, ``unsloth/Qwen3-1.7B-GGUF Q4_K_M`` in the
  planner's own query - and each carries the artifact's ``sha256`` beside its
  byte count.
* ``tools/verify_model_catalog.py`` resolves every entry against the publishing
  registry and compares size **and digest**, and the release manifest refuses to
  seal without its passing report. That check needs the network, so it cannot
  live in the unit suite; it is a release gate rather than a warning because a
  warning is what the previous guard effectively was.

The digest is what makes the check discriminating rather than decorative.
``Qwen/Qwen3-4B-GGUF`` and ``unsloth/Qwen3-4B-GGUF`` both publish a file called
``Qwen3-4B-Q4_K_M.gguf``, 2,497,280,256 and 2,497,281,312 bytes - **1,056 bytes
apart**. A size comparison alone treats one publisher's file as evidence for the
other's; a sha256 comparison cannot.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

#: Where the fine-tuned on-device NPU model release is published. A fine-tune is
#: not in any public catalog, so the appliance pulls it from a pinned release
#: (model + its bundled flm-real runtime), verified against the entry's sha256.
#: Today this is a simulated local source (a loopback server standing in for the
#: release host); set ``VAELOR_NPU_RELEASE_SOURCE`` to the published release base
#: once it exists - the install code is the same either way.
NPU_RELEASE_SOURCE = os.environ.get(
    "VAELOR_NPU_RELEASE_SOURCE", "http://127.0.0.1:8899"
)


GB = 1000 ** 3

#: How a download size was arrived at.
#:
#: ``measured`` is a byte count read off the artifact itself - a completed
#: download, or the size the publishing registry reports for that exact file -
#: and it is the only source a catalog entry may now carry, because
#: ``tools/verify_model_catalog.py`` compares every recorded size against
#: upstream and an estimate cannot survive that comparison.
#:
#: ``upper-bound`` is a deliberate over-estimate that errs towards refusing an
#: install rather than starting one that cannot finish. It remains a legible
#: value - the frontend still renders "up to about" for it - but **no catalog
#: entry uses it**, and the release gate rejects one that does. The 4B entry
#: carried ``4 * GB`` against a real artifact of 2,497,281,312 bytes: a 60%
#: over-estimate, which is not a conservative bound but an uninformative one,
#: and a size bound exists for exactly one purpose - deciding whether a machine
#: can hold the thing.
SIZE_MEASURED = "measured"
SIZE_UPPER_BOUND = "upper-bound"


def _size_note(download_bytes: int, source: str) -> str:
    """The size sentence, derived from the number the fit check uses.

    Derived rather than written, because a hand-written note is exactly what
    said "about 1.1 GB" beside the name of a 20 GB model.

    GiB with a binary divisor (#145): the frontend renders model files in
    IEC units (`QUANTITY_MODES.model = "iec"`), so this sentence in decimal
    GB put "about 2.4 GB" in the catalog against "2.21 GiB" in Manage for
    the same file — two numbers for one artifact, on adjacent screens.
    """
    figure = "{:.2f} GiB".format(download_bytes / (1024 ** 3))
    return (
        "about {}".format(figure) if source == SIZE_MEASURED
        else "up to about {}".format(figure)
    )


def _entry(
    identifier: str,
    name: str,
    parameter_size: str,
    parameter_billions: float,
    repo: str,
    file: str,
    download_bytes: int,
    sha256: str,
    size_source: str,
    experience: str,
    quantization: str = "Q4_K_M",
    runtime: Optional[Dict[str, Any]] = None,
    engine: str = "",
    effective_billions: Optional[float] = None,
    surface: str = "assistant",
    source_url: str = "",
    release_tag: str = "",
) -> Dict[str, Any]:
    return {
        "id": identifier,
        "name": name,
        "parameter_size": parameter_size,
        "parameter_billions": parameter_billions,
        # **The size that governs how demanding this model is to RUN, which for
        # a mixture-of-experts model is not its nominal parameter count.** A
        # dense model activates all of its weights every token, so nominal
        # params predict both its footprint and its compute; a MoE activates a
        # small fraction (Qwen3-30B-A3B routes ~3B of 30B, gpt-oss-20b ~3.6B of
        # ~21B), so its *speed* tracks the active count while its *footprint*
        # still tracks the whole. The RAM tier ladder keyed on nominal
        # `parameter_billions` hid these on accelerated boxes whose memory could
        # hold them (LESSONS 12/5): a ~17 GiB MoE was excluded from the 18-28 GiB
        # band because 30 > the band's 14B ceiling, though it fits by bytes. So
        # surfacing keys on this effective size — the ladder judges "is this too
        # heavy to be worth recommending" by what the model actually computes —
        # while the honest deploy-time byte fit (`local_runtime_settings`,
        # `_storage_verdict`, `model_discovery._fit`) still gates FOOTPRINT and
        # refuses a model the machine cannot physically hold. Defaults to
        # `parameter_billions`, so every dense entry tiers exactly as before.
        "effective_billions": (
            float(effective_billions)
            if effective_billions is not None
            else float(parameter_billions)
        ),
        # Which serving surface this model is for — the Assistant, or AI Chat.
        # An `"ai-chat"` entry is served on the GPU as the AI-Chat model, not as
        # the Assistant: :meth:`ExecutorModelDeployMixin._deploy_model` reads
        # this (via :func:`catalog_surface`) to run the GPU compose path and
        # claim the ``ai-chat`` credential through
        # :func:`vaelor.managed_local_credentials.activate_managed_chat` instead
        # of ``deployment-agent``, and the ``"ai-chat"`` surface is exempt from
        # the NPU gate (`flm_supervisor.should_serve_on_npu`) so it is never
        # re-served on the neural processor. It is DATA carried by the entry, not
        # per-model behaviour in the deployer: any stock GGUF marked this way
        # takes the same generic route. Defaults to ``"assistant"``.
        "surface": surface,
        # Which inference engine this entry *requires* to run at all. The empty
        # string is the normal path - stock llama.cpp / flm - and every entry
        # that predates GPU AI-Chat keeps it. A non-empty engine (e.g.
        # ``rocmfpx``) marks a model whose tensors only load in a specific fork,
        # and it is load-bearing in two places: :func:`catalog_engine` tells the
        # deployer which binary to launch, and :func:`largest_within` refuses to
        # surface an engine-gated entry on the generic RAM ladder, so a model
        # that needs gfx1151+ROCmFP4 is never recommended to a machine that
        # cannot run it (the phantom-model defect in a new direction).
        "engine": engine,
        # A fine-tuned NPU model is not a Hugging Face GGUF: it is delivered as a
        # release the appliance downloads and unpacks (model + its bundled
        # flm-real runtime). `source_url` is the pinned release base the deploy
        # verifies against `sha256`, and `release_tag` is the flm tag it serves.
        # Empty on every stock GGUF entry, which is downloaded and served the
        # existing way; `catalog_release_for_tag` reads them to route an install
        # through `model.install_release` instead of model.download/deploy.
        "source_url": source_url,
        "release_tag": release_tag,
        # Not always Q4_K_M since VD-076. The Pi's A76 is ARMv8.2 with
        # `asimddp` and no `i8mm`; `Q4_0` and `IQ4_NL` repack into the ARM
        # dot-product kernel and K-quants do not, which is worth +82% prefill
        # on this hardware - more than the difference between two models.
        "quantization": quantization,
        "repo": repo,
        "file": file,
        "download_bytes": download_bytes,
        # The content address of the artifact this entry claims to be about.
        # Required, because it is the only field of the four that a wrong
        # publisher cannot accidentally satisfy: two publishers' Q4_K_M builds
        # of the same model share a filename and sit 1,056 bytes apart.
        "sha256": sha256,
        "size_source": size_source,
        "size_note": _size_note(download_bytes, size_source),
        # Kept so the existing search-and-verify download path is unchanged;
        # it is built from the repository this entry names rather than written
        # separately, so the two cannot drift.
        "search_query": "{} {}".format(repo, quantization),
        "experience": experience,
        # **The parameters a deployment needs, travelling with the model that
        # needs them.** Written here because a clean install has no session
        # transcript: without this the 4,096 window, the quantisation and the
        # KV-restore path are tribal knowledge, and the next person to set up a
        # box gets llama.cpp's defaults and none of what VD-076 measured.
        "runtime": dict(runtime or {}),
    }


#: The deployment parameters VD-076 measured, kept beside the model that needs
#: them so a clean install inherits them instead of llama.cpp's defaults.
#:
#: ``context`` 4,096 rather than 8,192. The 8,192 window came from
#: ``AGENT_PROMPT_TOKENS = 4668``, roughly six times the real 757-token prompt;
#: its KV cache cost ~460 MiB the Pi did not have and stalled the machine hard
#: enough to need a physical power cycle. At 4,096 the same model scored 93.0%
#: with 1,254 MiB to spare (VD-075).
#:
#: ``slot_save`` turns on ``--slot-save-path``, which is the only way the
#: standing brief survives ``--sleep-idle-seconds``. Measured on the appliance:
#: the first question after an idle period costs 95.5s without it and 20.4s
#: with it, because waking otherwise pays the model reload *and* a full
#: 786-token prefill. The orchestration lives in ``assistant_slot_cache``
#: (task #152): a save where the reply is appended, and a restore gated on the
#: brief's sha256 and the conversation's last message - a stale prefix must
#: not answer for a machine that has changed.
#: ``cache_ram_mib`` bounds llama-server's RAM prompt cache. **Its default is
#: 8192 MiB and this appliance has 7,931 MB of RAM in total**, so the engine
#: was permitted a cache larger than the whole machine: it kept a KV prefix per
#: distinct prompt and never evicted, because eviction needs a ceiling the
#: hardware can reach. Measured over 80 prompts on this model at this window,
#: the default never plateaued - anonymous memory rose ~13 MB per prompt with
#: the cgroup at 87% and climbing. Bounded, it flattens at prompt 20 and holds,
#: and it is *faster*: 28.4 s per answer against 30.3 s (VD-080).
#:
#: 256 is what was measured. It is a **ceiling, not a floor** - the deploy path
#: also derives a bound from the container's own limit and takes whichever is
#: smaller, so this number can never put a cache on a machine too small to hold
#: it. On the Pi's 4,352 MiB limit the derived 217 wins, and 217 is the value
#: that was deployed and re-run.
#: **`prompt_state_mib` is deliberately absent, and the reason is worth
#: keeping.** llama-server logged "prompt state size 288.024 MiB exceeds cache
#: size limit 217.000 MiB, skipping" on 2026-08-11, so the RAM prompt cache is
#: inert here and every request pays a full prefill. The first fix set a floor
#: of `288` - which is **25 KiB below the 288.024 MiB state it has to hold**,
#: so it would have reserved 71 MiB more and still skipped, in a cgroup with
#: three recorded OOM kills behind it (VD-070, #117, #108).
#:
#: The deeper error was treating one log line as a bound. A prompt state is
#: proportional to the *prompt's* token count, not fixed per model and window,
#: so 288.024 MiB is one observation of one conversation - a longer one
#: produces a larger state and skips again. A floor has to come from the
#: largest state this window can produce, which nobody has measured.
#:
#: So the bound stays where VD-080 measured it and #157 stays open with what
#: is now known. Reserving memory for a cache that still cannot hold anything
#: is the one outcome worse than the inert cache we have.
RUNTIME_PI_ASSISTANT: Dict[str, Any] = {
    "context": 4096,
    "slot_save": True,
    "cache_ram_mib": 256,
}


#: The launch parameters measured by hand on the Z2 (Strix Halo / Radeon 8060S,
#: gfx1151) for the ROCmFP4 27B, kept beside the model so the GPU deployer
#: inherits them instead of re-deriving the recipe. Same discipline as
#: ``RUNTIME_PI_ASSISTANT``: a clean install has no session transcript, and
#: without this the proven ``llama-server`` invocation is tribal knowledge.
#:
#: These are the values that were run and measured at ~37 t/s, not defaults:
#: ``device`` ``Vulkan0`` selects the Radeon adapter the ROCmFPX fork serves
#: through; ``ngl`` 99 offloads every layer to it; ``context`` 131,072 is the
#: window that fit in the shared aperture; ``kv_k`` ``q8_0`` / ``kv_v``
#: ``turbo4`` are the KV-cache quantisations the fork accepts for FP4 tensors;
#: and the ``spec_*`` fields turn on draft-MTP speculative decoding
#: (``--spec-type draft-mtp``, up to 4 drafted tokens, no probability floor),
#: which is where most of the throughput came from. Unlike the NPU's flm-real
#: this is an *unprivileged* host process - no bridge, no CAP_IPC_LOCK - so the
#: parameters travel to a plain ``subprocess.Popen`` launcher, not a root one.
RUNTIME_Z2_GPU_ROCMFP4: Dict[str, Any] = {
    "context": 131072,
    "device": "Vulkan0",
    "ngl": 99,
    "flash_attn": True,
    "kv_k": "q8_0",
    "kv_v": "turbo4",
    "spec_type": "draft-mtp",
    "spec_draft_n_max": 4,
    "spec_draft_p_min": 0.0,
}


#: Every local model this appliance can install, smallest first.
#:
#: An entry earns its place by having a repository and a file that the download
#: path can resolve. ``search_query`` is derived from them; it is not an
#: alternative to them. The tiers this ladder used to reach above 4B - 8B, 14B,
#: 32B, 70B - had a search string and nothing else, which is why one of them
#: could be recommended by name on a machine that then installed something else.
LOCAL_MODEL_CATALOG: Tuple[Dict[str, Any], ...] = (
    _entry(
        "qwen3-1.7b-q4", "Qwen3 1.7B", "1.7B", 1.7,
        # unsloth, not Qwen. `Qwen/Qwen3-1.7B-GGUF` publishes only a Q8_0, and
        # the 1,107,409,472 bytes this appliance downloaded on the Z2 are
        # unsloth's file to the byte.
        "unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf",
        # Read off the file this appliance actually downloaded on the Z2.
        1_107_409_472,
        "b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897",
        SIZE_MEASURED,
        "Fast, private everyday help with plenty of memory left for apps.",
    ),
    _entry(
        "qwen3-4b-q4", "Qwen3 4B", "4B", 4.0,
        "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf",
        # The artifact, not a bound over it. `Qwen/Qwen3-4B-GGUF` publishes a
        # file of the same name at 2,497,280,256 bytes; this is unsloth's, and
        # the digest is what tells the two apart.
        2_497_281_312,
        "f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a",
        SIZE_MEASURED,
        # #147/VD-108: earlier copy here made a speed claim ("slower to answer
        # after an idle period") that was measured only on the Pi's ARM kernel
        # (VD-076) and is false on x86, where the smaller 1.7B reloads fastest.
        # The genuine, hardware-neutral distinction is the variant: this is the
        # standard Qwen3 4B that thinks before answering, versus the
        # instruction-tuned 2507 sibling that answers directly.
        "Close in size to the recommended build; works through a reasoning "
        "step before it answers.",
    ),
    _entry(
        "qwen3-4b-instruct-2507-q4-0", "Qwen3 4B Instruct", "4B", 4.0,
        "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "Qwen3-4B-Instruct-2507-Q4_0.gguf",
        # Downloaded and verified 2026-08-10; the digest is this file, not a
        # same-named build from another publisher.
        2_375_773_280,
        "e0ba675d86ab277c61701c6793659b2ae801d95e3be791464c321e6fbf613be2",
        SIZE_MEASURED,
        "Instruction-tuned to answer directly, with no reasoning step.",
        # `Q4_0`, not `Q4_K_M`. VD-076: on the Pi's CPU it repacks into the ARM
        # dot-product kernel for +82% prefill (prefill is 68% of an answer) - a
        # Pi-only win, so it stays here in the reason for the quant and never in
        # the user string, which renders on x86 too (VD-108).
        quantization="Q4_0",
        runtime=RUNTIME_PI_ASSISTANT,
    ),
    _entry(
        "qwen3-14b-q4", "Qwen3 14B", "14B", 14.0,
        "unsloth/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf",
        # Byte-exact off the HF release; resolved and matched size + sha256
        # against unsloth by tools/verify_model_catalog.py (2026-08-21). The
        # download gate compares `!=` on both, so an estimate here would fail it.
        9_001_753_984,
        "5eaa0870bd81ed3b58a630a271234cfa604e43ffb3a19cd68e54a80dd9d52a66",
        SIZE_MEASURED,
        # Hardware-neutral, no machine name and no cross-hardware speed claim
        # (ExperienceCopyIsTrueOnEveryMachineTests): the true distinction is that
        # it is a larger dense model for stronger answers when memory allows.
        "A larger dense model for stronger, more detailed answers when there is "
        "memory to spare.",
        # Dense 14B: stock llama.cpp, served on the GPU as the AI-Chat model
        # through the generic stock-GGUF compose route (surface `ai-chat`), not
        # a special fork - so `engine` stays empty and it rides the normal RAM
        # ladder. A dense model, so its effective size is its nominal size.
        surface="ai-chat",
    ),
    _entry(
        "gpt-oss-20b-mxfp4", "gpt-oss 20B", "20B", 20.9,
        "ggml-org/gpt-oss-20b-GGUF",
        # Uppercase `MXFP4`, case-sensitive: the registry publishes this exact
        # spelling and a lowercase name does not resolve.
        "gpt-oss-20b-MXFP4.gguf",
        12_109_566_624,
        "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",
        SIZE_MEASURED,
        "An open mixture-of-experts model that answers directly, activating a "
        "small share of its weights for each word so it stays responsive.",
        # MXFP4, not a K-quant: gpt-oss ships its own 4-bit MoE tensor layout.
        # Stock llama.cpp reads it (a recent build - see the pinned accelerated
        # image digest in model_service_compose), so `engine` stays empty and it
        # takes the generic GPU compose route rather than a fork.
        quantization="MXFP4",
        # Mixture-of-experts: ~21B nominal, ~3.6B active per token. Tiered by the
        # active size (so an 18-28 GiB box that can hold its ~11.3 GiB footprint
        # is offered it), while the byte fit still gates the footprint.
        effective_billions=3.6,
        surface="ai-chat",
    ),
    _entry(
        "qwen3-8-27b-rocmfp4-fast", "Qwen 3.8 27B", "27B", 27.0,
        "julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF",
        "Qwen3.8-27B-ROCmFP4-FAST.gguf",
        # Exact byte count off the HF release; the download gate compares `!=`
        # against what the registry reports, so an estimate here would fail it.
        14_562_236_384,
        # The LFS content oid the registry serves as the artifact's digest.
        "fb89c78d2be91cdb68eaaaa45b1270710bf34aa721dc1f0b9e3aa7b98d2e1da9",
        SIZE_MEASURED,
        "Runs on this machine's GPU for markedly stronger answers; about "
        "13.6 GiB, served on the Radeon graphics.",
        # ROCmFP4, not a K-quant: an FP4 tensor layout that only loads in the
        # ROCmFPX llama.cpp fork. The distinction is not cosmetic - stock
        # llama.cpp cannot read these tensors - so it is what `catalog_engine`
        # keys the `rocmfpx` engine on.
        quantization="ROCmFP4",
        runtime=RUNTIME_Z2_GPU_ROCMFP4,
        # The one FORK-gated entry. `rocmfpx` tells the GPU deployer to launch
        # the fork binary, and it also keeps this model *out* of the generic RAM
        # ladder (see `largest_within`): a 27B FP4 build is only runnable on a
        # gfx1151 GPU with the fork, never a plain accelerator. The stock-GGUF
        # AI-Chat entries below carry NO engine because stock llama.cpp runs
        # them, so they DO ride the ladder - the surface, not the engine, routes
        # them to the GPU AI-Chat path.
        engine="rocmfpx",
        # This is the AI-Chat model, not the Assistant. Recorded for truth; the
        # routing is by `engine` (above) which is checked before `surface`, so
        # the field is documentation here rather than the discriminator.
        surface="ai-chat",
    ),
    _entry(
        "qwen3-30b-a3b-instruct-2507-q4", "Qwen3 30B A3B", "30B-A3B", 30.5,
        "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
        18_556_686_752,
        "6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0",
        SIZE_MEASURED,
        "A large mixture-of-experts model that answers directly, activating a "
        "small share of its weights for each word so it stays responsive.",
        # Stock llama.cpp K-quant (Q4_K_M), served on the GPU as the AI-Chat
        # model through the generic compose route - no fork, so `engine` stays
        # empty and it rides the ladder.
        # Mixture-of-experts: ~30.5B nominal, ~3.3B active per token. Its ~17 GiB
        # footprint fits a machine in the 18-28 GiB band that the nominal-30
        # ladder ceiling excluded (LESSONS 12/5); tiered by the active size, with
        # the honest byte fit gating the footprint.
        effective_billions=3.3,
        surface="ai-chat",
    ),
)

#: Fine-tuned on-device NPU models, delivered as pinned releases rather than
#: Hugging Face GGUFs. Kept SEPARATE from LOCAL_MODEL_CATALOG on purpose: that
#: catalog's invariant is that every entry resolves to a real HF repo/file on the
#: RAM ladder, and a release-sourced NPU model satisfies neither - it installs
#: through the bridge and runs on the neural processor, not system RAM. It is
#: surfaced by `copilot_setup` as the recommended Assistant on a machine with an
#: NPU, and installed through `model.install_release`, never model.download.
NPU_RELEASE_MODELS: Tuple[Dict[str, Any], ...] = (
    _entry(
        "qwen3-5-4b-ff-4b-h-npu", "Qwen3.5-4B on-device (NPU)", "4B", 4.0,
        # Synthetic repo/file: it names the artifact in the managed layout, but
        # the real source is `source_url`. engine="flm-npu" + `release_tag` are
        # what route the install through the bridge and serve it as that flm tag.
        "vaelor/Qwen3.5-4B-ff-4b-h-NPU", "ff-4b-h-npu-v1.tar",
        3_638_517_760,
        "71b21c7407c5c0ac15e5f45690f2945433f00e4d5f2b51d77178e3edbd54f19f",
        SIZE_MEASURED,
        "Runs on this machine's neural processor - the fastest, most private "
        "on-device Assistant; about 3.4 GiB.",
        quantization="Q4NX",
        engine="flm-npu",
        surface="assistant",
        source_url=NPU_RELEASE_SOURCE,
        release_tag="qwen3.5:4b",
    ),
)


def catalog_runtime(repo: str, file: str) -> Dict[str, Any]:
    """The measured runtime parameters for one artifact, or an empty dict.

    Matched on repository *and* file, because two publishers ship a
    `Qwen3-4B-Q4_K_M.gguf` 1,056 bytes apart and only the pair identifies which
    one a machine is holding.

    Returns a copy: a caller that mutates what it gets back must not be able to
    edit the catalog, which is the same reason `catalog_models` copies.
    """
    for item in LOCAL_MODEL_CATALOG:
        if item["repo"] == repo and item["file"] == file:
            return dict(item.get("runtime") or {})
    return {}


def catalog_engine(repo: str, file: str) -> str:
    """Which inference engine one artifact requires, or ``""`` for the stock path.

    Matched on repository *and* file, the same pair
    :func:`catalog_runtime` uses, because that pair is what identifies which
    exact artifact a machine is holding. Returns the entry's ``engine`` string -
    ``"rocmfpx"`` for the ROCmFP4 27B, ``""`` for every stock-llama/flm entry
    and for a pair the catalog does not contain. The deployer reads this to
    decide whether to launch the ROCmFPX fork rather than stock llama.cpp.

    A plain string, so there is nothing to copy defensively; the discipline
    mirrors ``catalog_runtime`` returning a fresh dict.
    """
    for item in LOCAL_MODEL_CATALOG:
        if item["repo"] == repo and item["file"] == file:
            return str(item.get("engine") or "")
    return ""


def catalog_surface(repo: str, file: str) -> str:
    """Which serving surface one artifact is for, or ``""`` for a pair we do not stock.

    Matched on repository *and* file, the pair :func:`catalog_runtime` and
    :func:`catalog_engine` use, because that pair is what identifies which exact
    artifact a machine is holding. Returns ``"ai-chat"`` for an entry served on
    the GPU as the AI-Chat model, ``"assistant"`` for the Assistant GGUFs, and
    ``""`` for a pair the catalog does not contain (so the deployer falls back to
    the payload's own surface, defaulting to ``"assistant"`` exactly as before).

    The deployer reads this to route a stock catalog GGUF to the GPU AI-Chat
    compose path and claim the ``ai-chat`` credential, without any per-model
    branch: the entry carries the fact, the route is generic.
    """
    for item in LOCAL_MODEL_CATALOG:
        if item["repo"] == repo and item["file"] == file:
            return str(item.get("surface") or "")
    return ""


CATALOG_IDS = tuple(item["id"] for item in LOCAL_MODEL_CATALOG)
CATALOG_NAMES = tuple(item["name"] for item in LOCAL_MODEL_CATALOG)


def catalog_models() -> Tuple[Dict[str, Any], ...]:
    """Copies, so a caller mutating a recommendation cannot edit the catalog."""
    return tuple(dict(item) for item in LOCAL_MODEL_CATALOG)


def catalog_model(identifier: str) -> Optional[Dict[str, Any]]:
    for item in LOCAL_MODEL_CATALOG:
        if item["id"] == identifier:
            return dict(item)
    return None


def is_release_model(item: Dict[str, Any]) -> bool:
    """Whether an entry installs from a release (via the bridge) not an HF download."""
    return str(item.get("engine") or "") == "flm-npu" and bool(item.get("source_url"))


def npu_release_models() -> Tuple[Dict[str, Any], ...]:
    """Copies of the release-sourced NPU model entries, for the setup screen."""
    return tuple(dict(item) for item in NPU_RELEASE_MODELS)


def catalog_release_for_tag(release_tag: str) -> Optional[Dict[str, Any]]:
    """The pinned release source for a fine-tuned NPU model, matched by its flm tag.

    Returns the fields the deploy needs to install it - the release base URL, the
    sha256 to verify the artifact against, and the tag to serve - or ``None`` for
    a tag no release entry stocks. The source and digest are read HERE, from the
    catalog shipped in the wheel, so a ``model.install_release`` job need only
    carry the tag: the trust anchor is the code, never the job payload.
    """
    for item in NPU_RELEASE_MODELS:
        if is_release_model(item) and str(item.get("release_tag")) == str(release_tag):
            return {
                "id": item["id"],
                "name": item["name"],
                "tag": str(item["release_tag"]),
                "source_url": str(item.get("source_url") or ""),
                "sha256": str(item.get("sha256") or ""),
            }
    return None


def _tier_billions(tier: str) -> float:
    """The parameter count a tier name stands for, as a number.

    The ladder speaks in strings - ``"1.7B"``, ``"32B"`` - and a string cannot
    be compared to a model's size. That is the join that was missing: the tier
    was looked up in a table of names instead of being used as a *ceiling* over
    models that exist.
    """
    text = str(tier or "").strip().upper().rstrip("B")
    try:
        return float(text)
    except ValueError:
        return 0.0


def largest_within(tier: str) -> Dict[str, Any]:
    """The biggest catalog model this hardware tier can hold.

    Never empty and never a name from outside the catalog: the smallest entry
    is returned when the tier is below all of them, because refusing to
    recommend anything on a machine that can run the smallest model would be a
    worse answer than recommending it.

    **A tie is broken by catalog order, and that turned out to be
    load-bearing.** Two entries sit at 4.0B, so the later one wins on every
    tier at or above it. Appending `qwen3-4b-instruct-2507-q4-0` therefore
    moved *every* machine from 4B upwards onto it, x86 workstations included -
    and its `Q4_0` was chosen for the Pi's A76, which is ARMv8.2 with
    `asimddp` and no `i8mm`, so `Q4_0` and `IQ4_NL` repack into the ARM
    dot-product kernel. An ARM CPU fact became the default on a machine with a
    GPU, unmeasured, because of a list position.

    `tests/test_model_catalog.py` now pins what each tier resolves to, so the
    next append fails there and the choice is made rather than inherited.
    Which model an accelerated x86 machine *should* recommend is task #81 and
    wants its own measurements, not a tie-break rule invented here.

    **Engine-gated entries are excluded from this ladder.** An entry that
    declares an ``engine`` (the ROCmFP4 27B, which only loads in the ROCmFPX
    fork on a gfx1151 GPU) must not leak into the generic RAM-tier ladder, or
    every machine with a large enough memory budget - including accelerated x86
    boxes and non-gfx1151 GPUs that cannot run ROCmFP4 at all - would be
    recommended a model it cannot serve. That is the phantom-model defect this
    module exists to prevent, pointed in a new direction. The 27B is surfaced
    *only* through a capability-gated GPU override built in a later phase, never
    this plain ladder, so it is filtered out here.
    """
    ceiling = _tier_billions(tier)
    fitting = [
        item for item in LOCAL_MODEL_CATALOG
        if item["parameter_billions"] <= ceiling
        and not item.get("engine")
    ]
    return dict((fitting or [_smallest_non_engine_entry()])[-1])


def _non_engine_entries() -> Tuple[Dict[str, Any], ...]:
    """The catalog entries the generic ladder is allowed to recommend.

    Engine-gated entries are excluded here for the same reason
    :func:`largest_within` excludes them: they are not part of the plain
    RAM-tier ladder and must not stand in for its top or bottom. Never empty in
    practice - every stock-llama entry qualifies - but the callers below fall
    back to the whole catalog if it ever were, rather than raising.
    """
    return tuple(item for item in LOCAL_MODEL_CATALOG if not item.get("engine"))


def _smallest_non_engine_entry() -> Dict[str, Any]:
    entries = _non_engine_entries() or LOCAL_MODEL_CATALOG
    return entries[0]


def _largest_non_engine_entry() -> Dict[str, Any]:
    """The biggest entry the generic ladder can reach, by catalog order.

    ``exceeds_catalog`` and ``headroom_note`` reference *this* rather than
    ``LOCAL_MODEL_CATALOG[-1]``, because appending the engine-gated 27B made the
    last entry a model the generic ladder never offers. Keeping the headroom
    banner pinned to the largest *non-engine* entry is what keeps "the catalog
    stops at Qwen3 4B Instruct" a true statement about the ladder a machine
    without the fork actually sees.
    """
    entries = _non_engine_entries() or LOCAL_MODEL_CATALOG
    return entries[-1]


def exceeds_catalog(tier: str) -> bool:
    """Whether this machine could hold more than the *generic* catalog offers.

    Measured against the largest non-engine entry, not ``[-1]``: the 27B is
    engine-gated and never on the generic ladder, so a machine sitting between
    the 4B ladder-top and the 27B still has headroom worth stating.
    """
    return (
        _tier_billions(tier)
        > _largest_non_engine_entry()["parameter_billions"]
    )


def headroom_note(tier: str) -> str:
    """Say that the machine outruns the catalog, without naming a model.

    The previous answer to this situation was to name a tier - "Vaelor
    recommends 32B" - which is a claim about a model that could not be
    installed from here. This states the same fact about the *hardware* and
    sends the reader to the search box, which can resolve a real file.
    """
    if not exceeds_catalog(tier):
        return ""
    return (
        "This machine could run a larger model than Vaelor's reviewed catalog "
        "offers - its memory budget reaches the {} class, and the catalog "
        "stops at {}. Search Hugging Face from this screen to choose one; "
        "Vaelor checks format, memory and free space before any download "
        "starts."
    ).format(tier, _largest_non_engine_entry()["name"])
