"""The context windows the appliance's own prompt needs, and what they cost."""

from __future__ import annotations



#: Size of the appliance's own standing prompt, in tokens: the largest single
#: path, plus the machine brief that rides on all of them.
#:
#: **Measured 2026-08-10 on the appliance, with the deployed model's own
#: /tokenize.** Every system prompt Vaelor sends, each rendered the way its
#: runner renders it, then counted with the 545-token brief appended:
#:
#:     CUSTOM_AGENT_PROMPT        799 alone   1344 with brief   <- largest
#:     LOCAL_CUSTOM_AGENT_PROMPT  725         1270
#:     ASSISTANT_PROMPT           273          818
#:     SYSTEM_PROMPT              222          767
#:     LOCAL_PLANNER_PROMPT       169          714
#:     SPECIALIST_PROMPT          126          671
#:     LOCAL_ASSISTANT_PROMPT     119          664
#:     LOCAL_SPECIALIST_PROMPT    102          647
#:
#: **A maximum, not a sum.** One system prompt goes out per request and the
#: paths are mutually exclusive; adding them would describe a request that is
#: never made. The custom-agent figure already carries a deliberately
#: oversized 600-word instruction block, so it is a ceiling on Vaelor's own
#: text rather than a typical case - a real one is nearer 700.
#:
#: It replaces **4668**, which nothing in the tree derived and which was 3.5x
#: the worst measured path. That number gated the `below-brief` warning in
#: `accelerator_runtime`, so a 4,096-token window - the one that scored 93.0%
#: over 200 items - was reported to the owner as unable to hold the
#: appliance's own instructions. It also produced the 8,192 window whose KV
#: cache cost ~460 MiB the Pi did not have and stalled the machine (VD-075).
#:
#: `instructions` on a custom agent is owner-written and unbounded, so a long
#: enough one can still exceed this. That is a property of the request, not of
#: Vaelor's standing prompt, and belongs to whatever bounds that field - not
#: here. `tests/test_inference_context.py` fails if this drifts from the
#: measurement recorded above.
AGENT_PROMPT_TOKENS = 1344

#: The context window the benchmarks recommend **for the local CPU tier** - the
#: Pi's Assistant, and the managed local model on any machine without an
#: accelerator. Still the smallest power of two that holds
#: :data:`AGENT_PROMPT_TOKENS` with room for a conversation - the derivation has
#: not changed, the number it derives from has. 1,344 standing leaves 2,752
#: tokens of a 4,096 window for the question and the answer.
#:
#: **It is not the NPU's window and it is not the GPU's.** Until 2026-08-10 both
#: of those tiers took their default from this constant, so lowering the Pi's
#: window from a Pi measurement silently moved the Z2's NPU window to 4,096 -
#: a window nothing on that tier has ever been measured at, which made its whole
#: KV saving unreportable. Those tiers now carry
#: :data:`NPU_RECOMMENDED_CONTEXT_TOKENS` and
#: :data:`GPU_RECOMMENDED_CONTEXT_TOKENS`, each set from its own hardware's
#: measurements. A number measured on one machine must not reach across to
#: another by sharing a name.
#:
#: **And 4,096 is the window that was actually measured.** Qwen3 4B completed
#: all 200 evaluation items at 93.0% here with 1,254 MiB to spare, and scored
#: identically (41/43) to the 8,192 run on the items both covered - the larger
#: window bought nothing and cost ~460 MiB the Pi did not have (VD-075).
#:
#: Was 8192, from `AGENT_PROMPT_TOKENS = 4668`.
RECOMMENDED_CONTEXT_TOKENS = 4096

#: The runtime mode a deploy asks for when the request names none. ``balanced``
#: was the old default and its window is 4096 tokens — smaller than the agent
#: prompt above — and because the sizing selector only ever walks *down* from
#: what it is asked for, asking for the middle setting made the recommended
#: window unreachable on every machine regardless of how much memory it had.
RECOMMENDED_RUNTIME_MODE = "recommended"

#: The flag ``flm-real`` takes for its context window.
#:
#: **Not in ``flm serve --help``.** It was read off Lemonade's own invocation in
#: the live process table - `flm-real serve llama3.2:1b --ctx-len 131072 --port
#: 8001 --host 127.0.0.1 --quiet` - and Lemonade additionally refuses it from
#: user-supplied `flm_args` ("flm_args flag is not allowed: --ctx-len"), which
#: is what a wrapper does with a flag it is pinning for itself.
#:
#: The distinction matters: llama.cpp's equivalent is ``--ctx-size`` and the GPU
#: tier uses that. Sending llama.cpp's spelling to `flm-real` would silently get
#: the default window back, which is the whole 17.4 GB this exists to avoid.
NPU_CONTEXT_FLAG = "--ctx-len"

#: FLM's native window when :data:`NPU_CONTEXT_FLAG` is not passed. It
#: preallocates KV for all of it - and Lemonade pins it here, so a Lemonade-
#: driven NPU tier cannot recover the saving at all.
NPU_NATIVE_CONTEXT_TOKENS = 131072

#: Measured KV footprint of a 3 B on the NPU tier, by context window. The
#: difference is the whole reason the context flag must be set.
#: **Keyed on the window each figure was measured at, not on whatever
#: `RECOMMENDED_CONTEXT_TOKENS` happens to be.** The 4.0 GB was measured at
#: 8,192; while the constant also equalled 8,192 the distinction did not show,
#: and moving the recommendation to 4,096 would have relabelled an 8,192
#: measurement as a 4,096 one without anybody touching the number. There is no
#: measurement at 4,096 on this tier - `footprint_for` says so rather than
#: interpolating one.
#:
#: **16,384 was measured on 2026-08-22 (VD-110), qwen3.5:4b, RSS 6.0 GB / peak
#: 6.5 GB**, after the 8,192 window was found to reject any agent loop whose
#: accumulated tool responses cross it: FLM returns HTTP 400 on a prompt past
#: its `--ctx-len`, which surfaced as empty and degraded multi-step runs. The
#: NPU loop's worst case is ~10-11 k tokens (5 iterations x 3 tool calls x
#: 2,400 chars plus the system and user turns), so 16,384 clears it with
#: margin, and the Z2 carries it with ~20 GB to spare.
NPU_CONTEXT_FOOTPRINT_BYTES = {
    NPU_NATIVE_CONTEXT_TOKENS: int(17.4 * 1_000_000_000),
    8192: int(4.0 * 1_000_000_000),
    16384: int(6.0 * 1_000_000_000),
}

#: The NPU tier's own window. **The NPU is not on the Pi**, so the Pi's
#: :data:`RECOMMENDED_CONTEXT_TOKENS` has no authority over it: that tier is the
#: Z2's, with its own memory budget and its own measurements.
#:
#: 16,384 is a measured window in :data:`NPU_CONTEXT_FOOTPRINT_BYTES` (VD-110),
#: so `footprint_for` can still say what the flag saves - 11.4 GB against the
#: 131,072-token native window. It was raised from 8,192, which was measured and
#: safe on memory but too small for the agent loop: FLM 400s a prompt that
#: crosses `--ctx-len`, so a multi-step run that accumulated enough tool
#: responses was rejected outright rather than answered. Choosing an unmeasured
#: window would leave `footprint_for` with no figure - which is the reason the
#: flag is set - so any future change must add its measured row here first.
NPU_RECOMMENDED_CONTEXT_TOKENS = 16384

#: The GPU tier's own window, likewise not the Pi's. It is the window the
#: measured AI Chat configuration was benchmarked at, alongside `-ctk q8_0
#: -ctv q8_0 -fa auto -ub 1024`; changing it detaches the window from the flags
#: that were measured with it.
GPU_RECOMMENDED_CONTEXT_TOKENS = 8192
