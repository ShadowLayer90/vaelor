"""The reviewed GPU serving presets: what vLLM pulls, and the geometry it is sized by.

Housed out of `gpu_pool_operations` (at the 1,000-line ceiling `CLAUDE.md`
sets) and read by it, by `gpu_model_library` (a pull by catalog id resolves its
repo here) and by the frontend's mirror `frontend/src/lib/gpuCatalog.ts`, which
`tests/test_gpu_catalog_frontend_parity.py` holds byte-for-byte to this table.

Each entry carries the weights repo vLLM pulls and the transformer geometry the
fit engine sizes against - exactly the fields `cluster_gpu_sizing.build_model_spec`
validates, named by :data:`SPEC_FIELDS`. A pasted Hugging Face link supplies its
own geometry through ``payload["model_spec"]`` instead; the catalog is the
curated shortcut, not the only path.

**``weight_bytes`` is MEASURED, never estimated.** It is the resident weight
size at the stated precision, read off the repository itself: the sum of the
safetensors shard FILES the Hub tree lists (``/api/models/<repo>/tree/main``),
as integers. That is the one measurement every entry uses - the repo's
``model.safetensors.index.json`` ``total_size`` is a different figure that
reads a few KiB to a few MiB either side of the file sum, so it is named in a
comment where it was looked at and never written as the value. The fit
decision therefore never falls back to `cluster_gpu_sizing.BYTES_PER_PARAMETER`,
the nominal bytes-per-parameter table that exists for a spec with no measured
size. `build_model_spec` reads a supplied ``weight_bytes`` first and derives
from ``parameter_billions`` only when it is absent; an entry here always
supplies it, and `tests/test_gpu_model_catalog.py` pins EVERY entry to the
shard sizes it was read from, so a hand-edited "round" number goes red rather
than sizing a deploy.

**``quantization`` names the WEIGHT format.** The KV cache and compute dtype
are not a per-entry fact: the engine's KV formula (`KV_CACHE_FORMULA`) counts
two bytes per element, which is the bf16/f16 cache every entry here runs with -
the FP8 entry included, whose weights are fp8 but whose activations and cache
are bf16. A model whose cache were narrower than that would need a field this
schema does not have, and the honest thing would then be to add one, not to
write a precision the cache does not use.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .hf_model_source import resolve_model_source

#: The preset the deploy form offers first and marks as recommended (owner
#: decision, 2026-09-30, from the vLLM 0.27 benchmark on the pair).
RECOMMENDED_CATALOG_ID = "qwen3_30b_a3b_instruct_2507_w4a16"

#: The reviewed presets, keyed by the id a deploy submits as ``model_source``.
#: The recommended one is first: the form's picker opens on the first entry.
GPU_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    RECOMMENDED_CATALOG_ID: {
        # The recommended cluster model: a 30.5B mixture-of-experts model
        # (128 experts, about 3.3B parameters active per token) in 4-bit
        # compressed-tensors weights (W4A16, groups of 32), which vLLM 0.27
        # runs on the RDNA 4-bit kernel with its tuned MoE table
        # (`vllm_entry_program`). Measured on one machine on 2026-09-30, at a
        # 44 GiB GPU memory pool and 0.80 of it, 16k context: 47.9 tokens a
        # second for one user, 178.7 across 16 users, 5.4 s to the first word
        # of an 8k prompt, peaking at about 36.1 GiB of GPU memory (38.7 GB).
        #
        # It is an Instruct-2507 release: hermes tool calls, NO thinking
        # switch and NO reasoning parser (`vllm_model_profile`) - with the
        # parser and thinking requested its whole answer landed in
        # ``reasoning`` and ``content`` was null.
        #
        # Measured off the Hub tree at commit 84455dc6 on 2026-09-30: four
        # safetensors shards (`tests/test_gpu_model_catalog.py` pins the four
        # figures). The geometry is the repo's ``config.json``: 48 layers, 32
        # heads over 4 KV heads of width 128, a 262,144-token window. One
        # machine does not hold that whole window (its KV cache alone is
        # 24 GiB): a one-copy-per-machine deployment sets a smaller Max
        # context, and the split across two machines holds all of it.
        # ``parameter_billions`` is the model card's 30.5B.
        #
        # ``default_intent`` and ``default_max_model_len`` are what the deploy
        # form opens on for this entry (:func:`catalog_defaults`): the owner's
        # layout - one copy on each machine, at a 32,768-token context - so
        # an untouched form previews what was decided rather than a split at
        # a context only a split can hold.
        "name": "Qwen 3 30B-A3B Instruct (4-bit)",
        "repo": "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
        "weight_bytes": 18_094_507_352,
        "hidden_layers": 48,
        "attention_heads": 32,
        "kv_heads": 4,
        "head_dim": 128,
        "context_length": 262144,
        "quantization": "w4a16",
        "parameter_billions": 30.5,
        "default_intent": "throughput",
        "default_max_model_len": 32768,
    },
    "qwen3_8b_fp16": {
        # Measured off the Hub tree on 2026-09-05: five safetensors shards
        # (the test pins the five figures). The entry used to carry a round
        # 16.4 GB that no shard sum produces.
        "name": "Qwen 3 8B (FP16)",
        "repo": "Qwen/Qwen3-8B",
        "weight_bytes": 16_381_516_776,
        "hidden_layers": 36,
        "attention_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "context_length": 32768,
        "quantization": "f16",
        "parameter_billions": 8.2,
    },
    "qwen3_14b_bf16": {
        # The two-node live-proof model: too big for one 30 GiB-GTT appliance,
        # comfortably sharded across two. Measured off the Hub tree on
        # 2026-09-05: eight safetensors shards (the test pins the eight
        # figures). The entry used to carry the index's ``total_size``,
        # 29,536,614,400 - 51,240 bytes under the file sum, which no verdict
        # can turn on, but one measurement for every entry is the rule.
        "name": "Qwen 3 14B (BF16)",
        "repo": "Qwen/Qwen3-14B",
        "weight_bytes": 29_536_665_640,
        "hidden_layers": 40,
        "attention_heads": 40,
        "kv_heads": 8,
        "head_dim": 128,
        "context_length": 40960,
        "quantization": "bf16",
        "parameter_billions": 14.8,
    },
    "qwen3_32b_fp8": {
        # The bigger-than-one-box proof model: about 32 GiB of weights plus
        # a 10 GiB bf16 KV cache at the full window, so it will not fit one
        # ~30 GiB-GTT appliance and shards across two. Measured off the Hub
        # tree and ``config.json`` on 2026-09-05: ``weight_bytes`` is the sum
        # of the seven safetensors shard FILES (`tests/test_gpu_model_catalog.py`
        # pins it to the seven figures; the index's ``total_size`` reads
        # 34,335,541,248, 12.4 MiB apart, which no verdict can turn on);
        # 64 layers, 64 heads over 8 KV heads
        # of width 128 (hidden 5120, intermediate 25600), a 40960-token
        # window; weights fp8 e4m3 block-quantised [128, 128] with dynamic
        # activations, cache and compute bf16 (see the module docstring for
        # why ``quantization`` says only "fp8"). ``parameter_billions`` is the
        # model card's 32.8B, which the config geometry reproduces: 64 layers
        # of 487.6M (attention 94.4M, MLP 393.2M) plus two untied
        # 151936 x 5120 embeddings.
        "name": "Qwen 3 32B (FP8)",
        "repo": "Qwen/Qwen3-32B-FP8",
        "weight_bytes": 34_322_567_640,
        "hidden_layers": 64,
        "attention_heads": 64,
        "kv_heads": 8,
        "head_dim": 128,
        "context_length": 40960,
        "quantization": "fp8",
        "parameter_billions": 32.8,
    },
    "qwen3_32b_fp16": {
        # The model the fit engine must REFUSE on the two-appliance pair (about
        # 61 GiB of weights against about 60 GiB of GPU memory in total, VD-127).
        # Measured off the Hub tree on 2026-09-05: seventeen safetensors shards
        # (the test pins the seventeen figures). The entry used to carry a
        # round 65 GB that no shard sum produces.
        "name": "Qwen 3 32B (FP16)",
        "repo": "Qwen/Qwen3-32B",
        "weight_bytes": 65_524_328_560,
        "hidden_layers": 64,
        "attention_heads": 64,
        "kv_heads": 8,
        "head_dim": 128,
        "context_length": 32768,
        "quantization": "f16",
        "parameter_billions": 32.5,
    },
    "mistral_small_24b_bf16": {
        # Not recommended - a sizing proof, not a suggested serve. It exists
        # for the same bigger-than-one-box case as the FP8 32B above, but
        # without that entry's FP8 Triton JIT, whose cold first start ran
        # past twenty minutes per attempt on gfx1151 on 2026-09-05: about
        # 44 GiB of bf16 weights, over one node's ~30 GiB-GTT aperture and
        # under the pair's ~60 GiB, plain bf16 so vLLM starts without a
        # quantization kernel to compile.
        #
        # Repo: ``unsloth/Mistral-Small-24B-Instruct-2501``, a non-gated bf16
        # re-upload of the same weights, not the official ``mistralai/
        # Mistral-Small-24B-Instruct-2501``. That repo sits behind a
        # click-through license gate, and its Hub tree carries a second full
        # copy of the weights as a single ``consolidated.safetensors`` (about
        # 44 GiB again, alongside the same ten shards below) that a plain
        # tree listing cannot tell apart from the ones vLLM actually loads.
        # The unsloth mirror carries only the ten shards.
        #
        # Measured off the Hub tree and ``config.json`` on 2026-09-05:
        # ``weight_bytes`` is the sum of the ten safetensors shard FILES
        # (`tests/test_gpu_model_catalog.py` pins it to the ten figures; the
        # index's ``total_size`` reads 47,144,806,400, 41.5 KiB under the
        # file sum, which no verdict can turn on); 40 layers, 32 heads over
        # 8 KV heads of width 128 (hidden 5120, intermediate 32768), a
        # 131072-token vocabulary, untied embeddings, a 32768-token window.
        # ``parameter_billions`` is the Hub's own safetensors parameter count
        # for this repo, 23,572,403,200 (23.6B), which the config geometry
        # reproduces: 40 layers of about 556.7M (attention 52.4M, MLP 503.3M)
        # plus two untied 131072 x 5120 embeddings.
        "name": "Mistral Small 24B (BF16)",
        "repo": "unsloth/Mistral-Small-24B-Instruct-2501",
        "weight_bytes": 47_144_848_872,
        "hidden_layers": 40,
        "attention_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "context_length": 32768,
        "quantization": "bf16",
        "parameter_billions": 23.6,
    },
}

#: The spec fields a catalog entry contributes to `build_model_spec` (the repo
#: is serving metadata, not a sizing input; the display name rides along so
#: the verdict can name the model).
SPEC_FIELDS = (
    "name", "weight_bytes", "hidden_layers", "attention_heads", "kv_heads",
    "head_dim", "context_length", "quantization", "parameter_billions",
)


def catalog_spec_body(catalog_id: str) -> Dict[str, Any]:
    """The `build_model_spec` request body for one catalog id.

    The one place an entry is projected onto the spec's fields, so the deploy
    and every test that sizes a preset hand the engine the same body.
    ``KeyError`` for an id the catalog does not carry: callers resolve the id
    through `hf_model_source.resolve_model_source` first.
    """
    entry = GPU_MODEL_CATALOG[catalog_id]
    return {field: entry[field] for field in SPEC_FIELDS}


def catalog_defaults(catalog_id: str) -> Dict[str, Any]:
    """What the deploy form opens on for one entry: ``{intent, max_model_len}``, or ``{}``.

    An entry says so only where a layout was decided for it; every other
    entry leaves the form's own defaults alone. These are choices the form
    starts from, not sizing inputs: they are not in :data:`SPEC_FIELDS`, and
    the form sends whatever its fields hold when Serve is pressed.
    """
    entry = GPU_MODEL_CATALOG[catalog_id]
    if "default_intent" not in entry:
        return {}
    return {
        "intent": entry["default_intent"],
        "max_model_len": entry["default_max_model_len"],
    }


def served_repo(model_source: Any) -> Tuple[str, Optional[str]]:
    """``(repo, revision)`` vLLM serves for a deploy's ``model_source``.

    A catalog id serves its entry's repo at the default revision; a pasted
    Hugging Face reference serves what it names. The deploy and the fit
    preview both read this, so the form's thinking switch is decided about
    the same repo the units are rendered for. ``ValueError`` for a source
    `resolve_model_source` refuses.
    """
    return repo_of_source(resolve_model_source(model_source, GPU_MODEL_CATALOG.keys()))


def repo_of_source(source: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """``(repo, revision)`` of an already-resolved source (`resolve_model_source`'s answer)."""
    if source["kind"] == "catalog":
        return GPU_MODEL_CATALOG[source["id"]]["repo"], None
    return str(source["repo"]), source["revision"]
