"""Make the 4-bit copy of the DFlash guesser, inside the pinned vLLM image (VD-216 work order item 2).

This file is Vaelor's own (GPL-2.0-only). It never runs in the control plane:
`vllm_guesser_build` checks its sha256, copies those bytes into a root-only
build folder and runs them as ``python3 -I`` in ONE locked-down container of
the pinned vLLM 0.27 image (no network, no GPU, read-only root filesystem,
every capability dropped), whose own torch, safetensors and compressed-tensors
0.17 do the work. Nothing is downloaded.

**What it makes** (measured by WO2 on the Z2, 2026-10-15, kb N57/G60): a
compressed-tensors ``pack-quantized`` w4a16 copy of the pinned bf16 drafter -
int4, symmetric, group 32, the same scheme as the 30B's own checkpoint - by
round-to-nearest with a per-group search over the clip ratio 1.00 to 0.80
(steps of 0.01) that keeps the smallest sum of ``|error| ** 2.4``, the
quantity llm-compressor's ``mse`` observer minimises. No calibration data.
o, gate, up, down and fc are quantized; q, k and v stay bf16 because vLLM
0.27's DFlash loader reads the fused qkv weight directly (G60), and lm_head,
the embeddings, the norms and the vocabulary maps stay as they are.

**Deterministic by construction, and checked so** (LESSONS 3): torch runs on a
fixed number of threads with deterministic algorithms, every step is
elementwise or a fixed-length reduction over one 32-wide group, and the scale
is quantized against its stored bf16 value. Measured on the Z2 (2026-10-15):
repeated builds on 4 cores, 8 cores and all 32 gave the same bytes, and the
same bytes as the file WO2 measured. The build module pins the sha256 of
every output file, so a machine whose result differs serves the bf16 drafter
and says so rather than serving bytes nobody measured.

**The input is checked before it is read** - the very bytes quantized are the
bytes hashed (#178): each source file is read once, its sha256 compared with
the pin the plan carries, and only then parsed.

usage (inside the container): ``python3 -I quantize.py PLAN_JSON``; prints
one JSON line and exits 0, or prints the reason and exits 1.
"""

import hashlib
import json
import os
import sys
import time

#: The fixed thread count: the result must not depend on the machine's cores.
THREADS = 8
#: int4, symmetric: the stored range and the half-width the scale divides by.
QMIN, QMAX, HALF = -8, 7, 7.5
#: The clip ratios searched per group, largest first: 1.00, 0.99, ... 0.80, as
#: ``torch.arange(CLIP_FROM, CLIP_TO, CLIP_STEP)`` makes them in float32 - the
#: exact values WO2 measured with. A Python-float spelling of the same ratios
#: rounds differently and moved three of 1.5 million stored values (measured
#: on the Z2, 2026-10-15), so the spelling is part of the recipe.
CLIP_FROM, CLIP_TO, CLIP_STEP = 1.0, 0.795, -0.01
#: The error power the search minimises (llm-compressor's ``mse`` observer).
ERROR_POWER = 2.4
#: Projections quantized; anything else named ``*.weight`` is left as it is.
LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
#: Projections kept bf16 (G60: vLLM 0.27's DFlash loader reads qkv directly).
KEEP_BF16 = ("q_proj", "k_proj", "v_proj")
#: The weight file and the two files copied beside it.
WEIGHTS = "model.safetensors"
CONFIG = "config.json"
CONFIG_CODE = "config.py"


def refuse(reason):
    """Print why, and stop with a failure: the build records the reason."""
    sys.stdout.write(json.dumps({"ok": False, "reason": str(reason)[:300]}) + "\n")
    sys.exit(1)


def read_checked(path, expected):
    """The file's bytes, once, after its sha256 matched ``expected``."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as error:
        refuse("the pinned drafter file {} could not be read ({})".format(
            os.path.basename(path), error.strerror or error))
    if hashlib.sha256(data).hexdigest() != expected:
        refuse("the drafter file {} is not the pinned one (its checksum differs)".format(
            os.path.basename(path)))
    return data


def group_scales(torch, weight, group):
    """``(int4 values, bf16 scales)`` for one weight, by the clip search above."""
    rows, columns = weight.shape
    x = weight.float().reshape(rows, columns // group, group)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best_error = torch.full(amax.shape, float("inf"))
    best_scale = amax / HALF
    for ratio in torch.arange(CLIP_FROM, CLIP_TO, CLIP_STEP):
        scale = (amax * ratio) / HALF
        values = torch.clamp(torch.round(x / scale), QMIN, QMAX)
        error = ((values * scale - x).abs() ** ERROR_POWER).sum(-1, keepdim=True)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
    # The stored scale is bf16: quantize against the value vLLM will read.
    stored = best_scale.to(torch.bfloat16).float()
    values = torch.clamp(torch.round(x / stored), QMIN, QMAX).to(torch.int8).reshape(rows, columns)
    return values, stored.reshape(rows, columns // group).to(torch.bfloat16)


def quantized_name(key):
    """The layer name a weight key is quantized under, or ``None`` to keep it."""
    if not key.endswith(".weight"):
        return None
    name = key[: -len(".weight")]
    leaf = name.rsplit(".", 1)[-1]
    if (leaf in LINEAR or name == "fc") and leaf not in KEEP_BF16:
        return name
    return None


def quantization_config(group):
    """The ``quantization_config`` vLLM reads: compressed-tensors, w4a16, this group size."""
    return {
        "config_groups": {"group_0": {
            "format": "pack-quantized", "input_activations": None,
            "output_activations": None, "targets": ["Linear"],
            "weights": {
                "actorder": None, "block_structure": None, "dynamic": False,
                "group_size": group, "num_bits": 4, "observer": "mse",
                "observer_kwargs": {}, "strategy": "group", "symmetric": True,
                "type": "int",
            },
        }},
        "format": "pack-quantized",
        "global_compression_ratio": None,
        # q, k and v by their own names and by the fused name the loader
        # builds (the draft config has no packed-modules map, G60).
        "ignore": ["lm_head"] + [r"re:.*self_attn\." + leaf + "$" for leaf in KEEP_BF16]
        + [r"re:.*self_attn\.qkv_proj$"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
    }


def main(argv):
    if len(argv) != 2:
        refuse("usage: quantize.py PLAN_JSON")
    try:
        plan = json.loads(argv[1])
        source, out, group = plan["source"], plan["out"], int(plan["group"])
        pins = dict(plan["inputs"])
    except (ValueError, KeyError, TypeError) as error:
        refuse("the build plan could not be read ({})".format(error))
    started = time.monotonic()
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    import torch
    from safetensors.torch import load, save
    from compressed_tensors.compressors.pack_quantized.helpers import (
        pack_to_int32, unpack_from_int32,
    )

    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    weights = load(read_checked(os.path.join(source, WEIGHTS), pins[WEIGHTS]))
    config = json.loads(read_checked(os.path.join(source, CONFIG), pins[CONFIG]).decode("utf-8"))
    code = read_checked(os.path.join(source, CONFIG_CODE), pins[CONFIG_CODE])
    tensors, errors = {}, []
    for key in sorted(weights):
        tensor = weights[key]
        name = quantized_name(key)
        if name is None:
            tensors[key] = tensor.contiguous()
            continue
        values, scales = group_scales(torch, tensor, group)
        packed = pack_to_int32(values, 4)
        if not torch.equal(unpack_from_int32(packed, 4, torch.Size(tensor.shape)), values):
            refuse("packing {} did not round-trip".format(name))
        tensors[name + ".weight_packed"] = packed
        tensors[name + ".weight_scale"] = scales
        tensors[name + ".weight_shape"] = torch.tensor(list(tensor.shape), dtype=torch.int64)
        restored = (values.float().reshape(tensor.shape[0], -1, group)
                    * scales.float().unsqueeze(-1)).reshape(tensor.shape)
        errors.append(((restored - tensor.float()).norm() / tensor.float().norm()).item())
    if not errors:
        refuse("the drafter holds no layer this recipe quantizes")
    config["quantization_config"] = quantization_config(group)
    outputs = {
        WEIGHTS: save(tensors, metadata={"format": "pt"}),
        CONFIG: (json.dumps(config, indent=2) + "\n").encode("utf-8"),
        CONFIG_CODE: code,
    }
    for name, data in sorted(outputs.items()):
        with open(os.path.join(out, name), "wb") as handle:
            handle.write(data)
    sys.stdout.write(json.dumps({
        "ok": True, "seconds": round(time.monotonic() - started, 1),
        "outputs": {name: hashlib.sha256(data).hexdigest() for name, data in outputs.items()},
        "layers": len(errors), "relative_error_max": round(max(errors), 5),
        "relative_error_mean": round(sum(errors) / len(errors), 5),
    }) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
