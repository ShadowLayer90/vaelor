"""The ``vllm serve`` arguments every GPU serving unit carries, built in one place.

Moved out of `gpu_pool_runtime` (at the 1,000-line ceiling) unchanged, so the
single-node replica and the merged distributed lead cannot drift on how they
spell the server. `GpuPoolRuntime._serve_arguments` still answers under its
historical name and delegates here; the root bridge reaches this through the
same template (VD-143), so every value is checked by its `gpu_unit_params` rule
before it becomes a word.

The one word this module added is the thinking default (`model_thinking`):
``--default-chat-template-kwargs.enable_thinking=<true|false>`` for a model
family whose chat template reads it, and nothing for any other.
"""

from __future__ import annotations

from typing import Any, List, Optional

from . import gpu_unit_params as unit_params
from .gpu_serving_target import SERVER_LOOPBACK_HOST
from .model_thinking import serve_thinking_arguments
from .serving_profiler import SERVING_PROFILES_MOUNT
from .vllm_serve_options import serve_arguments as model_arguments


def serve_arguments(
    *, model: str, port: int,
    tensor_parallel_size: int, pipeline_parallel_size: int,
    gpu_memory_utilization: float, max_model_len: int,
    distributed: bool, revision: Optional[str],
    api_host: str = SERVER_LOOPBACK_HOST, thinking_default: bool = False,
    vllm_options: Any = None,
) -> List[str]:
    """The ``serve <repo> ...`` arguments both serving forms share.

    The single-node unit appends them to the image, whose entrypoint is a thin
    ``vllm "$@"``; the merged lead overrides that with bash and so spells the
    server binary itself. Either way the repo is positional rather than
    ``--model``, and one builder keeps the forms from drifting.

    ``api_host`` is where the OpenAI API binds: loopback, the only value the
    allowlist admits (`gpu_serving_target.api_bind_host`; VD-156 amended), on
    every lead. A worker's is reached through its keyed gate. Ray's GCS and
    the RCCL sockets are unaffected - they bind the node's real NIC through
    ``--node-ip-address`` and the RCCL environment, which is a different
    address on a different socket.

    ``thinking_default`` is whether the model thinks before answering unless a
    request says otherwise (`model_thinking`); a boolean, never text.
    ``vllm_options`` is the model's serving options (`vllm_serve_options`):
    its parsers - the reasoning parser among them, so an answer a request asks
    to think arrives with its thinking in ``reasoning``, not inline in
    ``content`` - the text-only switch, a hybrid model's cache switches on a
    GPU without FP8 hardware, and multi-token prediction.

    **The word order is fixed:** the model's serving options where
    ``--tool-call-parser hermes`` always stood, then the profiler, then the
    thinking default - so a unit rendered with neither value is byte-identical
    to one rendered before either existed.
    """
    served = unit_params.model_repo(model)
    pinned = unit_params.revision(revision)
    tensor = unit_params.degree(tensor_parallel_size, "tensor-parallel-size")
    pipeline = unit_params.degree(pipeline_parallel_size, "pipeline-parallel-size")
    published = unit_params.port(port)
    utilization = unit_params.memory_fraction(gpu_memory_utilization)
    context = unit_params.context_length(max_model_len)
    thinking = unit_params.thinking_default(thinking_default)
    parts = [
        "serve", served,
        "--host", unit_params.api_host(api_host),
        "--port", str(published),
        "--tensor-parallel-size", str(tensor),
    ]
    if distributed:
        parts += ["--pipeline-parallel-size", str(pipeline)]
    parts += [
        "--gpu-memory-utilization", f"{utilization:.2f}",
        "--max-model-len", str(context),
    ]
    parts += model_arguments(unit_params.serve_options(vllm_options))
    parts += [
        # vLLM's torch profiler -> host-mounted dir (VD-128); idle until /start_profile.
        "--profiler-config.profiler=torch",
        f"--profiler-config.torch_profiler_dir={SERVING_PROFILES_MOUNT}",
    ]
    parts += serve_thinking_arguments(served, thinking)
    if distributed:
        parts += ["--distributed-executor-backend", "ray"]
    if pinned is not None:
        parts += ["--revision", pinned]
    return parts
