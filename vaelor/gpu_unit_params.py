"""Every value a vLLM/Ray unit is rendered from, validated - on both sides of the bridge.

A GPU-tier unit's ``ExecStart`` is assembled by `gpu_pool_runtime` from a small
set of typed values: a deployment name, a repo and revision, addresses, a NIC,
ports, degrees, a memory fraction, a context length and the node's numeric
GIDs. Each value is checked by exactly one function here, and nothing reaches a
unit that did not pass one of them.

**Why these rules live in their own module (VD-143).** The controller's root
hardware bridge renders every unit it installs from the SAME template the
runtime uses on a worker, out of parameters an unprivileged client sends it -
never out of unit text. So these checks are no longer only input hygiene for a
deploy the operator started: on the controller they are the root boundary's own
refusal of a parameter that could carry a second argument, a systemd directive
or a shell word into a root-run container command. They are pure functions of
their input, with no filesystem, network or state, which is what lets the root
side call them on a value it received over a socket. `GpuPoolRuntime` binds
each one as a static method under its historical name, so its callers and tests
read exactly what they always did.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Iterable, List, Optional

from . import gpu_node_facts
#: `model_thinking` owns what the thinking setting is and its refusal sentence.
from .model_thinking import require_boolean
#: `cluster_gpu_sizing` owns the launch fraction's band (VD-129): the fit sizes
#: a replica at it, the deploy launches at it, and this module refuses a
#: ``--gpu-memory-utilization`` outside it - one rule, three readers.
from .cluster_gpu_sizing import validate_gpu_memory_utilization
#: `gpu_serving_target` owns where a served model may answer - the port band and
#: the two API bind hosts - because the module that decides whether an endpoint
#: may be exposed is the one that has to know both.
from .gpu_serving_target import api_bind_host, serving_port
#: `vllm_serve_options` owns the model's serving options - parsers, text-only,
#: multi-token prediction, the tuned MoE tables - and their one rule.
from .vllm_serve_options import ServeOptions, require_options

#: Characters that must never reach the merged lead's ``bash -c`` script.
#: systemd keeps a double-quoted ``ExecStart`` argument as ONE word, but inside
#: it ``%`` is still a specifier and ``$VAR``/``${...}`` are still expanded, and
#: bash then reads the rest; ``"`` would end the argument outright.
_UNSAFE_SCRIPT_CHARACTERS = "\"'$%`;|&<>(){}\\\n\r\x00"

#: The most numeric GIDs one container is granted. Two (render, video) on every
#: box measured; the ceiling only stops a request carrying a list long enough to
#: turn one ``ExecStart`` into a megabyte.
_MAX_GROUP_IDS = 8

#: The largest GID the kernel hands out: ``(gid_t)-1`` is "no group", so the
#: last usable number is one below it.
_MAX_GROUP_ID = 2 ** 32 - 2


def shell_safe(script: str) -> str:
    """Refuse a start script carrying shell or systemd metacharacters.

    Belt-and-braces: every token the script is built from - name, repo, revision,
    IP, interface, integers, fraction - has already been through its own regex,
    so nothing should trip this, which is why failing loudly here is cheaper than
    emitting a unit whose meaning depends on how systemd and bash split it.
    """
    text = str(script)
    if any(character in text for character in _UNSAFE_SCRIPT_CHARACTERS):
        raise ValueError(
            "The vLLM start script may not contain shell metacharacters."
        )
    return text


def cluster_address(value: Any) -> str:
    """A node's private cluster IPv4, or a refusal: never loopback, link-local
    or the unspecified address.

    ``0.0.0.0`` is named explicitly because `ipaddress` counts it PRIVATE
    (the ``0.0.0.0/8`` "this network" block), so ``is_private`` alone
    admitted it - and as a Ray head's ``bind_ip`` or a gate's listen
    address (VD-129) it means every interface, which is the one thing
    this rule exists to refuse.
    """
    address = ipaddress.ip_address(str(value))
    if (
        address.version != 4
        or not address.is_private
        or address.is_unspecified
        or address.is_loopback
        or address.is_link_local
    ):
        raise ValueError("GPU serving requires private static IPv4 addresses.")
    return str(address)


def interface(value: Any) -> str:
    """Validate a NIC name through the one rule `gpu_node_facts` owns."""
    return gpu_node_facts.interface_name(value)


def model_repo(value: Any) -> str:
    """A model reference vLLM ``serve`` accepts: an ``org/name`` repo id.

    The repo has already been validated by `hf_model_source`; this is a
    second, narrower gate so a value that never went through the parser (a
    catalog entry's ``repo``) cannot inject shell-meaningful text into the
    unit's ``ExecStart`` or a container argument.
    """
    model = str(value).strip()
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}",
        model,
    ):
        raise ValueError("The model must be a 'org/name' Hugging Face repo.")
    return model


def revision(value: Any) -> Optional[str]:
    """A git ref a pull or a server may pin, or ``None`` for the default branch.

    Plain characters, and the shape ``git check-ref-format`` accepts for them:
    no ``..``, no empty ``/`` part (so no ``//`` and no trailing ``/``), no part
    beginning with ``.`` and no part ending ``.lock``. A commit hash, a branch
    such as ``main`` and a tag such as ``v1.0`` or ``refs/pr/3`` all pass.
    """
    if value is None:
        return None
    pinned = str(value).strip()
    parts = pinned.split("/")
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", pinned)
        or ".." in pinned
        or any(not part or part.startswith(".") or part.endswith(".lock")
               for part in parts)
    ):
        raise ValueError("The model revision is not a valid git ref.")
    return pinned


def degree(value: Any, field: str) -> int:
    """A parallel degree or device count: a whole number from one to eight."""
    try:
        whole = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"The {field} must be a whole number.")
    if not 1 <= whole <= 8:
        raise ValueError(f"The {field} must be between 1 and 8.")
    return whole


def context_length(value: Any) -> int:
    """``--max-model-len``: a whole number inside the range vLLM is run at."""
    try:
        context = int(value)
    except (TypeError, ValueError):
        raise ValueError("The max model length must be a whole number.")
    if not 256 <= context <= 1_048_576:
        raise ValueError("The max model length is out of the supported range.")
    return context


def group_ids(values: Iterable[Any]) -> List[int]:
    """The numeric GIDs a GPU container is granted, each a real group number.

    A GID reaches the unit as ``--group-add <n>``; ``int`` alone admitted a
    negative number (a flag-shaped word to docker) and a list of any length.
    Booleans are refused because ``int(True)`` is 1, which is a real group on
    most systems and never what a caller meant.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError("The GPU device group ids must be a list of numbers.")
    numbers: List[int] = []
    for value in list(values or []):
        try:
            if isinstance(value, bool):
                raise TypeError("a boolean is not a group number")
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError("A GPU device group id must be a whole number.")
        if not 0 <= number <= _MAX_GROUP_ID:
            raise ValueError("A GPU device group id is out of range.")
        numbers.append(number)
    if len(numbers) > _MAX_GROUP_IDS:
        raise ValueError("Too many GPU device groups were requested.")
    return numbers


def thinking_default(value: Any) -> bool:
    """Whether the served model thinks by default: a real boolean and nothing else.

    It becomes one fixed ``vllm serve`` word (`model_thinking`), so no value
    can carry text into the unit; a string such as ``"false"`` is refused
    rather than read for truthiness, because it is truthy.
    """
    return require_boolean(value)


#: The port band and the API bind-host allowlist, and the launch fraction's band,
#: under the names the runtime binds - each rule still has its one home.
port = serving_port
api_host = api_bind_host
memory_fraction = validate_gpu_memory_utilization


def serve_options(value: Any) -> ServeOptions:
    """The model's serving options: five typed fields from two allowlists, or refused.

    Each field becomes a fixed ``vllm serve`` word or a fixed container
    variable (`vllm_serve_options`), so no value can carry text into a unit.
    """
    return require_options(value)
