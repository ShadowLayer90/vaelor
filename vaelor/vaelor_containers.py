"""Which Docker containers are Vaelor's own, rather than apps the owner installed.

VD-133 1.5.5 #12: the 1.5.4 Apps page listed Phoenix, the web-research search
container and the cluster's model-serving containers as "discovered apps" the
owner had installed. They are parts of Vaelor. The rule is a LABEL Vaelor sets
where it creates a container - ``io.vaelor.owner=<component>`` - and never a
list of names typed here: the next Vaelor container is recognised by carrying
the label, not by someone remembering to extend a list (LESSONS: per-symptom
patches).

Two kinds of container cannot carry the label, and are recognised by the name
their creator gives them, read from that creator's own constant or naming rule
(so this module never spells a container name itself):

* containers an earlier release started, which keep running unlabelled until
  they are next recreated (an upgrade does not recreate Phoenix or a serving
  container); and
* the containers the root bridge starts from units whose exact argv it pins
  (the vLLM serving and pull units, `bridge_managed_units`), the gates
  rendered beside them, and the speed-up build containers; their names are a
  namespace `gpu_pool_units`, `agent_service` and the build modules own.

`deploy/maintain-vaelor.sh` spells the same families for the uninstall,
where no Python runs; `tests/test_vaelor_containers.py` holds the two equal.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Mapping, Tuple

#: The label every container Vaelor starts for itself carries; its value names
#: the part of Vaelor that owns it. The web-research compose file has set it
#: since before this module existed.
VAELOR_OWNER_LABEL = "io.vaelor.owner"


def owner_label_args(component: str) -> list[str]:
    """The ``docker run`` words that mark a container as ``component``'s."""
    return ["--label", f"{VAELOR_OWNER_LABEL}={component}"]


#: Docker's compose label. A container a compose project started is an app -
#: the owner's, or Vaelor's own web research, which carries the label - and is
#: never matched by name (review F3).
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"


@lru_cache(maxsize=1)
def _creator_names() -> Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[Any, str], ...]]:
    """``(exact names, (compiled name pattern, component) rules)`` from each creator.

    Imported here, lazily, so that a creator importing this module's label
    does not import itself back at load time. Each pattern is exactly the
    shape its creator makes and no wider - the uninstall script spells the
    same families, and `tests/test_vaelor_containers.py` holds the two to the
    same verdict on every name (review F5).
    """
    from .agent_service import gate_container_name
    from .cluster_plan_contract import LLM_SERVICE_PREFIX
    from .gpu_pool_units import _PULL_PREFIX, _ROLES, _SERVING_PREFIX, BALANCER_ROLE
    from .gpu_rocmfpx_service import GPU_CONTAINER_NAME
    from .llm_server_proxy import PROXY_CONTAINER_NAME
    from .phoenix_service import PHOENIX_CONTAINER_NAME
    from .vllm_guesser_build import CONTAINER_PREFIX as GUESSER_BUILD_PREFIX
    from .vllm_moe_hip_build import CONTAINER_PREFIX as HIP_BUILD_PREFIX

    exact = (
        (PHOENIX_CONTAINER_NAME, "phoenix"),
        (PROXY_CONTAINER_NAME, "llm-server"),
        (GPU_CONTAINER_NAME, "gpu-model"),
    )
    probe = "x" * 8
    gate_prefix, gate_suffix = gate_container_name(probe).split(probe)
    roles = "|".join(re.escape(role) for role in (*_ROLES, BALANCER_ROLE))
    word = "[a-z0-9-]+"
    rules = (
        (_SERVING_PREFIX + word + "-(?:" + roles + ")", "gpu-serving"),
        (_PULL_PREFIX + word, "gpu-serving"),
        (GUESSER_BUILD_PREFIX + "[0-9a-f]+", "gpu-serving"),
        (HIP_BUILD_PREFIX + "[0-9a-f]+", "gpu-serving"),
        # A Swarm task container: <service>.<slot>.<task>. CPU serving, not GPU.
        (LLM_SERVICE_PREFIX + word + r"\.[0-9]+\.[a-z0-9]+", "cluster-serving"),
        (re.escape(gate_prefix) + word + re.escape(gate_suffix), "cluster-agent"),
    )
    return exact, tuple((re.compile(pattern), component) for pattern, component in rules)


def vaelor_component(labels: Mapping[str, Any], name: str) -> str:
    """The part of Vaelor a container belongs to, or ``""`` for the owner's own.

    ``labels`` are the container's Docker labels and ``name`` its name without
    the leading ``/``. The label decides when present; otherwise, for a
    container no compose project started, the creators' naming rules (see the
    module docstring for why they are needed).
    """
    labels = labels if isinstance(labels, Mapping) else {}
    owner = labels.get(VAELOR_OWNER_LABEL)
    if isinstance(owner, str) and owner.strip():
        return owner.strip()
    if labels.get(COMPOSE_PROJECT_LABEL):
        return ""
    exact, rules = _creator_names()
    for known, component in exact:
        if name == known:
            return component
    for pattern, component in rules:
        if pattern.fullmatch(name):
            return component
    return ""
