"""Assignment and cleanup policy for Vaelor-managed local model credentials."""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Optional

from .credential_broker import CredentialError
from .model_credential_roles import (  # noqa: F401 - the id markers, re-exported
    AI_CHAT_MODEL_PREFIX,
    ASSISTANT_NPU_PREFIX,
    MANAGED_LOCAL_PREFIX,
    carries_the_npu_marker,
)
from .gpu_model_choice import names_a_gfx1151
from .inference_model_choice import INSTALLED_FLM_MODELS
from .platforms.accelerators import discover_accelerators


#: Every managed-local credential id starts with this; the NPU and GPU-chat
#: deploys add their own marker after it (`model_credential_roles`, VD-202).
PREFIX = MANAGED_LOCAL_PREFIX

#: The broker label a managed-local credential carries. One home, because
#: three deploy paths write it - the llama.cpp path, the flm-real NPU path and
#: the GPU fork path - and :func:`managed_model_by_label` reads it back; a
#: sentence written in two places in one tree is the shape #98 records
#: (LESSONS 6 / VD-090). `{}` is filled with the model's short name.
MANAGED_LOCAL_CREDENTIAL_LABEL = "Managed local model · {}"


def managed_model_by_label(models_root: Path, label: str) -> Optional[Path]:
    """The managed ``.gguf`` a managed-local credential LABEL names, or ``None``.

    A managed-local deploy labels its credential
    ``MANAGED_LOCAL_CREDENTIAL_LABEL.format(model.stem[:48])``, so the label
    carries the served file's (48-char-truncated) stem - the one durable
    identifier a generic fork lease keeps across a reboot, since its stored
    profile is normalised to base_url/model/api_key with an empty model. This
    reverses that: it scans ``models_root`` for a ``.gguf`` whose ``stem[:48]``
    equals the label's stem.

    ``None`` when the label is not a managed-local label, when no file matches
    (the model was removed), or when MORE than one file matches - the same
    ambiguity discipline :func:`vaelor.model_footprint.identify_by_file` keeps
    (VD-065): a stem two files share resolves to nothing rather than a guess,
    so the reconcile never relaunches the wrong model.

    **The whole models directory is scanned, uncapped.** An earlier
    ``[:200]`` slice on the UNORDERED ``rglob`` could miss the target model
    non-deterministically on a box with more than 200 gguf files - after a
    reboot the AI-Chat tier would then stay down. The scan is bounded by the
    managed models directory, so it is a directory walk, not an unbounded one.

    **Known bound (LOW/MEDIUM-6, left as-is):** the label carries only
    ``stem[:48]``, so two models whose file stems agree in their first 48
    characters collide and BOTH resolve to nothing (the len==1 guard). That
    is the safe direction - a no-op reconcile, never the wrong model - and
    widening the label format is riskier than the low-likelihood collision,
    so the truncation stands.
    """
    prefix = MANAGED_LOCAL_CREDENTIAL_LABEL.format("")
    if not label.startswith(prefix):
        return None
    stem = label[len(prefix):]
    if not stem or not models_root.exists():
        return None
    matches = {
        str(path.resolve())
        for path in models_root.rglob("*.gguf")
        if path.is_file() and path.stem[:48] == stem
    }
    return Path(matches.pop()) if len(matches) == 1 else None


def _endpoint_of(profile: dict) -> str:
    """Return ``scheme://host:port`` for a resolved compatible-endpoint profile.

    The discriminator between "the same local model, re-registered" and "a
    different, independent local model" is the connection endpoint, not the path
    or the model name. ``base_url`` may carry a ``/v1`` suffix (llama-server) or
    none (flm-real), so only the scheme and network location are compared. An
    unparseable or non-endpoint profile (e.g. a hosted ``openai`` lease that
    carries a token instead of a ``base_url``) yields ``""``, which never matches
    a real managed-local endpoint.
    """
    parts = urllib.parse.urlsplit(str(profile.get("base_url", "")).strip())
    if not parts.scheme or not parts.hostname:
        return ""
    return "{}://{}".format(parts.scheme, parts.netloc.lower())


def _managed_local_endpoint(broker, credential_id: str) -> str:
    """Read a managed-local credential's endpoint from its stored profile.

    Managed-local credentials are always ``openai-compatible``, which the
    ``deployment-agent`` purpose accepts, so ``resolve`` returns the decrypted
    profile (including ``base_url``) without needing the credential to already be
    active for any purpose. A resolve failure yields ``""`` so the caller treats
    the endpoint as unknown and errs toward preservation.
    """
    try:
        return _endpoint_of(broker.resolve(credential_id, "deployment-agent"))
    except CredentialError:
        return ""


def pins_an_flm_tag(model: str) -> bool:
    """Whether a managed-local credential's pinned model is an NPU (FLM) tag.

    The one test for "this is the NPU Assistant's credential" (W5): the NPU
    deploy pins the FLM tag it serves before activating (VD-108), and a GPU or
    llama.cpp credential never carries one. The credential sweep, the NPU's own
    port and the mode switch's restore contract all ask it here.
    """
    return str(model or "") in set(INSTALLED_FLM_MODELS)


def _tiers(broker) -> dict:
    """``{credential_id: "npu" | "gpu"}`` for every Vaelor-managed local credential.

    **The tier is part of "the same server" (W5, W4d-D27, LESSONS 6).** The
    endpoint alone was the discriminator, and a loopback port is reused the
    moment its owner stops: in Mode B the 27B is stopped, the NPU Assistant came
    back on the 27B's port, and the sweep deleted the 27B's credential - the
    Mode A fallback - as a "re-registration". An NPU credential is pinned to the
    FLM tag it serves (`_deploy_npu_assistant` selects it before activation); a
    GPU or llama.cpp one never carries an FLM tag. Read off the broker's
    listing, which reports the pin without decrypting anything.
    """
    return {
        item_id: "npu" if carries_the_npu_marker(item) else "gpu"
        for item in broker.list()
        if (item_id := str(item.get("id", ""))).startswith(PREFIX)
    }


def _managed_local_peers(tiers: dict, credential_id: str) -> list:
    """Every OTHER Vaelor-managed local credential OF THIS ONE'S TIER.

    A managed-local credential carries the :data:`PREFIX`; a hosted or
    user-managed provider does not, so it is never a candidate. A credential of
    the other tier is an independent server whatever port it names, so it is
    never a candidate either (W4d-D27).
    """
    tier = tiers.get(credential_id, "gpu")
    return [
        item_id for item_id, item_tier in tiers.items()
        if item_id != credential_id and item_tier == tier
    ]


def _same_endpoint_peers(broker, peers: list, endpoint: str) -> list:
    """The peers that share THIS deploy's connection endpoint.

    The discriminator between "the same local model, re-registered" and "a
    different, independent local model" is the endpoint (see :func:`_endpoint_of`)
    within one tier, so only same-endpoint peers are a redeploy of this server
    and stale. An unresolved endpoint (``""``) matches nothing: that is the
    primary data-loss guard, because a genuine endpoint never collapses to the
    empty string, so an unreadable peer can never be mistaken for a
    same-endpoint one.
    """
    return [
        peer for peer in peers
        if endpoint and _managed_local_endpoint(broker, peer) == endpoint
    ]


def gpu_tier_endpoints(broker) -> list:
    """The endpoint of every managed-local credential that is NOT the NPU's.

    A GPU or llama.cpp model keeps its credential while it is stopped (Mode B,
    an unload) and comes back on the port it names, so the NPU's port choice
    reads these as claimed (F6, :func:`vaelor.flm_supervisor.npu_port_claims`).
    An unreadable one yields ``""``, which names no port.
    """
    return [
        _managed_local_endpoint(broker, credential_id)
        for credential_id, tier in _tiers(broker).items() if tier == "gpu"
    ]


def model_name_from_label(label: str) -> str:
    """The model name a :data:`MANAGED_LOCAL_CREDENTIAL_LABEL` carries, or ``""``.

    The inverse of the one template (W7-D2, LESSONS 6): a label written any
    other way names no model here, rather than being guessed at.
    """
    head, _, tail = MANAGED_LOCAL_CREDENTIAL_LABEL.partition("{}")
    text = str(label or "")
    if len(text) <= len(head) + len(tail) or not text.startswith(head) or not text.endswith(tail):
        return ""
    return text[len(head):len(text) - len(tail)].strip()


def gpu_tier_models(broker) -> list:
    """``[(endpoint, model name)]`` for every GPU-tier managed-local credential.

    The same credentials as :func:`gpu_tier_endpoints`, with the model each
    names: its selected model, else the model its label was written with, else
    ``""``. Never its id or anything secret (W7-D2: a port refusal names the
    stored model that comes back on the port).
    """
    names = {
        str(item.get("id", "")): (
            str(item.get("selected_model") or "").strip()
            or model_name_from_label(item.get("label", ""))
        )
        for item in broker.list()
    }
    return [
        (_managed_local_endpoint(broker, credential_id), names.get(credential_id, ""))
        for credential_id, tier in _tiers(broker).items() if tier == "gpu"
    ]


def _sweep_stale(broker, stale: list) -> None:
    """Delete the superseded same-endpoint credentials, non-critically.

    The purpose assignments have already been made by the time this runs, so a
    failure to remove obsolete generated metadata is not worth failing the
    activation over - the same reasoning both surfaces share.
    """
    for stale_id in stale:
        try:
            broker.delete(stale_id)
        except CredentialError:
            # Assignment succeeded; stale metadata cleanup is non-critical.
            pass


def activate_managed_local(
    broker, credential_id: str, *, accelerators: Optional[list] = None
) -> bool:
    """Make a healthy managed model the default without replacing user providers.

    This is the NPU Assistant deploy surface: it always claims
    ``deployment-agent`` (the Assistant's connection). Whether it ALSO adopts
    ``ai-chat`` on a fresh box now depends on the hardware, which is the fix
    VD-007 and VD-042 require.

    VD-007: the always-on Assistant runs on the NPU with a model Vaelor chooses;
    AI Chat runs on the GPU with a model the USER selects. VD-042: on the Z2 the
    NPU Assistant and the GPU AI-Chat are DELIBERATELY INDEPENDENT surfaces. So a
    dual-accelerator box (a gfx1151 GPU present) must NOT let the NPU Assistant
    deploy steal ``ai-chat`` onto the NPU - that would silently bind AI Chat to
    the NPU and offer flm-real's whole advertised catalog as "selectable", where
    picking an uninstalled model hangs. Only a SINGLE-accelerator box (a Pi, with
    no gfx1151 GPU) shares one managed model across both surfaces, and only that
    box adopts ``ai-chat`` on a fresh appliance - and never onto an NPU model,
    which AI Chat may not run on whatever the box (VD-202 item 2).

    The capability signal is the accelerator inventory: ``accelerators`` may be
    injected (kept injectable so this policy stays testable without threading
    hardware through callers); when omitted it is discovered via
    :func:`vaelor.platforms.accelerators.discover_accelerators`. A gfx1151 GPU in
    that list (:func:`vaelor.gpu_model_choice.names_a_gfx1151`) marks the box as
    dual-accelerator, so ``ai-chat`` is left UNASSIGNED for the user's later GPU
    pick.

    VD-085-style correction (unchanged): the original policy assumed a
    single-model appliance, so it treated EVERY other managed-local credential as
    stale and stole ``ai-chat`` onto whatever model was last deployed. The stale
    sweep and the multi-model ``ai-chat`` migration are keyed off the connection
    ENDPOINT so an independent model on a different endpoint (a GPU AI-Chat that
    already holds ``ai-chat``) is left untouched. The single-model Pi case is
    unchanged: its lone managed model shares the endpoint it is replacing, so both
    consumers still move and the old credential is still removed.

    The just-deployed model always becomes the ``deployment-agent`` connection;
    that is this surface's job and is unconditional.
    """
    # Other managed-local credentials are re-registration candidates only when
    # they are this one's tier and share this deploy's endpoint. A different
    # endpoint, or the other tier on a reused port, is an independent model
    # (the GPU AI Chat) and must survive (W4d-D27).
    tiers = _tiers(broker)
    npu_tier = tiers.get(credential_id) == "npu"
    peers = _managed_local_peers(tiers, credential_id)

    # Whether ai-chat should follow this deploy depends on what it points at now.
    active_chat_endpoint = None
    migrate_ai_chat = False
    try:
        active_chat = broker.resolve_active("ai-chat")
        active_chat_id = str(active_chat.get("credential_id", ""))
        if active_chat_id.startswith(PREFIX) and tiers.get(active_chat_id) == tiers.get(
            credential_id, "gpu"
        ):
            # A managed-local model of THIS tier holds ai-chat: migrate only if it
            # is this same endpoint being redeployed. A different-endpoint model,
            # or the other tier on a reused port, keeps ai-chat (W4d-D26: AI Chat
            # is never moved onto the NPU by a port coincidence).
            active_chat_endpoint = _endpoint_of(active_chat)
        # A user/hosted ai-chat assignment is never migrated (unchanged).
    except CredentialError:
        # No credential is active for ai-chat: a fresh appliance. Adopt ai-chat
        # onto this NPU deploy ONLY on a single-accelerator box (no gfx1151 GPU),
        # where one managed model serves both surfaces. On a dual-accelerator box
        # (the Z2) leave ai-chat unassigned so the user selects a GPU model later
        # (VD-007/VD-042) rather than the Assistant silently binding it to the NPU.
        inventory = (
            accelerators if accelerators is not None else discover_accelerators()
        )
        migrate_ai_chat = not names_a_gfx1151(inventory)

    # Resolve this deploy's endpoint only when it is needed to disambiguate a
    # same-vs-different-model decision - not on a first/only-consumer deploy.
    new_endpoint = ""
    if peers or active_chat_endpoint is not None:
        new_endpoint = _managed_local_endpoint(broker, credential_id)

    if active_chat_endpoint is not None:
        migrate_ai_chat = bool(new_endpoint) and active_chat_endpoint == new_endpoint
    if npu_tier:
        # VD-202 item 2 (VD-007): AI Chat never runs on the Assistant's NPU
        # model, not even on a fresh box with no GPU - the vault would refuse
        # it anyway, and asking would fail the Assistant's own deploy.
        migrate_ai_chat = False

    # One NPU runs one server, so a new NPU credential supersedes every older
    # NPU credential, whatever port it names: the old one was left pointing at a
    # dead port when a redeploy moved the server (W4d-D13 follow-on).
    stale = peers if npu_tier else _same_endpoint_peers(broker, peers, new_endpoint)

    broker.activate(credential_id, "deployment-agent")
    if migrate_ai_chat:
        broker.activate(credential_id, "ai-chat")
    _sweep_stale(broker, stale)
    return migrate_ai_chat


def activate_managed_chat(broker, credential_id: str) -> bool:
    """Make a healthy managed GPU model the AI-Chat default, and nothing else.

    The AI-Chat mirror of :func:`activate_managed_local`, and it exists as a
    separate surface for exactly one reason: the deployment-agent purpose must be
    left alone. On a dual-accelerator box (the Z2) the NPU Assistant holds
    ``deployment-agent`` and the GPU tier holds ``ai-chat``, and the two are
    deliberately independent - deploying the GPU AI-Chat model must move
    ``ai-chat`` onto the GPU endpoint WITHOUT stealing ``deployment-agent`` from
    the NPU Assistant. That is the VD-085 lineage inverted: where
    :func:`activate_managed_local` unconditionally claims ``deployment-agent``
    because that is the deploy surface's job, this surface never touches it,
    because the GPU model is not the Assistant's connection and never was.

    ``ai-chat`` is assigned UNCONDITIONALLY: this call IS the point where AI Chat
    moves onto the GPU (off the NPU or a hosted provider if it was there), so
    there is no "adopt only on a fresh box" clause - the caller only reaches here
    once the GPU server is healthy and its connection tested.

    The stale sweep is the SAME endpoint-keyed logic
    :func:`activate_managed_local` uses, factored into the shared helpers so the
    two cannot drift: only OTHER managed-local credentials that share THIS
    deploy's endpoint are removed (a redeploy of the same GPU model on the same
    loopback port). A different-endpoint credential - the NPU Assistant's - is an
    independent model and is never swept.

    Returns whether ``ai-chat`` was (re)assigned to this credential, which on
    this surface is always ``True``; the boolean mirrors
    :func:`activate_managed_local`'s return so both read the same at the call
    site.
    """
    peers = _managed_local_peers(_tiers(broker), credential_id)
    # The endpoint is only needed to key the sweep, so it is resolved only when
    # there is a peer to compare against - a first/only GPU deploy resolves
    # nothing, exactly as the deployment-agent surface does.
    new_endpoint = _managed_local_endpoint(broker, credential_id) if peers else ""
    stale = _same_endpoint_peers(broker, peers, new_endpoint)

    broker.activate(credential_id, "ai-chat")
    _sweep_stale(broker, stale)
    return True
