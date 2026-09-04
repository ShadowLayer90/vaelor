"""Assignment and cleanup policy for Vaelor-managed local model credentials."""

from __future__ import annotations

import urllib.parse

from .credential_broker import CredentialError


PREFIX = "cred_managed_local_"


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


def _managed_local_peers(broker, credential_id: str) -> list:
    """Every OTHER Vaelor-managed local credential - the redeploy candidates.

    A managed-local credential carries the :data:`PREFIX`; a hosted or
    user-managed provider does not, so it is never a candidate for the
    endpoint-keyed sweep below. Shared by both activation surfaces because the
    candidate set is the same question - "what else did Vaelor generate?" -
    whichever consumer is being pointed at this deploy.
    """
    return [
        item_id for item in broker.list()
        if (item_id := str(item.get("id", ""))).startswith(PREFIX)
        and item_id != credential_id
    ]


def _same_endpoint_peers(broker, peers: list, endpoint: str) -> list:
    """The peers that share THIS deploy's connection endpoint.

    The discriminator between "the same local model, re-registered" and "a
    different, independent local model" is the endpoint (see :func:`_endpoint_of`),
    so only same-endpoint peers are a redeploy of this server and stale. An
    unresolved endpoint (``""``) matches nothing: that is the primary data-loss
    guard, because a genuine endpoint never collapses to the empty string, so an
    unreadable peer can never be mistaken for a same-endpoint one.
    """
    return [
        peer for peer in peers
        if endpoint and _managed_local_endpoint(broker, peer) == endpoint
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


def activate_managed_local(broker, credential_id: str) -> bool:
    """Make a healthy managed model the default without replacing user providers.

    A fresh appliance has no AI Chat assignment, so the first managed model
    becomes its default. Existing hosted or user-managed compatible endpoints
    remain selected. Re-deploying THE SAME local model server (same endpoint)
    migrates both consumers and removes the obsolete generated credential.

    VD-085-style correction: the original policy assumed a single-model
    appliance (one Raspberry Pi, one managed model), so it treated EVERY other
    managed-local credential as stale and stole ``ai-chat`` onto whatever model
    was last deployed. On a multi-accelerator box (Z2: an NPU Assistant on one
    port and an independent GPU AI-Chat model on another) that deleted the GPU
    model's credential and hijacked its ``ai-chat`` purpose. The fix keys both
    the stale sweep and the ``ai-chat`` migration off the connection ENDPOINT so
    an independent model on a different endpoint is left untouched. The
    single-model Pi case is unchanged: its lone managed model shares the endpoint
    it is replacing, so both consumers still move and the old credential is still
    removed.

    The just-deployed model always becomes the ``deployment-agent`` connection;
    that is this surface's job and is unconditional.
    """
    # Other managed-local credentials are re-registration candidates only when
    # they share this deploy's endpoint. A different endpoint is an independent
    # model (the GPU AI Chat) and must survive.
    peers = _managed_local_peers(broker, credential_id)

    # Whether ai-chat should follow this deploy depends on what it points at now.
    active_chat_endpoint = None
    migrate_ai_chat = False
    try:
        active_chat = broker.resolve_active("ai-chat")
        if str(active_chat.get("credential_id", "")).startswith(PREFIX):
            # A managed-local model holds ai-chat: migrate only if it is THIS
            # same endpoint being redeployed. A different-endpoint managed model
            # (the GPU one) keeps ai-chat.
            active_chat_endpoint = _endpoint_of(active_chat)
        # A user/hosted ai-chat assignment is never migrated (unchanged).
    except CredentialError:
        # No credential is active for ai-chat: fresh appliance, adopt the default.
        migrate_ai_chat = True

    # Resolve this deploy's endpoint only when it is needed to disambiguate a
    # same-vs-different-model decision - not on a first/only-consumer deploy.
    new_endpoint = ""
    if peers or active_chat_endpoint is not None:
        new_endpoint = _managed_local_endpoint(broker, credential_id)

    if active_chat_endpoint is not None:
        migrate_ai_chat = bool(new_endpoint) and active_chat_endpoint == new_endpoint

    stale = _same_endpoint_peers(broker, peers, new_endpoint)

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
    peers = _managed_local_peers(broker, credential_id)
    # The endpoint is only needed to key the sweep, so it is resolved only when
    # there is a peer to compare against - a first/only GPU deploy resolves
    # nothing, exactly as the deployment-agent surface does.
    new_endpoint = _managed_local_endpoint(broker, credential_id) if peers else ""
    stale = _same_endpoint_peers(broker, peers, new_endpoint)

    broker.activate(credential_id, "ai-chat")
    _sweep_stale(broker, stale)
    return True
