"""The tool-enforcement primitives a deployed cluster Agent's tool use passes.

This is the security spine of the F4b agent runtime (design section 5, items
S3/B1/B2/B3), built and tested standalone before the F4b-ii server wires it. It
holds no server, launches no job, and reaches no node: it is pure logic wrapping
existing collaborators, so the whole enforcement surface is reviewable in one
short module.

Three primitives live here:

* :class:`AgentToolRegistry` (S3) wraps an :class:`assistant_tools.
  AssistantToolRegistry` so that a runtime call CANNOT omit the authorisation
  arguments. The plain ``run`` defaults ``administrator=True`` and treats
  ``granted_scopes=None`` as "no scope gate at all"; a call site that forgets
  either loses its whole boundary. This wrapper removes the choice: it accepts
  neither an ``administrator`` argument nor a ``granted_scopes`` override, always
  forwarding ``administrator=False`` and a concrete, construction-time scope set.
* :func:`build_mcp_grant_gate` (B1/B2) turns one agent version's ``mcp_grants``
  plus the catalog into a standing-consent gate for the agent's OWN outbound
  MCP client. The gate intersects each grant with the catalog's *current*
  ``approved_tools`` so a grant can never exceed what the catalog globally
  approved, denies by default, and refuses an out-of-grant qualified tool name
  before ``mcp_client`` is ever dialled.
* :func:`assert_read_only` (B3) asserts a deployed agent's ``permissions`` axis
  (the ``knowledge:write`` / ``workloads:propose`` acting axis, distinct from
  tool scopes) is empty, documenting that the runtime offers no mutating tool.

The module has nothing to do with the INBOUND ``VaelorMcpServer`` surface. It
imports no inbound type and mints no ``EXTERNAL_SCOPE``; keeping that boundary
structural is deliberate, so a reader never has to trace a call to be sure this
enforcement cannot leak into an inbound path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from .agent_tool_loop import build_tool_specifications
from .mcp_client import EXTERNAL_SCOPE, NAMESPACE

LOGGER = logging.getLogger(__name__)


class AgentRuntimeGateError(ValueError):
    """A safe, user-presentable agent-runtime enforcement error."""


class ToolNotGrantedError(AgentRuntimeGateError):
    """Raised by the MCP call seam when a tool is outside the effective grant."""


# Sentences written at one call site each still keep a single home so a later
# edit cannot let two copies drift; each is phrased to differ from the appliance
# loop's own refusals rather than share a literal with another module.
_SCOPES_REQUIRED = (
    "granted_scopes must be a concrete iterable of scope names, never None."
)
_SCOPES_NOT_ITERABLE = (
    "granted_scopes must be an iterable of scope strings, not a bare string."
)
_TOOL_NOT_GRANTED = (
    "tool_not_granted: {} is not in this agent's approved MCP tool set."
)
_NAME_NOT_QUALIFIED = (
    "tool_not_granted: an outbound MCP tool is named mcp.<server>.<tool>."
)
_PERMISSIONS_NOT_EMPTY = (
    "A deployed agent offers only read-only tools; it may hold no acting "
    "permission, but was given: {}."
)


class AgentToolRegistry:
    """The hardened wrapper: a runtime call can never omit its authorisation.

    Constructed once per deployed-agent run around an inner registry that
    exposes the :class:`assistant_tools.AssistantToolRegistry` ``run``/``names``
    surface (accepted by duck type, so a fake in a test works the same). This
    object is the ONLY sanctioned way the runtime reaches ``inner.run``: it takes
    no ``administrator`` parameter and no ``granted_scopes`` override, so
    ``administrator=True`` and ``granted_scopes=None`` - the two ways the inner
    ``run`` disables its own gate - are structurally unreachable through it.

    ``granted_scopes`` is required, non-None, and iterable (a bare string is
    refused, so a single scope written as ``"system:read"`` cannot silently
    become a set of characters). An EMPTY set is allowed: an agent granted no
    Vaelor tools is a valid, fully locked-down agent.
    """

    def __init__(self, inner: Any, granted_scopes: Iterable[str]):
        if not callable(getattr(inner, "run", None)) or not callable(
            getattr(inner, "names", None)
        ):
            raise AgentRuntimeGateError(
                "The wrapped registry must expose callable run and names."
            )
        if granted_scopes is None:
            raise AgentRuntimeGateError(_SCOPES_REQUIRED)
        if isinstance(granted_scopes, (str, bytes)):
            raise AgentRuntimeGateError(_SCOPES_NOT_ITERABLE)
        try:
            scopes = frozenset(str(scope) for scope in granted_scopes)
        except TypeError as error:
            raise AgentRuntimeGateError(_SCOPES_REQUIRED) from error
        self._inner = inner
        self._scopes: FrozenSet[str] = scopes

    @property
    def scopes(self) -> FrozenSet[str]:
        """The concrete, immutable scope set every call is bounded to."""
        return self._scopes

    def run(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        actor: Optional[str] = None,
        web_access: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one tool with the boundary forced on, never optional.

        ``administrator`` is hard-wired to ``False`` and the concrete scope set
        is always supplied, so neither can be weakened by a call site.
        """
        return self._inner.run(
            name,
            arguments,
            actor=actor,
            administrator=False,
            granted_scopes=self._scopes,
            web_access=web_access,
        )

    def names(self) -> Any:
        """Delegate the inner registry's registered tool names."""
        return self._inner.names()

    def specifications(self, catalog: Any, web_access: Any) -> List[Dict[str, Any]]:
        """The read-only tool offering, bounded to this agent's granted scopes.

        Delegates to :func:`agent_tool_loop.build_tool_specifications`, which
        already excludes any mutating or approval-gated tool and any tool whose
        scope was not granted - so the offered set is read-only by construction.
        """
        return build_tool_specifications(catalog, self._scopes, web_access)


class McpGrantGate:
    """Standing-consent gate for a deployed agent's OWN outbound MCP client.

    Built by :func:`build_mcp_grant_gate` from the agent version's grants and the
    catalog ALONE. It holds the effective ``(server_name, tool)`` set - each
    grant already intersected with the catalog's current ``approved_tools`` - and
    denies everything else. It is meant to be attached as the ``approval``
    callback of the agent's own :class:`mcp_client.ExternalMcpTools`, where
    attaching the server to the agent (through the F6 admin review) IS the
    consent: there is no interactive human at call time, so this standing gate
    replaces the per-run human approval callback.

    Boundary: this gate must NEVER be passed to ``VaelorMcpServer(external=...)``
    nor into any inbound path, and it never mints ``EXTERNAL_SCOPE`` - it only
    answers whether the agent may reach one OUTBOUND tool. The module keeps that
    boundary structural by importing nothing from the inbound server.

    Server name uniqueness: the effective set is keyed by ``(server_name, tool)``
    because ``mcp_client.call`` passes the server NAME (not id) to the approval
    callback. The catalog enforces a unique name per server, so this mapping is
    unambiguous; the internal ``_by_server`` map is keyed by id and records the
    name it resolved to, so a future non-unique-name catalog is still auditable.
    """

    def __init__(
        self,
        effective_pairs: Iterable[Tuple[str, str]],
        by_server: Mapping[str, Dict[str, Any]],
        dropped: Iterable[Dict[str, str]],
    ):
        self._pairs: FrozenSet[Tuple[str, str]] = frozenset(
            (str(server), str(tool)) for server, tool in effective_pairs
        )
        self._by_server: Dict[str, Dict[str, Any]] = dict(by_server)
        self._dropped: List[Dict[str, str]] = list(dropped)

    @property
    def effective_pairs(self) -> FrozenSet[Tuple[str, str]]:
        """Every ``(server_name, tool)`` this agent may call, deny-by-default."""
        return self._pairs

    @property
    def dropped_grants(self) -> List[Dict[str, str]]:
        """Grant entries the catalog did not (or no longer) approve, for logs.

        Never widening: a dropped entry is a grant the effective set excludes;
        it is surfaced only so an operator can see WHY a named tool is unreachable.
        """
        return list(self._dropped)

    def approves(self, server_name: str, tool: str) -> bool:
        """Whether ``(server_name, tool)`` is in the intersected effective set."""
        return (str(server_name), str(tool)) in self._pairs

    def approval(self, server_name: str, tool: str, actor: Optional[str] = None) -> bool:
        """The standing-consent callback ``mcp_client.ExternalMcpTools`` calls.

        Returns True ONLY for an effective ``(server_name, tool)`` pair; the
        ``actor`` is accepted to match the callback signature but does not widen
        the answer - the consent is the admin's attachment, not a per-actor one.
        """
        return self.approves(server_name, tool)

    def _split(self, qualified_name: str) -> Tuple[str, str]:
        text = str(qualified_name)
        if not text.startswith(NAMESPACE):
            raise ToolNotGrantedError(_NAME_NOT_QUALIFIED)
        server_name, separator, tool = text[len(NAMESPACE):].partition(".")
        if not separator or not server_name or not tool:
            raise ToolNotGrantedError(_NAME_NOT_QUALIFIED)
        return server_name, tool

    def guard(self, qualified_name: str) -> Tuple[str, str]:
        """Refuse an out-of-grant qualified tool BEFORE any ``mcp_client.call``.

        ``qualified_name`` is the ``mcp.<server>.<tool>`` name ``mcp_client``
        namespaces with. Raises :class:`ToolNotGrantedError` when the name is
        malformed or its pair is not in the effective set, mirroring the loop's
        ``tool_not_granted`` hard refusal. Returns the resolved
        ``(server_name, tool)`` on success.
        """
        server_name, tool = self._split(qualified_name)
        if (server_name, tool) not in self._pairs:
            raise ToolNotGrantedError(_TOOL_NOT_GRANTED.format(qualified_name))
        return server_name, tool

    def call(
        self, client: Any, qualified_name: str, arguments: Dict[str, Any], session: Any
    ) -> Any:
        """Guard, then delegate to the injected outbound client's ``call``.

        Belt-and-suspenders with :meth:`approval`: :meth:`guard` refuses an
        out-of-grant tool here, and the client's own approval check (this same
        gate) refuses it independently inside ``call`` - both must pass.
        """
        self.guard(qualified_name)
        return client.call(qualified_name, arguments, session)

    @classmethod
    def from_pairs(cls, effective_pairs: Iterable[Tuple[str, str]]) -> "McpGrantGate":
        """Build a gate directly from an injected effective ``(server, tool)`` set.

        The F4b-ii agent server receives the effective pairs already computed
        by the control plane (grant INTERSECT catalog) and holds no catalog of
        its own, so it cannot call :func:`build_mcp_grant_gate`. This keeps the
        same deny-by-default gate keyed by server NAME - which is what the
        outbound ``mcp_client`` passes to :meth:`approval` - with no
        dropped-grant log, since the narrowing already happened elsewhere.
        """
        pairs = [(str(server), str(tool)) for server, tool in effective_pairs]
        by_server: Dict[str, Dict[str, Any]] = {}
        for server_name, tool in pairs:
            entry = by_server.setdefault(server_name, {"name": server_name, "tools": []})
            if tool not in entry["tools"]:
                entry["tools"].append(tool)
        return cls(pairs, by_server, [])


def _grant_tools(grant: Mapping[str, Any]) -> List[str]:
    raw = grant.get("allowed_tools")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return []
    tools: List[str] = []
    for item in raw:
        tool = str(item).strip()
        if tool and tool not in tools:
            tools.append(tool)
    return tools


def _record_dropped(
    dropped: List[Dict[str, str]], server_id: str, tools: Iterable[str], reason: str
) -> None:
    for tool in tools:
        dropped.append({"server_id": server_id, "tool": tool, "reason": reason})


def build_mcp_grant_gate(
    mcp_grants: Iterable[Mapping[str, Any]],
    catalog_lookup: Callable[[str], Optional[Mapping[str, Any]]],
) -> McpGrantGate:
    """Compute the effective outbound-tool gate from grants and the catalog.

    ``mcp_grants`` is this agent version's ``[{server_id, allowed_tools[]}]`` (the
    F4a store's shape). ``catalog_lookup`` resolves one ``server_id`` to its
    current record ``{id, name, enabled, approved_tools[]}`` or ``None`` - the
    ``McpCatalogStore.get`` shape.

    Per server the effective tools are ``allowed_tools`` INTERSECTED with the
    catalog's current ``approved_tools`` ([[same-allowlist-is-not-same-boundary]]):
    a grant naming a tool the catalog does not approve is dropped, never widened.
    A grant for an unknown or DISABLED server contributes nothing. The result is
    keyed by ``(server_name, tool)`` for ``mcp_client``'s approval callback.
    """
    pairs: set = set()
    by_server: Dict[str, Dict[str, Any]] = {}
    dropped: List[Dict[str, str]] = []
    for grant in mcp_grants or []:
        if not isinstance(grant, Mapping):
            continue
        server_id = str(grant.get("server_id") or "").strip()
        if not server_id:
            continue
        requested = _grant_tools(grant)
        record = catalog_lookup(server_id)
        if not isinstance(record, Mapping):
            _record_dropped(dropped, server_id, requested, "server_unknown")
            continue
        if not record.get("enabled"):
            _record_dropped(dropped, server_id, requested, "server_disabled")
            continue
        server_name = str(record.get("name") or "").strip()
        approved = {
            str(item).strip()
            for item in (record.get("approved_tools") or [])
            if str(item).strip()
        }
        effective: List[str] = []
        for tool in requested:
            if tool in approved and server_name:
                effective.append(tool)
                pairs.add((server_name, tool))
            else:
                dropped.append({
                    "server_id": server_id, "tool": tool,
                    "reason": "not_catalog_approved",
                })
        entry = by_server.setdefault(server_id, {"name": server_name, "tools": []})
        for tool in effective:
            if tool not in entry["tools"]:
                entry["tools"].append(tool)
    return McpGrantGate(pairs, by_server, dropped)


def assert_read_only(permissions: Optional[Iterable[str]]) -> None:
    """B3: a deployed agent holds no acting permission - refuse a non-empty set.

    The ``permissions`` axis (``knowledge:write`` / ``workloads:propose``) is
    distinct from tool scopes and grants the power to MUTATE. A deployed cluster
    agent offers only read-only tools, so this raises on ANY permission and
    passes only on the empty case (``None`` or an empty collection). It is a
    belt to :class:`AgentToolRegistry`'s ``administrator=False`` + concrete-scope
    construction, which already keeps the OFFERED tool set read-only.
    """
    if permissions is None:
        return
    held = [str(item).strip() for item in permissions if str(item).strip()]
    if held:
        raise AgentRuntimeGateError(
            _PERMISSIONS_NOT_EMPTY.format(", ".join(sorted(set(held)))[:200])
        )


class _OutboundMcpSession:
    """The agent's OWN outbound MCP session, minted only inside dispatch.

    It carries exactly ``external:mcp`` - the one scope a Vaelor grant can never
    supply, so the outbound ``mcp_client`` admits the call - plus the acting
    ``actor`` the approval callback's signature expects. It is built here, handed
    straight to the outbound client, and used nowhere else: it is deliberately
    NOT a :class:`VaelorMcpServer` inbound session, and nothing in this module
    passes it to an inbound path. ``__slots__`` keep it from growing a field.
    """

    __slots__ = ("scopes", "actor")

    def __init__(self, actor: Optional[str]) -> None:
        self.scopes: FrozenSet[str] = frozenset({EXTERNAL_SCOPE})
        self.actor = actor


class AgentToolset:
    """The unified, gated tool surface a deployed agent's model reaches (design R1).

    Wraps two offerings behind one dispatch so the F4b-ii server has a single
    seam, and routes every model tool call by namespace so nothing outside the
    grant can be reached:

    * ``specifications`` merges the hardened :class:`AgentToolRegistry`'s
      read-only Vaelor specs (bounded to the granted scopes, mutating and
      approval-gated tools already excluded) with the injected MCP specs - the
      OpenAI function specs the control plane computed for the effective
      ``mcp.<server>.<tool>`` pairs at deploy. This toolset never computes those.
    * ``dispatch`` sends an ``mcp.`` name to the OUTBOUND client through
      :meth:`McpGrantGate.call` (guard first, then the client's own
      gate-backed approval - denied twice for an out-of-grant tool), minting the
      outbound ``external:mcp`` session HERE and only here; every other name goes
      to :meth:`AgentToolRegistry.run`, which forces ``administrator=False`` and
      the concrete scope set. A malformed ``mcp.`` name is denied by the guard;
      an unknown Vaelor name is denied by the inner registry - never a silent pass.

    The ``mcp_client`` MUST have been constructed with this gate's ``approval`` as
    its callback, so the guard here and the approval there are independent denials.
    """

    def __init__(
        self,
        vaelor_registry: AgentToolRegistry,
        mcp_gate: McpGrantGate,
        mcp_client: Any,
        mcp_specs: Iterable[Mapping[str, Any]],
        *,
        web_access: Optional[Dict[str, Any]] = None,
    ):
        self._vaelor = vaelor_registry
        self._gate = mcp_gate
        self._mcp_client = mcp_client
        self._mcp_specs: List[Dict[str, Any]] = [dict(spec) for spec in (mcp_specs or [])]
        self._web_access = web_access

    def specifications(self, catalog: Any, web_access: Any) -> List[Dict[str, Any]]:
        """Merge the granted read-only Vaelor specs with the injected MCP specs.

        The union is the entire set the model may call: the Vaelor half is
        already read-only and scope-bounded by construction, the MCP half is
        exactly the effective set the control plane injected.
        """
        merged = list(self._vaelor.specifications(catalog, web_access))
        merged.extend(dict(spec) for spec in self._mcp_specs)
        return merged

    def dispatch(
        self, name: str, arguments: Optional[Dict[str, Any]], actor: Optional[str]
    ) -> Any:
        """Route one model tool call by namespace, denying anything unrecognised."""
        text = str(name)
        if text.startswith(NAMESPACE):
            # Mint the outbound EXTERNAL_SCOPE session HERE, for this call only,
            # and hand it to the guarded outbound client - never an inbound path.
            session = _OutboundMcpSession(actor)
            return self._gate.call(self._mcp_client, text, dict(arguments or {}), session)
        return self._vaelor.run(
            text, dict(arguments or {}), actor=actor, web_access=self._web_access
        )
