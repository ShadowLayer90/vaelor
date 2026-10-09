"""A worker replica's gate: the one LAN door in front of a loopback vLLM replica.

Split out of `gpu_pool_runtime` (VD-143) when the root-rendered unit verbs took
that module past the 1,000-line ceiling, and cohesive on its own: the gate is
the one unit this tier writes that is NOT a vLLM container. It is the LLM
Server proxy's nginx image, argv and config shape (VD-129, the amendment),
rendered by `llm_server_proxy`, beside a root ``0600`` config that carries the
cluster key. `GpuPoolRuntime` inherits these methods, so every caller still
asks the runtime for them.

**Worker-only, and now stated as a refusal.** The controller's replica sits
behind the balancer on loopback and has no gate, so the root bridge carries no
gate shape at all: no config write, no ``tee``, no gate unit (VD-143). A gate
asked of the controller's transport is refused by name rather than by a
policy error about an ``install`` argv.
"""

from __future__ import annotations

from . import gpu_render_ledger as render_ledger
from .bridge_transport import root_renders_units
from .gpu_pool_units import (
    DOCKER, GATE_CONFIG_ROOT, GATE_CONFIG_ROOT_MODE, GATE_ROLE, container_name, gate_config_path,
    unit_name, unit_text,
)
#: A worker's replica gate IS the LLM Server proxy's image, argv and config
#: shape, rendered by that module: one nginx door, two places it stands, so the
#: two cannot drift on the Bearer check or the mount.
from .llm_server_proxy import PROXY_IMAGE, proxy_container_command, render_gate_config


class GateMixin:
    """`start_gate`, `render_gate`, `rekey_gate` and the ``0600`` config write they share."""

    def start_gate(
        self, transport, *, name: str, listen_host: str, port: int, api_key: str,
    ) -> str:
        """Launch a WORKER replica's gate: the product's nginx image, as a unit.

        The one LAN door on a worker (VD-129, the amendment). It listens on
        the worker's cluster address on the replica port - the SAME port the
        replica holds on loopback; two binds on two addresses coexist, so
        the balancer's ``<address>:<port>`` upstream reaches the gate and no
        second port per role is needed - requires the cluster key on ``/v1/``,
        forwards ``/health`` unkeyed, and answers 404 to everything else
        (`llm_server_proxy.render_gate_config`).

        The config, with the key in it, is landed FIRST as a root-owned
        ``0600`` file (:meth:`_write_gate_config`) and mounted read-only; the
        unit is the same ``docker run`` shape the LLM Server proxy uses, in
        the foreground under ``Restart=on-failure`` with
        ``StartLimitIntervalSec=0`` like the replica it fronts. The key is on
        no argv and in no unit file. `stop_unit` removes the config with the
        unit. The listen host goes through the runtime's own address rule: a
        private, non-loopback IPv4 - a gate on loopback would front nothing
        and one on every interface would collide with the replica's bind.

        **Worker-only.** The controller's replica sits behind the balancer on
        loopback and never has a gate, so the root bridge has no gate shape
        (VD-143) and a gate asked of it is refused here, by name.
        """
        if root_renders_units(transport):
            raise ValueError("The controller's replica has no gate; only a worker's does.")
        unit, content, config = self.render_gate(
            name=name, listen_host=listen_host, port=port, api_key=api_key,
        )
        path = self._write_gate_config(transport, unit, config)
        try:
            self._write_unit_text(transport, unit, content, oneshot=False)
        except Exception:
            # The config carries the cluster key: a gate that never got its
            # unit must not leave it on the node. Best effort; the start's
            # own failure is what the caller hears.
            try:
                transport.run(["rm", "-f", path], sudo=True)
            except Exception:  # noqa: BLE001 - the original error wins
                pass
            raise
        # What this gate was rendered from, for the upgrade's render check.
        render_ledger.note(transport, render_ledger.GATE_KIND, {
            "name": name, "listen_host": listen_host, "port": port,
        })
        return unit

    def render_gate(
        self, *, name: str, listen_host: str, port: int, api_key: str,
    ):
        """``(unit, unit text, config)`` of a worker's gate - rendered, not written.

        The one rendering `start_gate` writes, and the one the upgrade's render
        check (`gpu_render_ledger`) re-renders with a placeholder key: the key
        lives only in the config, never in the unit text.
        """
        deployment = self._name(name)
        unit = unit_name(deployment, GATE_ROLE)
        container = container_name(deployment, GATE_ROLE)
        published = self._port(port)
        config = render_gate_config(
            listen_host=self._address(listen_host), port=published,
            api_keys=[api_key],
        )
        command = proxy_container_command(
            listen_port=published, api_keys=[api_key], docker=DOCKER,
            image=PROXY_IMAGE, container_name=container,
            config_host_file=gate_config_path(unit), foreground=True,
        )
        content = unit_text(
            description=f"Vaelor GPU inference gate ({deployment})",
            container_name=container, exec_start=" ".join(command),
            unlimited_restarts=True,
        )
        return unit, content, config

    def rekey_gate(
        self, transport, *, name: str, listen_host: str, port: int,
        api_keys,
    ) -> str:
        """Re-render a WORKER gate's config with a new key SET and restart it live.

        The rotate half of :meth:`start_gate` (design F3e, step b/d): the gate's
        unit is unchanged, so only its root ``0600`` config file is re-rendered
        - through the SAME `render_gate_config` byte-exact multi-key gate the
        deploy uses - and the container is ``systemctl restart``ed so nginx
        re-reads it. The listen host and port go through the same address/band
        rules as the initial start, so a re-key cannot widen the bind. Called
        only for a worker's gate; the controller's loopback replica has none.
        """
        deployment = self._name(name)
        unit = unit_name(deployment, GATE_ROLE)
        published = self._port(port)
        config = render_gate_config(
            listen_host=self._address(listen_host), port=published,
            api_keys=list(api_keys),
        )
        self._write_gate_config(transport, unit, config)
        transport.run(["systemctl", "restart", unit], sudo=True)
        return unit

    @staticmethod
    def _write_gate_config(transport, unit: str, config: str) -> str:
        """Land a gate's config on its node, ``0600`` from the first byte.

        The file is pre-created empty at ``0600`` from ``/dev/null``, then the
        config is streamed into that existing file with ``tee`` - which does
        not change an existing file's mode - so it is ``0600`` from the first
        byte AND a config of any size lands reliably. ``install -m 0600
        /dev/stdin <path>`` was replaced because it intermittently fails to
        open ``/dev/stdin`` over an SSH exec channel once the payload passes
        about a kilobyte (the F3e rotate's multi-key config), returning
        ``install: No such file or directory``; ``tee`` reads the inherited
        stdin descriptor directly rather than opening the ``/dev/stdin`` named
        path, so it is immune to that size/EOF race. The config, and the key
        in it, travel on stdin exactly as a unit body does, so they appear on
        no argv and in no process listing. The directory is Vaelor's alone, so
        ``install -d`` re-moding it is harmless.
        """
        path = gate_config_path(unit)
        transport.run(["install", "-d", "-m", GATE_CONFIG_ROOT_MODE, GATE_CONFIG_ROOT], sudo=True)
        transport.run(["install", "-m", "0600", "/dev/null", path], sudo=True)
        transport.run(["tee", path], sudo=True, stdin_text=config)
        return path
