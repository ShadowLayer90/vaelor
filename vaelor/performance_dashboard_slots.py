"""Each machine's chart colour slot, kept for as long as its history is (VD-147 S4, spec §3.4 N-S3).

Colour follows the machine, never its rank: the controller is slot 1 and a
worker takes the lowest free slot the first time it appears. A removed
machine's slot is released, not freed: it is free again only once the
telemetry retention has passed since the removal, because until then the
machine's history is still drawn in that colour and a newcomer must not
inherit it. So removing a middle worker repaints no one.

The table lives in its own small SQLite file beside the cluster store (the
node record is deleted at removal; the slot must outlive it). A fourth
machine and beyond get no generated colour: the chart's chooser shows them.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from typing import Dict, Iterable, Optional

from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .runtime_paths import state_path
from .telemetry_store import DEFAULT_RETENTION_DAYS

#: The slots that have a colour token; a machine beyond them has a slot number
#: but is drawn only when chosen.
COLOURED_SLOTS = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS chart_slots (
    node_id TEXT PRIMARY KEY,
    slot INTEGER NOT NULL UNIQUE,
    released_at REAL
);
"""


#: Every held slot, a released machine's included.
_HELD = "SELECT node_id, slot FROM chart_slots"


class ChartSlots:
    def __init__(self, path: Optional[str] = None, retention_days: int = DEFAULT_RETENTION_DAYS) -> None:
        self.path = path or str(state_path("cluster/chart-slots.sqlite3"))
        self.retention_seconds = int(retention_days) * 86400
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        """A connection with the table in place; opened on first use, never at import."""
        connection = sqlite3.connect(self.path, timeout=15)
        if not self._ready:
            connection.executescript(SCHEMA)
            connection.commit()
            self._ready = True
        return connection

    def assign(self, node_ids: Iterable[str], now: Optional[float] = None) -> Dict[str, int]:
        """``{node id: slot}`` for every node, assigning the lowest free slot to a newcomer."""
        moment = time.time() if now is None else float(now)
        wanted = [str(node) for node in node_ids if node]
        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM chart_slots WHERE released_at IS NOT NULL AND released_at < ?",
                (moment - self.retention_seconds,),
            )
            held = {row[0]: row[1] for row in connection.execute(_HELD)}
            # A held machine no longer in the fleet was removed, however it was
            # removed (pass-3 review S4-B1: the executor's removal job never
            # called release). Its slot is kept until its history ages out.
            connection.execute(
                "UPDATE chart_slots SET released_at = ? WHERE released_at IS NULL AND node_id NOT IN ({})".format(
                    ",".join("?" for _ in wanted) or "''"),
                (moment, *wanted),
            )
            for node in wanted:
                if node in held:
                    connection.execute("UPDATE chart_slots SET released_at = NULL WHERE node_id = ?", (node,))
                    continue
                # Slot 1 is the controller's alone; a worker takes the lowest other free slot.
                slot = 1 if node == CONTROLLER_PLACEMENT_ID else self._lowest_free(set(held.values()) | {1})
                connection.execute("INSERT INTO chart_slots(node_id, slot) VALUES (?, ?)", (node, slot))
                held[node] = slot
            connection.commit()
        return {node: held[node] for node in wanted}

    def held(self) -> Dict[str, int]:
        """Every slot still held, a removed machine's included: history keeps its colour."""
        with closing(self._connect()) as connection:
            return {row[0]: row[1] for row in connection.execute(_HELD)}

    def release(self, node_id: str, now: Optional[float] = None) -> None:
        """A machine was removed: keep its slot until its history has aged out."""
        moment = time.time() if now is None else float(now)
        with closing(self._connect()) as connection:
            connection.execute("UPDATE chart_slots SET released_at = ? WHERE node_id = ?", (moment, str(node_id)))
            connection.commit()

    @staticmethod
    def _lowest_free(taken: set) -> int:
        slot = 1
        while slot in taken:
            slot += 1
        return slot
