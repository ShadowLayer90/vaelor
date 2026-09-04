"""Permission-aware RAG reads and typed write proposals for custom agents."""

from __future__ import annotations

import json


def retrieve_agent_knowledge(store, actor: str, profile: dict, task: str):
    if store is None or "knowledge:read" not in profile.get("permissions", []):
        return []
    return store.retrieve(
        actor, task, profile.get("read_collection_ids", []), limit=6
    )


def attach_write_proposal(
    result: dict, profile: dict, task_id: str, store=None, actor: str = "",
):
    if "knowledge:write" not in profile.get("permissions", []):
        return result
    collection_id = profile.get("write_collection_id", "")
    if not collection_id:
        return result
    collection_name = ""
    if store is not None and actor:
        collection_name = next(
            (
                item["name"] for item in store.collections(actor)
                if item["id"] == collection_id
            ),
            "",
        )
    report = {
        "summary": result.get("summary", ""),
        "findings": result.get("findings", []),
        "recommendations": result.get("recommendations", []),
        "next_actions": result.get("next_actions", []),
    }
    return {
        **result,
        "proposed_writes": [{
            "type": "knowledge.document",
            "collection_id": collection_id,
            "collection_name": collection_name,
            "name": "Agent report {}.json".format(task_id[:12]),
            "media_type": "application/json",
            "content": json.dumps(report, indent=2, default=str),
            "approval_required": True,
            "executed": False,
        }],
    }
