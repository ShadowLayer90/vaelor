"""HTTP routes for the agent-facing skills/plugins library and attachments (F2).

These routes administer the skill manifests in the library. Reads are
``operator``; every mutation is ``administrator`` + CSRF + audit, matching the
conventions in ``api_mcp_catalog_routes`` and ``api_integration_routes``. The
library routes are what the Skills tab calls (``frontend/src/lib/skillsLibrary.ts``).

What actually hands a skill to a running agent is NOT this module: a cluster
agent deploy pins skill ids on its deployment record, and
``agent_pool_operations`` resolves each against the live library at deploy and
relaunch, refusing (or, on relaunch, dropping) any skill that is not read-only.

The ``/skills/attachments`` routes are a separate, currently UNUSED record: they
write and revoke per-target attach rows in ``SkillAttachmentStore`` and return a
review view (the grants and broker credential ids a skill assumes, ids only,
never a secret value). No screen calls these routes and no runtime reads the
attach rows - nothing asks an administrator to approve through them, and a row
here neither grants nor withholds anything from a deployed agent. They are kept
because the store and its tests exist; treat them as an API-only record, not as
the approval surface.

The GitHub import is a preview: it lists a repository's ``SKILL.md`` files and
marks, per file, every reason the register route would refuse it - computed by
the library's own register rules (``SkillsLibraryStore.refusals``), not a copy -
so the operator surface can leave refused files unselected and say why.
"""

from __future__ import annotations

from flask import g, request

from .api_common import ApiContext, payload
from .skills_github_import import import_repo_skills
from .skills_library import SkillsLibraryError, register_refusals


def register_skills_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    require_auth = context.require_auth
    security = context.security
    callbacks = context.callbacks

    def library():
        return callbacks.get("skills_library")

    def attachments():
        return callbacks.get("skill_attachments")

    def _unavailable():
        return payload(
            error={
                "code": "skills_library_unavailable",
                "message": "The skills library is unavailable.",
            },
            status=503,
        )

    def _audit(action: str, outcome: str, target: str = "", **details):
        security.audit(
            g.auth_session.username, action, outcome,
            target=str(target)[:64], remote_addr=request.remote_addr or "",
            details=details or None,
        )

    def _screen_import(skills):
        """Mark each previewed skill with every reason register would refuse it.

        The field rules and the taken-name rule come from the library itself;
        the one rule added here is the same unique-name rule applied across the
        batch, since two files naming one skill cannot both be registered.
        """
        store = library()
        first_path_by_name = {}
        for skill in skills:
            fields = {
                "name": skill.get("name", ""),
                "kind": "skill",
                "description": skill.get("description", ""),
                "instructions": skill.get("instructions", ""),
            }
            reasons = store.refusals(**fields) if store is not None else register_refusals(**fields)
            name = str(fields["name"]).strip()
            earlier = first_path_by_name.get(name)
            if earlier is not None:
                reasons.append(
                    "Another file in this repository, {}, already uses this name, "
                    "and the library keeps one skill per name.".format(earlier)
                )
            elif not reasons:
                # Only an importable file claims its name for the batch.
                first_path_by_name[name] = skill.get("path", "")
            skill["refusals"] = reasons
        return skills

    def _skill_view(item):
        """A manifest as an administrator sees it. No secret ever appears; the
        credentials are broker ids, and the grants are scope/permission names."""
        return {
            "id": item["id"],
            "name": item["name"],
            "description": item["description"],
            "instructions": item.get("instructions", ""),
            "kind": item["kind"],
            "enabled": item["enabled"],
            "grants": list(item["grants"]),
            "credentials": list(item["credentials"]),
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
        }

    # -- library manifests ------------------------------------------------
    @blueprint.get("/skills/library")
    @require_auth("operator")
    def skills_library_list():
        store = library()
        if store is None:
            return payload({"skills": [], "configured": False})
        return payload({
            "configured": True,
            "skills": [_skill_view(item) for item in store.list()],
            # ACC-078: what attaching to a deployed agent actually does, as
            # `agent_skill_surface` does it - never "does not widen".
            "note": (
                "A manifest names the capabilities a skill assumes. Attaching a "
                "skill to a deployed cluster agent adds the skill's read scopes "
                "to what that agent can read, and sends the skill's name, "
                "description and instructions with the agent's guidance, within "
                "a size limit the deploy review reports. A skill with an acting "
                "permission or a credential cannot be attached."
            ),
        })

    @blueprint.post("/skills/library")
    @require_auth("administrator", csrf=True)
    def skills_library_register():
        store = library()
        if store is None:
            return _unavailable()
        body = request.get_json(silent=True) or {}
        try:
            created = store.register(
                name=body.get("name", ""),
                kind=body.get("kind", ""),
                description=body.get("description", ""),
                instructions=body.get("instructions", ""),
                grants=body.get("grants", []),
                credentials=body.get("credentials", []),
                enabled=bool(body.get("enabled", True)),
            )
        except SkillsLibraryError as error:
            _audit("skills.library.register", "rejected", str(body.get("name", ""))[:64])
            return payload(
                error={"code": "skills_library_rejected", "message": str(error)},
                status=400,
            )
        _audit("skills.library.register", "success", created["id"], kind=created["kind"])
        return payload(_skill_view(created), status=201)

    @blueprint.post("/skills/library/import/github")
    @require_auth("administrator", csrf=True)
    def skills_library_import_github():
        # Preview only: fetch + parse a repo's SKILL.md files and hand the
        # admin {name, description, instructions} to review, then register the
        # chosen ones through the register route (which screens each). The
        # optional token is used for the fetch and is never stored or audited.
        body = request.get_json(silent=True) or {}
        owner = str(body.get("owner", ""))
        repo = str(body.get("repo", ""))
        ref = body.get("ref") or None
        token = body.get("token") or None
        target = (owner + "/" + repo)[:80]
        try:
            result = import_repo_skills(owner, repo, ref=ref, token=token)
        except ValueError as error:
            _audit("skills.library.import_github", "rejected", target)
            return payload(
                error={"code": "skills_import_rejected", "message": str(error)},
                status=400,
            )
        _screen_import(result.get("skills", []))
        _audit("skills.library.import_github", "success", target,
               found=len(result.get("skills", [])), ref=result.get("ref", ""))
        return payload(result)

    @blueprint.post("/skills/library/<skill_id>")
    @require_auth("administrator", csrf=True)
    def skills_library_update(skill_id):
        store = library()
        if store is None:
            return _unavailable()
        body = request.get_json(silent=True) or {}
        try:
            updated = store.update(str(skill_id), body)
        except SkillsLibraryError as error:
            _audit("skills.library.update", "rejected", str(skill_id))
            return payload(
                error={"code": "skills_library_rejected", "message": str(error)},
                status=400,
            )
        _audit("skills.library.update", "success", updated["id"])
        return payload(_skill_view(updated))

    @blueprint.delete("/skills/library/<skill_id>")
    @require_auth("administrator", csrf=True)
    def skills_library_remove(skill_id):
        store = library()
        if store is None:
            return _unavailable()
        try:
            result = store.remove(str(skill_id))
        except SkillsLibraryError as error:
            _audit("skills.library.remove", "rejected", str(skill_id))
            return payload(
                error={"code": "skills_library_rejected", "message": str(error)},
                status=404,
            )
        _audit("skills.library.remove", "success", str(skill_id))
        return payload(result)

    # -- per-target attachments -------------------------------------------
    def _attachment_view(item):
        return {
            "id": item["id"],
            "created_by": item["created_by"],
            "target_type": item["target_type"],
            "target_id": item["target_id"],
            "target_version": item["target_version"],
            "skill_id": item["skill_id"],
            "revoked": item["revoked"],
            "revoked_at": item.get("revoked_at"),
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
        }

    def _review_view(skill):
        """The grants and broker credential ids an attached skill assumes.

        Returned by the attach route only; no screen renders it today (see the
        module docstring). Credentials are ids only - no secret value is here.
        """
        return {
            "skill_id": skill["id"],
            "name": skill["name"],
            "description": skill["description"],
            "grants": list(skill["grants"]),
            "credentials": list(skill["credentials"]),
        }

    @blueprint.get("/skills/attachments")
    @require_auth("operator")
    def skills_attachments_list():
        store = attachments()
        if store is None:
            return payload({"attachments": [], "configured": False})
        target_id = request.args.get("target_id") or None
        target_type = request.args.get("target_type") or None
        items = store.list(target_id=target_id, target_type=target_type, limit=200)
        return payload({
            "configured": True,
            "attachments": [_attachment_view(item) for item in items],
        })

    @blueprint.post("/skills/attachments")
    @require_auth("administrator", csrf=True)
    def skills_attachments_attach():
        store = attachments()
        skills = library()
        if store is None:
            return _unavailable()
        body = request.get_json(silent=True) or {}
        try:
            created = store.create(
                g.auth_session.username,
                body.get("target_type", ""),
                body.get("target_id", ""),
                body.get("target_version", 0),
                body.get("skill_id", ""),
                idempotency_key=body.get("idempotency_key"),
            )
        except SkillsLibraryError as error:
            _audit("skills.attach", "rejected", str(body.get("skill_id", ""))[:64])
            return payload(
                error={"code": "skills_attach_rejected", "message": str(error)},
                status=400,
            )
        _audit(
            "skills.attach", "success", created["id"],
            target_id=created["target_id"], skill_id=created["skill_id"],
        )
        # The review view names the grants and credential ids the attached skill
        # assumes, read back from the stored manifest rather than the request
        # body. It is returned to the API caller only; no screen shows it.
        review = None
        if skills is not None:
            skill = skills.get(created["skill_id"])
            if skill is not None:
                review = _review_view(skill)
        return payload(
            {"attachment": _attachment_view(created), "review": review},
            status=201,
        )

    @blueprint.delete("/skills/attachments/<attachment_id>")
    @require_auth("administrator", csrf=True)
    def skills_attachments_revoke(attachment_id):
        store = attachments()
        if store is None:
            return _unavailable()
        try:
            revoked = store.revoke(
                str(attachment_id), "Skill attachment revoked by an administrator."
            )
        except SkillsLibraryError as error:
            _audit("skills.detach", "rejected", str(attachment_id))
            return payload(
                error={"code": "skills_attach_rejected", "message": str(error)},
                status=404,
            )
        _audit("skills.detach", "success", str(attachment_id))
        return payload(_attachment_view(revoked))
