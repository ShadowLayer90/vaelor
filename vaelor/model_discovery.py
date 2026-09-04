"""Bounded Hugging Face GGUF discovery and hardware-fit classification."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict

from .copilot_setup import local_runtime_settings
from .inference_tuning import RECOMMENDED_RUNTIME_MODE


REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
FILE_QUERY = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")


def _files(entries: Any, name_key: str) -> list[Dict[str, Any]]:
    result = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        filename = str(entry.get(name_key, ""))
        if not filename.lower().endswith(".gguf"):
            continue
        result.append({"name": filename[:500], "size_bytes": int(entry.get("size") or 0)})
    return result[:24]


def _tree_files(model_id: str, opener: Callable) -> list[Dict[str, Any]]:
    url = "https://huggingface.co/api/models/{}/tree/main?recursive=true&expand=false".format(
        urllib.parse.quote(model_id, safe="/")
    )
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "Vaelor-Control-Plane/2"}
    )
    try:
        with opener(request) as response:
            entries = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    return _files(entries, "path")


def _fit(
    file: Dict[str, Any], hardware: Dict[str, Any], repo: str = ""
) -> Dict[str, Any]:
    """Whether this file fits, and on what evidence.

    **This is the verdict the owner acts on**, so where the number came from is
    part of the answer. For a catalogued artifact the sizing consults a measured
    footprint; for anything else it falls back to `model_size * 1.25 + overhead`,
    which for the one model where both exist is 1,024 MiB low (#117, VD-070).
    Saying "fits" on an estimate reads identically to saying it on a
    measurement, and only one of them is a promise the appliance can keep.
    """
    size = int(file.get("size_bytes") or 0)
    if size <= 0:
        return {**file, "fits_hardware": False, "fit_reason": "File size must be verified before download."}
    try:
        # Previewed with the mode the deploy will actually use, so "fits" is a
        # statement about the configuration that gets shipped.
        preview = local_runtime_settings(
            hardware, size, RECOMMENDED_RUNTIME_MODE,
            repo=repo, file=str(file.get("name", "")),
        )
    except ValueError as error:
        return {**file, "fits_hardware": False, "fit_reason": str(error)}
    measured = preview.get("footprint_source") != "estimated"
    return {
        **file, "fits_hardware": True,
        "fit_reason": (
            "Fits while preserving operating-system memory, from a measured "
            "footprint for this model on this platform."
            if measured else
            "Fits while preserving operating-system memory. The memory figure "
            "is estimated from the file size; this model has not been measured "
            "on this platform."
        ),
        "fit_basis": preview.get("footprint_source", "estimated"),
        "runtime_preview": preview,
    }


def inspect_models(
    payload: Dict[str, Any], opener: Callable, checkpoint: Callable,
    hardware: Dict[str, Any],
) -> Dict[str, Any]:
    query = str(payload.get("query", "")).strip()
    if not query:
        raise ValueError("A model search query is required.")
    requested_repo = str(payload.get("repo", "")).strip() or query.split(maxsplit=1)[0]
    exact_repo = requested_repo if REPOSITORY.fullmatch(requested_repo) else ""
    if payload.get("repo") and not exact_repo:
        raise ValueError("Choose a valid Hugging Face repository.")
    exact_file = str(payload.get("file", "")).strip()
    file_query = str(payload.get("file_query", "")).strip()
    if exact_file and (not FILE_QUERY.fullmatch(exact_file) or not exact_file.lower().endswith(".gguf")):
        raise ValueError("Choose a safe GGUF file name.")
    if file_query and not FILE_QUERY.fullmatch(file_query):
        raise ValueError("Choose a safe GGUF quantization or file filter.")
    checkpoint(25, "Checking the exact model repository" if exact_repo else "Searching the model catalog", "downloading")
    parameters = urllib.parse.urlencode({
        "search": (exact_repo or query)[:200], "filter": "gguf", "limit": 8, "full": "true",
    })
    request = urllib.request.Request(
        "https://huggingface.co/api/models?{}".format(parameters),
        headers={"Accept": "application/json", "User-Agent": "Vaelor-Control-Plane/2"},
    )
    with opener(request) as response:
        models = json.loads(response.read().decode("utf-8"))
    candidates = []
    for model in models[:8] if isinstance(models, list) else []:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id") or "")
        if exact_repo and model_id.casefold() != exact_repo.casefold():
            continue
        files = _tree_files(model_id, opener)
        if not files:
            files = _files(model.get("siblings", []), "rfilename")
        if exact_file:
            files = [item for item in files if item["name"].casefold() == exact_file.casefold()]
        elif file_query:
            needle = re.sub(r"[^a-z0-9]", "", file_query.casefold())
            files = [
                item for item in files
                if needle in re.sub(r"[^a-z0-9]", "", item["name"].casefold())
            ]
        candidates.append({
            "id": model_id,
            "downloads": model.get("downloads"),
            "likes": model.get("likes"),
            "gated": model.get("gated", False),
            "private": model.get("private", False),
            "tags": [tag for tag in model.get("tags", []) if tag in {"gguf", "text-generation", "conversational"}],
            "files": [_fit(item, hardware, model_id) for item in files[:24]],
        })
    checkpoint(90, "Exact model files and hardware fit checked", "validating")
    return {
        "query": query[:200],
        "request": {
            "repo": exact_repo or None, "file": exact_file or None,
            "file_query": file_query or None, "selection_required": True,
        },
        "candidates": candidates,
        "matching_files": sum(len(candidate["files"]) for candidate in candidates),
    }


def verify_inspected_selection(
    store: Any, inspection_job_id: Any, repo: str, filename: str, size_bytes: int,
) -> None:
    """Bind a download to one completed, hardware-compatible inspection."""
    if not inspection_job_id:
        return
    job = store.get(str(inspection_job_id))
    if not job or job.get("type") != "model.inspect" or job.get("state") != "completed":
        raise ValueError("Run the model compatibility check before downloading.")
    for candidate in job.get("result", {}).get("candidates", []):
        if str(candidate.get("id", "")).casefold() != repo.casefold():
            continue
        for file in candidate.get("files", []):
            if str(file.get("name", "")).casefold() != filename.casefold():
                continue
            if int(file.get("size_bytes") or 0) != size_bytes or size_bytes <= 0:
                raise ValueError("The selected model size changed; inspect it again.")
            if file.get("fits_hardware") is not True:
                raise ValueError(str(file.get("fit_reason") or "The selected model does not fit this node."))
            return
    raise ValueError("Choose a model file returned by the completed compatibility check.")
