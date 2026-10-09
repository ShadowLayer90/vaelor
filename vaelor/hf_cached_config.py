"""A model's ``config.json``, read from this controller's model library - never pasted.

The model profile (`vllm_model_profile`) and the hybrid KV fit are only as good
as the ``config.json`` they read, and asking the owner to paste one into a
deploy body is asking for a file Vaelor already has: the model library pulls
every repo into the node-local Hugging Face cache before it is served, and the
cache holds the repo's own ``config.json``. So the fit preview and the deploy
read it from there (:func:`with_cached_config`), and a pasted ``hf_config``
is only the override for a repo not yet fetched.

**Read, not trusted.** The file sits in the store under the layout the Hugging
Face hub writes - ``hub/models--org--name/refs/<branch>`` names a commit and
``snapshots/<commit>/config.json`` links to a blob - and it is read as the
control plane, without privilege, with every path it resolves to kept inside
that repo's own folder, a size cap, and the profile's strict parse after it.
The root bridge separately refuses to start a container over a repo tree any
account but root wrote (VD-143), so a config planted by another account is
never one a serving container then runs over.

**Absence is honest, not an error - and never silent.** A repo the library has
not fetched here returns ``None``; the deploy then goes on exactly as it did
before this module (the name-based profile, every layer's KV counted), and
only a choice that needs the config - multi-token prediction - is refused, in
words that say to fetch the model. A file that IS there but cannot be read (a
permission the control plane lacks, a broken link) or that the profile
refuses is the same ``None`` to the caller, but it is not the same fact: it is
logged, once per reason and path, in a plain sentence saying what was not
read and what the deploy does instead (:func:`_say_once`), so "Vaelor found
no config" can never hide "Vaelor was not allowed to look".
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Mapping, Optional, Tuple

from .gpu_model_catalog import GPU_MODEL_CATALOG
from .hf_model_source import resolve_model_source
from .vllm_container import MODEL_CACHE_ROOT
from .vllm_model_profile import ModelFacts, facts_from_hf_config

#: The largest config.json read (the profile's own cap, in bytes on disk).
_MAX_BYTES = 256 * 1024
_BRANCH_DEFAULT = "main"
_HEX = frozenset("0123456789abcdef")

LOGGER = logging.getLogger(__name__)
#: What the model is sized and served without, said after every reason.
_CONSEQUENCE = (
    "so this model is sized and served without its config.json (every layer's "
    "KV cache counted, parsers chosen from its name, multi-token prediction "
    "unavailable)."
)
_SAID: set = set()
_SAID_LOCK = threading.Lock()


def _say_once(reason: str, path: str) -> None:
    """Log one plain warning per reason and path, however often it recurs."""
    key = (reason, path)
    with _SAID_LOCK:
        if key in _SAID:
            return
        _SAID.add(key)
    LOGGER.warning("%s (%s), %s", reason, path, _CONSEQUENCE)


def _read_failure(error: OSError, path: str) -> None:
    """Say why a file the cache holds could not be read; absence stays quiet."""
    if isinstance(error, FileNotFoundError):
        return
    if isinstance(error, PermissionError):
        _say_once(
            "The control plane is not allowed to read the model library's "
            "copy of a model's config.json", path,
        )
        return
    _say_once(
        "The model library's copy of a model's config.json could not be read: "
        "{}".format(error.strerror or type(error).__name__), path,
    )


def _repo_folder(store: str, repo: str) -> str:
    return os.path.join(store, "hub", "models--" + repo.replace("/", "--"))


def _commit(folder: str, revision: Optional[str]) -> Optional[str]:
    """The snapshot commit ``revision`` (a branch or a commit) names, or ``None``."""
    wanted = str(revision or _BRANCH_DEFAULT)
    if len(wanted) == 40 and set(wanted) <= _HEX:
        return wanted
    if "/" in wanted or wanted.startswith("."):
        _say_once("A model revision that is not a branch name or a commit was not "
                  "looked up in the model library", folder)
        return None
    ref = os.path.join(folder, "refs", wanted)
    try:
        with open(ref, encoding="utf-8") as handle:
            commit = handle.read(64).strip()
    except OSError as error:
        _read_failure(error, ref)
        return None
    return commit if len(commit) == 40 and set(commit) <= _HEX else None


def cached_config(
    repo: str, revision: Optional[str] = None, *, store: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """``repo``'s ``config.json`` from the node cache under ``store`` (this
    controller's model store unless a test names another), or ``None``."""
    if not isinstance(repo, str) or repo.count("/") != 1 or ".." in repo:
        return None
    folder = _repo_folder(store or MODEL_CACHE_ROOT, repo)
    commit = _commit(folder, revision)
    if commit is None:
        return None
    path = os.path.join(folder, "snapshots", commit, "config.json")
    try:
        real = os.path.realpath(path)
        if os.path.commonpath([real, os.path.realpath(folder)]) != os.path.realpath(folder):
            _say_once("A model's cached config.json points outside its own folder", path)
            return None
        with open(real, "rb") as handle:
            raw = handle.read(_MAX_BYTES + 1)
    except ValueError:
        return None
    except OSError as error:
        _read_failure(error, path)
        return None
    if len(raw) > _MAX_BYTES:
        _say_once("A model's cached config.json is larger than any config", path)
        return None
    try:
        config = json.loads(raw.decode("utf-8"))
        facts_from_hf_config(config)
    except (ValueError, RecursionError) as error:
        _say_once("A model's cached config.json was refused: {}".format(error), path)
        return None
    if not isinstance(config, dict):
        _say_once("A model's cached config.json is not a config object", path)
        return None
    return config


def cached_facts(repo: Any, revision: Optional[str] = None) -> Optional[ModelFacts]:
    """What ``repo``'s ``config.json`` in this controller's library says, or ``None``.

    For the two moments a deployment's options can be decided after a first
    look found no file: once the deploy's own fetch has put it there, and at
    a later Load (`vllm_serve_options.options_for_load`).
    """
    config = cached_config(repo, revision)
    return facts_from_hf_config(config) if config is not None else None


def with_cached_config(
    spec_body: Mapping[str, Any], repo: Optional[str], revision: Optional[str] = None,
    *, store: Optional[str] = None,
) -> Dict[str, Any]:
    """``spec_body`` with ``hf_config`` from the cache added, unless one was given."""
    body = dict(spec_body or {})
    if body.get("hf_config") is None and repo:
        config = cached_config(repo, revision, store=store)
        if config is not None:
            body["hf_config"] = config
    return body


def source_repo(model_source: Any) -> Tuple[Optional[str], Optional[str]]:
    """``(repo, revision)`` a deploy's or fit's ``model_source`` names, or ``(None, None)``."""
    if not model_source:
        return None, None
    try:
        source = resolve_model_source(model_source, GPU_MODEL_CATALOG.keys())
    except ValueError:
        return None, None
    if source["kind"] == "catalog":
        return GPU_MODEL_CATALOG[source["id"]]["repo"], None
    return source["repo"], source["revision"]
