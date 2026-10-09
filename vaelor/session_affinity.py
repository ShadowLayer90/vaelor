"""The session key that keeps a conversation on one replica of a cluster model.

A replicated deployment runs one full copy of the model per machine behind the
replica balancer (VD-129). Each copy keeps its own prompt cache, so a
conversation whose turns alternate between copies recomputes its whole prompt
every turn: measured on the pair, turns two to five took 3.7-6.4 s to the first
word when they alternated and 1.5-1.7 s when they stayed on one machine.

**The rule, in one place.** A request may carry ``X-Session-Id``. The balancer
hashes a VALID key to a replica (`gpu_pool_balancer.render_balancer_config`),
so every request with that key reaches the same copy while it is up. A request
with no key, or with one that is not valid, is given a value of its own and
spread across the copies. This module owns the header's name, what a valid key
is - as the one pattern both Python and the balancer's nginx ``map`` read - and
how Vaelor's own callers make one.

**What a key is, and is not.** An opaque routing hint: it chooses a machine and
nothing else. It is never a credential, never looked up, and never written to a
log. Vaelor's own keys are a digest of the conversation's identity, so the
header carries no username and no conversation id onto the wire.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from typing import Any, Dict, Iterable, Mapping, Optional

#: The request header a caller names its conversation in.
SESSION_HEADER = "X-Session-Id"

#: The longest key the balancer will hash. Longer is treated as no key.
SESSION_KEY_MAX_LENGTH = 128

#: What a valid key looks like, anchored, in the syntax Python's ``re`` and
#: nginx's PCRE both read - the balancer's ``map`` is rendered from this very
#: string, so the two cannot admit different keys. Letters, digits and the
#: four separators an id commonly carries; no space, no quote, no backslash,
#: no ``$`` and no ``;``, so a key can never be more than a value to nginx.
#: The hyphen is last in the class, where it needs no escape in either syntax.
SESSION_KEY_PATTERN = "^[A-Za-z0-9._:-]{{1,{}}}$".format(SESSION_KEY_MAX_LENGTH)

_SESSION_KEY = re.compile(SESSION_KEY_PATTERN[1:-1])

#: How nginx names the header's value (``$http_<lower_snake>``).
NGINX_SESSION_VARIABLE = "$http_" + SESSION_HEADER.lower().replace("-", "_")

#: Says what a digest is of, so a Vaelor session key can never equal a digest
#: made for another purpose from the same identity.
_KEY_DOMAIN = "vaelor-session-affinity"


def valid_session_key(value: Any) -> str:
    """``value`` as a session key if it is one, else ``""`` (treated as absent).

    Never raises and never repairs: a key that is too long, empty, or carries
    a character outside the pattern is not trimmed into a different key - two
    conversations must not be folded onto one replica by a truncation.
    """
    if not isinstance(value, str):
        return ""
    return value if _SESSION_KEY.fullmatch(value) else ""


def conversation_session_key(*identity: Any) -> str:
    """A stable, opaque key for the conversation ``identity`` names, or ``""``.

    ``identity`` is whatever tells one conversation from another to the caller
    (AI Chat: the signed-in account and the conversation's id). The key is a
    digest of it, so the same conversation always gets the same key and the
    key reveals neither part. Any empty part means there is no conversation
    to name - a temporary chat, say - and no key is made, so that request is
    spread like any other keyless one.
    """
    parts = [str(part or "") for part in identity]
    if not parts or not all(parts):
        return ""
    digest = hashlib.sha256()
    digest.update(_KEY_DOMAIN.encode("utf-8"))
    for part in parts:
        encoded = part.encode("utf-8")
        # Length-prefixed, so ("ab", "c") and ("a", "bc") are different keys.
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()[:32]


def opening_session_key(owner: Any, messages: Optional[Iterable[Any]]) -> str:
    """A stable key for a conversation that has no id, from how it opened.

    A chat-completions client re-sends the whole conversation on every turn,
    so its first user message is the one thing every turn of it has in
    common. ``owner`` is whose conversation it is (an agent's id, an
    account); the key is a digest of the two, so the turns of one chat share
    a key, two people who open with the same words do not, and the words
    cannot be read off it. ``""`` when there is no owner or no user message
    to name it by. Two chats one owner opens with the same words share a
    key, which costs nothing: they share the prompt that would be cached.
    """
    for message in messages or ():
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            # Content in parts (text, an image): one canonical spelling.
            content = json.dumps(content, sort_keys=True, default=str) if content else ""
        return conversation_session_key(owner, "opening", content)
    return ""


def only_header_value(values: Optional[Iterable[Any]]) -> str:
    """The header's value when it was sent exactly once, else ``""``.

    nginx joins a header sent twice with a comma, which the balancer's map
    does not match, so it treats two as none. A Python reader that took the
    first would then forward a key the balancer would have ignored; this is
    the same rule, stated once.
    """
    found = [value for value in (values or ()) if isinstance(value, str)]
    return found[0] if len(found) == 1 else ""


def new_session_key() -> str:
    """A fresh random key, for one piece of work nothing else can name.

    A deployed agent's tool loop sends the same growing prompt several times
    inside one request; when the request has neither a client key nor a user
    message to derive one from (:func:`opening_session_key`), one random key
    for the loop still keeps those sends on the replica that holds the prompt.
    """
    return secrets.token_hex(16)


def session_headers(key: Any) -> Dict[str, str]:
    """``{X-Session-Id: key}`` for a valid key, else nothing to add."""
    clean = valid_session_key(key)
    return {SESSION_HEADER: clean} if clean else {}
