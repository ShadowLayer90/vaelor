"""One specialist review through the selected model.

Sibling of `custom_agent_model`: that module runs a user-defined agent, this one
runs a built-in specialist profile. Both are model-only passes, and keeping them
out of `deployment_agent` leaves that module about deciding what to run rather
than about assembling completion requests.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Mapping, Optional

from .agent_prompts import LOCAL_SPECIALIST_PROMPT, SPECIALIST_PROMPT
from .inference_client import chat_completion, inference_timeout, parse_model_object
from .model_calibration import AGENT_RESULT_SCHEMA
from .model_profiles import (
    reasoning_headroom_tokens,
    with_structured_response_format,
)
from .provider_runtime import (
    agent_context_chars,
    generation_parameters,
    managed_local_connection,
    provider_context,
    provider_system_prompt,
    provider_user_content,
)

# What a specialist review looks like once written out. A managed-local model is
# held to the short form because it runs on the appliance itself; how much room
# each model needs on top of this to think first is measured per model.
LOCAL_REVIEW_VISIBLE_TOKENS = 192
REVIEW_VISIBLE_TOKENS = 900


def specialist_review(
    connection: Mapping[str, str], profile: str, task: str,
    context: Optional[Dict[str, Any]], timeout_seconds: int,
) -> Dict[str, Any]:
    """Ask the model for one structured review and return the object it sent."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if connection.get("api_key"):
        headers["Authorization"] = "Bearer {}".format(connection["api_key"])
    model = connection.get("model", "")
    if not model:
        request = urllib.request.Request(
            "{}/models".format(connection["base_url"]), headers=headers
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read(1024 * 1024).decode("utf-8"))
        model = str(body["data"][0]["id"])
    local = managed_local_connection(dict(connection))
    # One window, two budgets: see `custom_agent_model.execute_custom_agent`.
    generation_tokens = reasoning_headroom_tokens(
        connection, LOCAL_REVIEW_VISIBLE_TOKENS if local else REVIEW_VISIBLE_TOKENS
    )
    body = chat_completion(
        connection,
        with_structured_response_format({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": provider_system_prompt(
                        (
                            LOCAL_SPECIALIST_PROMPT if local else SPECIALIST_PROMPT
                        ).format(profile=profile),
                        connection,
                    ),
                },
                {"role": "user", "content": provider_user_content({
                    "task": task,
                    "appliance_facts": provider_context(
                        context or {},
                        max_chars=agent_context_chars(
                            dict(connection), generation_tokens=generation_tokens
                        ),
                    ),
                }, connection)},
            ],
            **generation_parameters(
                connection,
                max_tokens=generation_tokens,
                temperature=0.1,
            ),
        }, connection, AGENT_RESULT_SCHEMA),
        headers,
        inference_timeout(dict(connection), timeout_seconds),
    )
    return parse_model_object(body["choices"][0]["message"]["content"])
