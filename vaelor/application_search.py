"""Bounded client for an appliance-owned SearXNG search service."""

from __future__ import annotations

import ipaddress
import json
import socket
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable

from .runtime_paths import env_value


MAX_SEARCH_RESPONSE = 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _private_search_base(value: str) -> str:
    base = str(value or "").strip().rstrip("/")
    if not base:
        return ""
    parsed = urllib.parse.urlsplit(base)
    if (
        parsed.scheme != "http" or not parsed.hostname or parsed.username
        or parsed.password or parsed.query or parsed.fragment
    ):
        raise ValueError("The application search service must use a private HTTP address.")
    port = parsed.port or 80
    try:
        addresses = {
            ipaddress.ip_address(item[4][0])
            for item in socket.getaddrinfo(parsed.hostname, port)
        }
    except (OSError, ValueError) as error:
        raise ValueError("The application search service address could not be resolved.") from error
    if not addresses or any(
        not (address.is_private or address.is_loopback)
        or address.is_link_local or address.is_multicast or address.is_unspecified
        for address in addresses
    ):
        raise ValueError("The application search service must stay on loopback or the private LAN.")
    return base


def _public_result_url(value: Any) -> str:
    text = str(value or "").strip()
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError:
        return ""
    if (
        parsed.scheme != "https" or not parsed.hostname or parsed.username
        or parsed.password or parsed.fragment or len(text) > 2000
    ):
        return ""
    return text


class SearxSearchClient:
    """Return small, non-authoritative candidate lists from local SearXNG."""

    def __init__(self, base_url: str | None = None, timeout_seconds: int = 20):
        configured = base_url if base_url is not None else env_value(
            "VAELOR_APPLICATION_SEARCH_URL", "PM_APPLICATION_SEARCH_URL", "",
        )
        self.base_url = _private_search_base(configured)
        self.timeout_seconds = max(3, min(int(timeout_seconds), 30))

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def search(self, queries: Iterable[str]) -> list[Dict[str, Any]]:
        if not self.configured:
            return []
        query_results: list[list[Dict[str, Any]]] = []
        seen = set()
        for raw_query in list(queries)[:3]:
            query = " ".join(str(raw_query).split())[:300]
            if not query:
                continue
            url = self.base_url + "/search?" + urllib.parse.urlencode({
                "q": query,
                "format": "json",
                "categories": "general",
                "language": "en-US",
                "safesearch": "1",
            })
            request = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "Vaelor-Search/1"},
            )
            with urllib.request.build_opener(_NoRedirect).open(
                request, timeout=self.timeout_seconds,
            ) as response:
                raw = response.read(MAX_SEARCH_RESPONSE + 1)
            if len(raw) > MAX_SEARCH_RESPONSE:
                raise ValueError("The application search response was too large.")
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("The application search service returned invalid JSON.") from error
            if not isinstance(body, dict) or not isinstance(body.get("results"), list):
                raise ValueError("The application search service returned an invalid result list.")
            current_query: list[Dict[str, Any]] = []
            for item in body["results"][:10]:
                if not isinstance(item, dict):
                    continue
                result_url = _public_result_url(item.get("url"))
                if not result_url or result_url in seen:
                    continue
                seen.add(result_url)
                current_query.append({
                    "url": result_url,
                    "title": " ".join(str(item.get("title", "")).split())[:240],
                    "snippet": " ".join(str(item.get("content", "")).split())[:280],
                    "query": query,
                    "trust": "untrusted_search_candidate",
                })
            query_results.append(current_query)

        # Interleave query groups so the bounded candidate window represents
        # every research angle. Previously the first query could consume the
        # entire small-model prompt and hide useful registry/deployment results.
        results: list[Dict[str, Any]] = []
        depth = 0
        while len(results) < 20 and any(depth < len(group) for group in query_results):
            for group in query_results:
                if depth >= len(group):
                    continue
                results.append({**group[depth], "index": len(results)})
                if len(results) >= 20:
                    break
            depth += 1
        return results
