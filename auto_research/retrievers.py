"""Free web source retrievers used by the Retriever agent."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib.parse import quote_plus

import requests

from auto_research.schemas import Snippet


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
DUCKDUCKGO_API_URL = "https://api.duckduckgo.com/"
DEFAULT_TIMEOUT = 20


@dataclass
class RetrievalResult:
    """Normalized retrieval output including debug metadata."""

    snippets: list[Snippet]
    metadata: dict[str, Any]


class FreeWebRetriever:
    """Retriever backed by free public endpoints."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AutoResearch/1.0"})

    def retrieve(self, sub_question: str, keywords: list[str]) -> RetrievalResult:
        """Query the free sources and return normalized snippets."""

        query = " ".join([sub_question, *keywords]).strip()
        snippets: list[Snippet] = []
        metadata: dict[str, Any] = {"query": query}

        wikipedia_snippets, wikipedia_metadata = self._safe_search(self._search_wikipedia, query)
        snippets.extend(wikipedia_snippets)
        metadata.update(wikipedia_metadata)

        duckduckgo_snippets, duckduckgo_metadata = self._safe_search(self._search_duckduckgo, query)
        snippets.extend(duckduckgo_snippets)
        metadata.update(duckduckgo_metadata)

        deduped = _dedupe_snippets(snippets)
        return RetrievalResult(
            snippets=deduped[:8],
            metadata=metadata,
        )

    def _safe_search(
        self,
        search_fn: Any,
        query: str,
    ) -> tuple[list[Snippet], dict[str, Any]]:
        """Return empty results instead of crashing on one flaky upstream source."""

        source_name = search_fn.__name__.removeprefix("_search_")
        try:
            snippets = search_fn(query)
            return snippets, {f"{source_name}_count": len(snippets)}
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            return [], {f"{source_name}_count": 0, f"{source_name}_error": str(exc)}

    def _search_wikipedia(self, query: str) -> list[Snippet]:
        """Search Wikipedia and normalize summary extracts."""

        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
        }
        response = self.session.get(WIKIPEDIA_API_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        snippets: list[Snippet] = []
        for result in payload.get("query", {}).get("search", []):
            title = result.get("title", "").strip()
            snippet_html = result.get("snippet", "")
            snippet_text = _strip_html(snippet_html)
            if not title or not snippet_text:
                continue
            snippets.append(
                {
                    "text": snippet_text,
                    "url": f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}",
                    "title": title,
                }
            )
        return snippets

    def _search_duckduckgo(self, query: str) -> list[Snippet]:
        """Search DuckDuckGo Instant Answer and related topics."""

        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 0,
        }
        response = self.session.get(DUCKDUCKGO_API_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        snippets: list[Snippet] = []
        abstract_text = payload.get("AbstractText", "").strip()
        abstract_url = payload.get("AbstractURL", "").strip()
        heading = payload.get("Heading", "").strip() or "DuckDuckGo Result"

        if abstract_text and abstract_url:
            snippets.append({"text": abstract_text, "url": abstract_url, "title": heading})

        for topic in payload.get("RelatedTopics", []):
            nested_topics = topic.get("Topics")
            if nested_topics:
                for nested in nested_topics:
                    snippet = _topic_to_snippet(nested)
                    if snippet:
                        snippets.append(snippet)
                continue
            snippet = _topic_to_snippet(topic)
            if snippet:
                snippets.append(snippet)

        return snippets[:5]


def _topic_to_snippet(topic: dict[str, Any]) -> Snippet | None:
    """Normalize a DuckDuckGo related topic entry."""

    text = topic.get("Text", "").strip()
    url = topic.get("FirstURL", "").strip()
    if not text or not url:
        return None
    title = text.split(" - ", 1)[0]
    return {"text": text, "url": url, "title": title}


def _strip_html(value: str) -> str:
    """Remove minimal HTML tags returned by the Wikipedia search endpoint."""

    text = value.replace("<span class=\"searchmatch\">", "").replace("</span>", "")
    return " ".join(text.split())


def _dedupe_snippets(snippets: list[Snippet]) -> list[Snippet]:
    """Deduplicate snippets by URL and text."""

    seen: set[tuple[str, str]] = set()
    unique: list[Snippet] = []
    for snippet in snippets:
        key = (snippet["url"], snippet["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(snippet)
    return unique
