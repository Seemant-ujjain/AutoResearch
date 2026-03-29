"""OpenAI client helpers and JSON parsing utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = os.getenv("AUTORESEARCH_MODEL", "gpt-4.1-mini")


class JSONParseError(ValueError):
    """Raised when a model response cannot be parsed as valid JSON."""


@dataclass
class LLMExchange:
    """Record of a single model call."""

    model: str
    system_prompt: str
    user_prompt: str
    raw_response: str


def safe_parse(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from text, tolerating fenced-code responses."""

    candidate = raw_text.strip()
    if candidate.startswith("```"):
        lines = [line for line in candidate.splitlines() if not line.startswith("```")]
        candidate = "\n".join(lines).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JSONParseError(str(exc)) from exc

    if not isinstance(parsed, dict):
        raise JSONParseError("Expected top-level JSON object.")
    return parsed


class OpenAIJSONClient:
    """Thin wrapper around the OpenAI SDK for JSON and Markdown outputs."""

    def __init__(self, api_key: str | None = None, default_model: str = DEFAULT_MODEL) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=resolved_key)
        self.default_model = default_model

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> tuple[dict[str, Any], LLMExchange]:
        """Generate a JSON object, retrying once on malformed output."""

        selected_model = model or self.default_model
        raw_response = self._complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=selected_model,
            json_mode=True,
        )

        try:
            parsed = safe_parse(raw_response)
            return parsed, LLMExchange(selected_model, system_prompt, user_prompt, raw_response)
        except JSONParseError:
            retry_prompt = (
                f"{user_prompt}\n\n"
                "Your previous reply was not valid JSON. "
                "Return one valid JSON object only, with no extra prose."
            )
            retry_response = self._complete(
                system_prompt=system_prompt,
                user_prompt=retry_prompt,
                model=selected_model,
                json_mode=True,
            )
            parsed = safe_parse(retry_response)
            return parsed, LLMExchange(selected_model, system_prompt, retry_prompt, retry_response)

    def generate_markdown(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> tuple[str, LLMExchange]:
        """Generate Markdown output for the Writer agent."""

        raw_response = self._complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            json_mode=False,
        )
        return raw_response.strip(), LLMExchange(model, system_prompt, user_prompt, raw_response)

    def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        json_mode: bool,
    ) -> str:
        """Issue a chat completion request and return the text body."""

        response_format: dict[str, str] | None = {"type": "json_object"} if json_mode else None
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Model returned an empty response.")
        return content
