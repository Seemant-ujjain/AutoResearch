"""Structured message schemas for agent communication."""

from __future__ import annotations

from typing import Any, Mapping, TypedDict, get_args, get_origin, get_type_hints


class PlannerInput(TypedDict):
    """Raw user question routed to the Planner agent."""

    question: str


class PlannerOutput(TypedDict):
    """Research plan produced by the Planner agent."""

    sub_questions: list[str]
    keywords: list[list[str]]
    scope: str


class RetrieverInput(TypedDict):
    """Single sub-question retrieval request."""

    sub_question: str
    keywords: list[str]


class Snippet(TypedDict):
    """Raw snippet returned from a free web source."""

    text: str
    url: str
    title: str


class RetrieverOutput(TypedDict):
    """Raw retrieval results for one sub-question."""

    snippets: list[Snippet]


class SummarizerInput(TypedDict):
    """Snippets to summarize for one sub-question."""

    sub_question: str
    snippets: list[Snippet]


class Fact(TypedDict):
    """Summarized claim with provenance and confidence."""

    claim: str
    source_url: str
    confidence: str


class SummarizerOutput(TypedDict):
    """Facts extracted from retrieved snippets."""

    facts: list[Fact]


class CriticInput(TypedDict):
    """Cross-question review input."""

    all_facts: list[Fact]
    sub_questions: list[str]


class CriticOutput(TypedDict):
    """Critic decision on coverage and follow-up retrieval."""

    gaps: list[str]
    requery_needed: bool
    requery_questions: list[str]


class WriterInput(TypedDict):
    """Input used to draft the final report."""

    sub_questions: list[str]
    all_facts: list[Fact]
    scope: str


def validate_typed_dict(data: Mapping[str, Any], schema: type[TypedDict]) -> list[str]:
    """Return schema validation errors for a TypedDict-like mapping."""

    errors: list[str] = []
    annotations = get_type_hints(schema)
    required_keys = getattr(schema, "__required_keys__", set(annotations))

    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")

    for key, value in data.items():
        if key not in annotations:
            errors.append(f"Unexpected key: {key}")
            continue
        expected_type = annotations[key]
        if not _matches_type(value, expected_type):
            errors.append(
                f"Key '{key}' expected {_type_name(expected_type)}, "
                f"got {type(value).__name__}"
            )

    return errors


def _matches_type(value: Any, expected_type: Any) -> bool:
    """Best-effort runtime validation for the schema types used here."""

    origin = get_origin(expected_type)
    if origin is None:
        if expected_type is Any:
            return True
        annotations = getattr(expected_type, "__annotations__", None)
        if annotations is not None:
            return isinstance(value, Mapping) and not validate_typed_dict(value, expected_type)
        if isinstance(expected_type, type):
            return isinstance(value, expected_type)
        return True

    if origin is list:
        if not isinstance(value, list):
            return False
        item_type = get_args(expected_type)[0]
        return all(_matches_type(item, item_type) for item in value)

    if origin is dict:
        return isinstance(value, dict)

    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        args = get_args(expected_type)
        if len(args) != len(value):
            return False
        return all(_matches_type(item, item_type) for item, item_type in zip(value, args))

    if origin is type(None):
        return value is None

    union_args = get_args(expected_type)
    if union_args:
        return any(_matches_type(value, union_type) for union_type in union_args)

    return True


def _type_name(expected_type: Any) -> str:
    """Return a readable type name for validation errors."""

    if hasattr(expected_type, "__name__"):
        return expected_type.__name__
    return str(expected_type)
