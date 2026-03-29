"""Agent implementations for the AutoResearch pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from auto_research.llm_client import LLMExchange, OpenAIJSONClient
from auto_research.retrievers import FreeWebRetriever, RetrievalResult
from auto_research.schemas import (
    CriticInput,
    CriticOutput,
    PlannerInput,
    PlannerOutput,
    RetrieverInput,
    RetrieverOutput,
    SummarizerInput,
    SummarizerOutput,
    WriterInput,
)


@dataclass
class AgentResult:
    """Agent output plus metadata used in run tracing."""

    output: dict[str, Any] | str
    metadata: dict[str, Any]


class PlannerAgent:
    """Decompose a user question into a structured research plan."""

    def __init__(self, llm_client: OpenAIJSONClient) -> None:
        self.llm_client = llm_client

    def run(self, message: PlannerInput) -> AgentResult:
        system_prompt = (
            "You are the Planner agent in a research pipeline. "
            "Return only valid JSON with keys sub_questions, keywords, and scope. "
            "Produce exactly 3 focused sub_questions. keywords must be a list aligned "
            "positionally with sub_questions, where each item is a list of short search terms."
        )
        user_prompt = json.dumps(message, indent=2)
        output, exchange = self.llm_client.generate_json(system_prompt, user_prompt)
        return AgentResult(output=output, metadata=_exchange_metadata(exchange))


class RetrieverAgent:
    """Fetch raw snippets for a sub-question using free web sources."""

    def __init__(self, retriever: FreeWebRetriever) -> None:
        self.retriever = retriever

    def run(self, message: RetrieverInput) -> AgentResult:
        result: RetrievalResult = self.retriever.retrieve(
            sub_question=message["sub_question"],
            keywords=message["keywords"],
        )
        output: RetrieverOutput = {"snippets": result.snippets}
        return AgentResult(output=output, metadata=result.metadata)


class SummarizerAgent:
    """Extract grounded facts from raw snippets."""

    def __init__(self, llm_client: OpenAIJSONClient) -> None:
        self.llm_client = llm_client

    def run(self, message: SummarizerInput) -> AgentResult:
        system_prompt = (
            "You are the Summarizer agent in a research pipeline. "
            "Return only valid JSON with key facts. facts must be a list of 3 to 5 objects, "
            "each with claim, source_url, and confidence. Use only the provided snippets. "
            "Set confidence to high, medium, or low."
        )
        user_prompt = json.dumps(message, indent=2)
        output, exchange = self.llm_client.generate_json(system_prompt, user_prompt)
        return AgentResult(output=output, metadata=_exchange_metadata(exchange))


class CriticAgent:
    """Review collected facts and decide whether re-querying is needed."""

    def __init__(self, llm_client: OpenAIJSONClient) -> None:
        self.llm_client = llm_client

    def run(self, message: CriticInput) -> AgentResult:
        system_prompt = (
            "You are the Critic agent in a research pipeline. "
            "Return only valid JSON with keys gaps, requery_needed, and requery_questions. "
            "Identify unanswered sub-questions, contradictions, or weak evidence. "
            "Set requery_needed to true only if another retrieval pass is justified."
        )
        user_prompt = json.dumps(message, indent=2)
        output, exchange = self.llm_client.generate_json(system_prompt, user_prompt)
        return AgentResult(output=output, metadata=_exchange_metadata(exchange))


class WriterAgent:
    """Draft the final Markdown report from verified facts only."""

    def __init__(self, llm_client: OpenAIJSONClient, model: str) -> None:
        self.llm_client = llm_client
        self.model = model

    def run(self, message: WriterInput) -> AgentResult:
        system_prompt = (
            "You are the Writer agent in a research pipeline. "
            "Write a Markdown research report with an executive summary, one section per "
            "sub-question, citations using source URLs from the facts, and a limitations section. "
            "Do not add information not present in the provided facts."
        )
        user_prompt = json.dumps(message, indent=2)
        output, exchange = self.llm_client.generate_markdown(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
        )
        return AgentResult(output=output, metadata=_exchange_metadata(exchange))


def _exchange_metadata(exchange: LLMExchange) -> dict[str, Any]:
    """Serialize the relevant LLM call details for tracing."""

    return {
        "model": exchange.model,
        "system_prompt": exchange.system_prompt,
        "user_prompt": exchange.user_prompt,
        "raw_response": exchange.raw_response,
    }
