"""Pipeline orchestration and artifact writing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from auto_research.agents import (
    CriticAgent,
    PlannerAgent,
    RetrieverAgent,
    SummarizerAgent,
    WriterAgent,
)
from auto_research.llm_client import OpenAIJSONClient
from auto_research.retrievers import FreeWebRetriever
from auto_research.schemas import (
    Fact,
    CriticInput,
    CriticOutput,
    PlannerInput,
    PlannerOutput,
    RetrieverInput,
    RetrieverOutput,
    SummarizerInput,
    SummarizerOutput,
    WriterInput,
    validate_typed_dict,
)
from auto_research.utils import dump_json, ensure_directory, slugify


DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_WRITER_MODEL = "gpt-4o"
MAX_PLANNER_SUBQUESTIONS = 3
MAX_CRITIC_ITERATIONS = 2


@dataclass
class TraceEvent:
    """Single trace record for agent-level message flow."""

    agent: str
    input_message: dict[str, Any]
    output_message: dict[str, Any] | str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunArtifacts:
    """Paths for the report and trace produced by one run."""

    report_path: Path
    trace_path: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the pipeline."""

    parser = argparse.ArgumentParser(description="Run the AutoResearch pipeline.")
    parser.add_argument("--question", required=True, help="Plain-English research question.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where reports and traces are saved.",
    )
    parser.add_argument(
        "--writer-model",
        default=DEFAULT_WRITER_MODEL,
        help="Optional writer model override.",
    )
    return parser.parse_args()


def validate_message(message: dict[str, Any], schema: type[Any], agent: str) -> None:
    """Raise a helpful error when an agent message violates its schema."""

    errors = validate_typed_dict(message, schema)
    if errors:
        details = "; ".join(errors)
        raise ValueError(f"{agent} message schema validation failed: {details}")


def initialize_trace(question: str) -> dict[str, Any]:
    """Create a run trace shell that can be appended during execution."""

    return {
        "question": question,
        "events": [],
        "models": {
            "default": DEFAULT_MODEL,
            "writer": DEFAULT_WRITER_MODEL,
        },
    }


def append_trace(
    trace: dict[str, Any],
    agent: str,
    input_message: dict[str, Any],
    output_message: dict[str, Any] | str,
    **metadata: Any,
) -> None:
    """Record one agent interaction in the run trace."""

    event = TraceEvent(
        agent=agent,
        input_message=input_message,
        output_message=output_message,
        metadata=metadata,
    )
    trace["events"].append(
        {
            "agent": event.agent,
            "input": event.input_message,
            "output": event.output_message,
            "metadata": event.metadata,
        }
    )


def build_artifact_paths(question: str, output_dir: Path) -> RunArtifacts:
    """Compute deterministic output paths for one run."""

    slug = slugify(question)
    return RunArtifacts(
        report_path=output_dir / f"report_{slug}.md",
        trace_path=output_dir / f"run_trace_{slug}.json",
    )


def run_pipeline(question: str, output_dir: Path, writer_model: str) -> RunArtifacts:
    """Run the end-to-end pipeline and write report and trace artifacts."""

    ensure_directory(output_dir)
    artifacts = build_artifact_paths(question, output_dir)
    trace = initialize_trace(question)
    trace["models"]["writer"] = writer_model

    llm_client = OpenAIJSONClient(default_model=DEFAULT_MODEL)
    planner = PlannerAgent(llm_client)
    retriever = RetrieverAgent(FreeWebRetriever())
    summarizer = SummarizerAgent(llm_client)
    critic = CriticAgent(llm_client)
    writer = WriterAgent(llm_client, model=writer_model)

    planner_input: PlannerInput = {"question": question}
    validate_message(planner_input, PlannerInput, "Planner input")
    planner_result = planner.run(planner_input)
    planner_output = _normalize_planner_output(planner_result.output)
    validate_message(planner_output, PlannerOutput, "Planner output")
    append_trace(trace, "planner", planner_input, planner_output, **planner_result.metadata)

    sub_questions = planner_output["sub_questions"]
    keyword_groups = planner_output["keywords"]
    all_facts: list[Fact] = []

    for sub_question, keywords in zip(sub_questions, keyword_groups):
        _retrieve_and_summarize(
            trace=trace,
            retriever=retriever,
            summarizer=summarizer,
            sub_question=sub_question,
            keywords=keywords,
            all_facts=all_facts,
        )

    critic_iterations = 0
    while critic_iterations < MAX_CRITIC_ITERATIONS:
        critic_input: CriticInput = {
            "all_facts": all_facts,
            "sub_questions": sub_questions,
        }
        validate_message(critic_input, CriticInput, "Critic input")
        critic_result = critic.run(critic_input)
        critic_output = critic_result.output
        validate_message(critic_output, CriticOutput, "Critic output")
        append_trace(
            trace,
            "critic",
            critic_input,
            critic_output,
            iteration=critic_iterations + 1,
            **critic_result.metadata,
        )

        critic_iterations += 1
        if not critic_output["requery_needed"]:
            break

        for requery_question in critic_output["requery_questions"]:
            _retrieve_and_summarize(
                trace=trace,
                retriever=retriever,
                summarizer=summarizer,
                sub_question=requery_question,
                keywords=[],
                all_facts=all_facts,
            )

    writer_input: WriterInput = {
        "sub_questions": sub_questions,
        "all_facts": all_facts,
        "scope": planner_output["scope"],
    }
    validate_message(writer_input, WriterInput, "Writer input")
    writer_result = writer.run(writer_input)
    report = writer_result.output
    append_trace(trace, "writer", writer_input, report, **writer_result.metadata)

    artifacts.report_path.write_text(f"{report}\n", encoding="utf-8")
    dump_json(artifacts.trace_path, trace)
    return artifacts


def _retrieve_and_summarize(
    trace: dict[str, Any],
    retriever: RetrieverAgent,
    summarizer: SummarizerAgent,
    sub_question: str,
    keywords: list[str],
    all_facts: list[Fact],
) -> None:
    """Run retrieval and summarization for one sub-question."""

    retriever_input: RetrieverInput = {
        "sub_question": sub_question,
        "keywords": keywords,
    }
    validate_message(retriever_input, RetrieverInput, "Retriever input")
    retriever_result = retriever.run(retriever_input)
    retriever_output = retriever_result.output
    validate_message(retriever_output, RetrieverOutput, "Retriever output")
    append_trace(trace, "retriever", retriever_input, retriever_output, **retriever_result.metadata)

    summarizer_input: SummarizerInput = {
        "sub_question": sub_question,
        "snippets": retriever_output["snippets"],
    }
    validate_message(summarizer_input, SummarizerInput, "Summarizer input")
    summarizer_result = summarizer.run(summarizer_input)
    summarizer_output = summarizer_result.output
    validate_message(summarizer_output, SummarizerOutput, "Summarizer output")
    append_trace(
        trace,
        "summarizer",
        summarizer_input,
        summarizer_output,
        **summarizer_result.metadata,
    )
    all_facts.extend(summarizer_output["facts"])


def _normalize_planner_output(output: dict[str, Any] | str) -> PlannerOutput:
    """Coerce planner output into the exact schema needed downstream."""

    if not isinstance(output, dict):
        raise ValueError("Planner output must be a JSON object.")
    sub_questions = list(output.get("sub_questions", []))[:MAX_PLANNER_SUBQUESTIONS]
    keywords = [list(group) for group in output.get("keywords", [])][: len(sub_questions)]
    normalized: PlannerOutput = {
        "sub_questions": sub_questions,
        "keywords": keywords,
        "scope": str(output.get("scope", "")),
    }
    return normalized


def main() -> int:
    """CLI entrypoint for the AutoResearch pipeline."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    artifacts = run_pipeline(
        question=args.question,
        output_dir=output_dir,
        writer_model=args.writer_model,
    )
    print(f"Report: {artifacts.report_path}")
    print(f"Trace: {artifacts.trace_path}")
    return 0
