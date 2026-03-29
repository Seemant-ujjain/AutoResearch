# AutoResearch

AutoResearch is a lightweight multi-agent research pipeline built for the CSE598 small assignment. It takes a natural-language question, decomposes it into sub-questions, retrieves public web snippets, distills those snippets into grounded facts, critiques coverage, and writes a final Markdown report plus a JSON execution trace.

## Setup

### Requirements

- Python 3.11+ recommended
- An OpenAI API key
- Internet access for both OpenAI API calls and public web retrieval

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure environment

Create a `.env` file or export the variables in your shell:

```bash
OPENAI_API_KEY=your_key_here
AUTORESEARCH_MODEL=gpt-4.1-mini
```

Important: the current CLI reads `OPENAI_API_KEY` from the process environment. It does not automatically load `.env`, so before running the app you should export the variables into your shell, for example:

```bash
set -a
source .env
set +a
```

Note: `AUTORESEARCH_MODEL` is used as the default model for planner, summarizer, and critic. The writer model is currently controlled by the `--writer-model` CLI flag rather than by an environment variable.

### Run

```bash
python3 run.py --question "What is retrieval-augmented generation?"
```

Optional writer override:

```bash
python3 run.py \
  --question "What is retrieval-augmented generation?" \
  --writer-model gpt-4.1-mini
```

### Outputs

Each run writes two files to `outputs/`:

- `report_<question-slug>.md`: final report
- `run_trace_<question-slug>.json`: agent-by-agent trace for that run

If you run the exact same question again, the current implementation overwrites the same pair of files.

## Architecture

### Pipeline description

```text
User Question
    |
    v
PlannerAgent (OpenAI JSON)
    |
    v
Up to 3 sub-questions + keywords
    |
    v
For each sub-question:
  RetrieverAgent (Wikipedia + DuckDuckGo)
      |
      v
  SummarizerAgent (OpenAI JSON -> facts)
    |
    v
Fact pool
    |
    v
CriticAgent (OpenAI JSON, max 2 iterations)
    |
    +--> if gaps remain: re-query missing questions
    |
    v
WriterAgent (OpenAI Markdown)
    |
    v
Markdown report + JSON trace
```

### Components

- `PlannerAgent`: turns one broad question into three focused sub-questions, aligned keyword groups, and a scope statement.
- `RetrieverAgent`: queries free public sources through the `FreeWebRetriever`.
- `SummarizerAgent`: converts raw snippets into structured facts with `claim`, `source_url`, and `confidence`.
- `CriticAgent`: checks whether the collected facts cover the planned scope and can trigger re-query loops.
- `WriterAgent`: produces the final Markdown report strictly from the accumulated facts.
- `orchestrator.py`: coordinates control flow, schema validation, artifact naming, and trace recording.

## Agent Design Decisions

### 1. Keep roles narrow

Each agent has one clearly defined job. This keeps prompts shorter, makes traces easier to inspect, and reduces the chance that one model call tries to do planning, retrieval reasoning, validation, and long-form writing all at once.

### 2. Use structured JSON between most agents

Planner, summarizer, and critic outputs are constrained to JSON-like typed structures. This makes downstream validation simpler and helps the orchestrator fail earlier when an agent returns something malformed.

### 3. Separate retrieval from synthesis

The retriever only gathers snippets; it does not interpret them. Interpretation is deferred to the summarizer and writer. This separation makes it clearer which facts came from external sources versus model synthesis.

### 4. Bound the pipeline length

The planner is capped at 3 sub-questions and the critic loop is capped at 2 iterations. This keeps runs from growing indefinitely while still allowing one or two passes to recover missing coverage.

### 5. Prefer free public retrieval sources

Wikipedia and DuckDuckGo were chosen because they are easy to access and avoid paid search APIs. This keeps the non-LLM operating cost near zero, which is useful for a course project.

### 6. Preserve execution traces

Every agent interaction is written into a run trace so the pipeline can be inspected after the fact. This is useful for debugging, grading, and understanding how the final report was assembled.

## Known Limitations

- Source quality is limited. The retriever currently depends on Wikipedia and DuckDuckGo Instant Answer data, which is broad but shallow for specialized topics.
- Retrieval is brittle for niche or recent subjects. Some queries return very few snippets or generic pages.
- `.env` is not auto-loaded by the CLI. The shell must export `OPENAI_API_KEY` before running `python3 run.py`.
- Cost tracking is approximate. The code records prompts and responses in the trace, but it does not yet persist official token usage values returned by the API.
- Repeated runs overwrite previous artifacts for the same question because filenames are deterministic and based only on the question slug.
- There is no citation verification beyond source URL propagation. The writer relies on upstream facts being grounded correctly.
- The pipeline is sequential. Retrieval and summarization are performed one sub-question at a time, so latency can grow on broader prompts.
- The JSON retry logic only stores the final successful exchange in the trace, not the full history of malformed intermediate attempts.

## Estimated Cost Per Run

This project uses free web retrieval plus OpenAI API calls, so the practical paid cost comes almost entirely from the LLM calls which is negligible < $0.01 per run

### Current model settings

- Default planner / summarizer / critic model: `gpt-4.1-mini`
- Default writer model in code: `gpt-4o`
- Some of the recent test runs used `gpt-4.1-mini` for the writer as well, which is cheaper than the default configuration

### Rough estimate

Using one successful trace (`What is RAG in machine learning?`) as a reference, a full run with `gpt-4.1-mini` for all LLM calls produced about:

- 20 LLM calls
- ~6.8K approximate input tokens
- ~3.4K approximate output tokens

Using OpenAI's published pricing for `gpt-4.1-mini`, that works out to well under one cent for that specific run, roughly around `$0.01` after rounding up for approximation and minor untracked overhead. If the default `gpt-4o` writer is used instead, the total cost will typically be somewhat higher because the final writing step tends to have the longest output.

For this project, a reasonable rule-of-thumb estimate is:

- `gpt-4.1-mini` for all agents: about `$0.01` to `$0.03` per run
- default config with `gpt-4o` writer: about `$0.02` to `$0.06` per run

These numbers are only estimates, not billing-grade measurements. They depend on:

- number of planner sub-questions
- whether the critic triggers re-query loops
- size of retrieved snippet payloads
- length of the final report
- any future pricing changes from OpenAI

Reference pricing:

- OpenAI pricing page: https://platform.openai.com/pricing
- GPT-4.1 model page: https://platform.openai.com/docs/models/gpt-4.1

## Repository Structure

```text
auto_research/
  agents.py        Agent implementations
  llm_client.py    OpenAI wrapper and JSON parsing
  orchestrator.py  End-to-end control flow
  retrievers.py    Wikipedia and DuckDuckGo retrieval
  schemas.py       Typed message schemas and validation
  utils.py         File and helper utilities
run.py             CLI entrypoint
outputs/           Generated reports and traces
written_analysis.md
```

## Summary

The project emphasizes transparency and simplicity over maximum research quality. It is intentionally small, inspectable, and easy to demo: one question in, one report and one trace out. The tradeoff is that retrieval quality, citation depth, and cost accounting are still fairly lightweight compared with production-grade research systems.
