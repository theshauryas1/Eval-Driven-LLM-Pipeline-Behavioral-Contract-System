# Eval-Driven LLM Pipeline Behavioral Contract System

Define tests for LLM outputs, run them at scale, detect failures, and auto-improve prompts.

This repo now has two complementary product surfaces:

- `llmtest` CLI for contract-style suites, model comparison, failure analysis, reporting, and prompt repair
- FastAPI + React dashboard for trace ingestion, contract monitoring, and inspection

## Product Promise

The core workflow is:

1. Define behavioral tests for an LLM output.
2. Run those tests across one or more models or configs.
3. Diagnose failures with typed reasons instead of plain pass/fail.
4. Auto-rewrite prompts, retry, and compare whether behavior improved.

That positions the project as a testing and validation framework for LLM pipelines with contract-based constraints, failure diagnostics, and automated prompt optimization.

## Core Architecture

### Layer 1 - Test Definition

Suites live in JSON and describe:

- `input`
- `context`
- `expected`
- `constraints`
- `prompt` with version metadata

Example:

```json
{
  "suite_name": "qa_tests",
  "prompt": {
    "id": "rag_answerer",
    "version": "v1",
    "template": "Answer the question using only the provided context.\nQuestion: {input}\nContext: {context}\nAnswer:"
  },
  "tests": [
    {
      "test_name": "qa_correctness",
      "input": "What is the capital of France?",
      "context": "Paris is the capital of France.",
      "expected": { "type": "contains", "value": "Paris" },
      "constraints": [
        { "type": "max_length", "value": 50 },
        { "type": "no_hallucination" }
      ]
    }
  ]
}
```

### Layer 2 - Execution Engine

`llmtest run` and `llmtest compare` execute the same suite across multiple model adapters.

Included adapters:

- `mock`
- `echo`
- `openai_compatible`

### Layer 3 - Evaluation Engine

Checks are split into reusable evaluators:

- correctness via `expected`
- format via regex and citation checks
- length via min/max limits
- safety via PII guards
- hallucination via semantic faithfulness evaluation

### Layer 4 - Failure Analysis

Failures return structured objects such as:

```json
{
  "status": "fail",
  "failure_type": "hallucination",
  "reason": "1 unsupported claim(s) detected.",
  "confidence": 0.88,
  "evaluator": "constraint.hallucination"
}
```

### Layer 5 - Auto-Repair

`llmtest fix` analyzes failing tests, appends targeted repair instructions to the prompt, retries the baseline model, and writes improved prompt versions under `.llmtest/fixes/`.

## CLI Commands

From `backend/`:

```bash
pip install -r requirements.txt
pip install -e .
```

Then run:

```bash
llmtest run ../tests
llmtest compare ../tests
llmtest report
llmtest fix ../tests --model mistral-mock
```

Artifacts are stored under `.llmtest/`:

- `reports/latest.json`
- `reports/<run_id>.json`
- `prompt_versions.json`
- `fixes/*.json`

## Built-In Constraint Types

- `max_length`
- `min_length`
- `contains_citation`
- `no_hallucination`
- `no_pii_email`
- `no_pii_phone`
- `regex`
- `custom`

`custom` constraints support a plugin-style callable target in `config.callable`.

## Sample Suites

The repo ships with:

- [`tests/qa_tests.json`](/c:/Eval-Driven%20LLM%20Pipeline%20Behavioral%20Contract%20System/llm-contracts/tests/qa_tests.json)
- [`tests/safety_tests.json`](/c:/Eval-Driven%20LLM%20Pipeline%20Behavioral%20Contract%20System/llm-contracts/tests/safety_tests.json)
- [`tests/reasoning_tests.json`](/c:/Eval-Driven%20LLM%20Pipeline%20Behavioral%20Contract%20System/llm-contracts/tests/reasoning_tests.json)
- [`tests/models.json`](/c:/Eval-Driven%20LLM%20Pipeline%20Behavioral%20Contract%20System/llm-contracts/tests/models.json)

These demonstrate:

- test suites like `pytest` for LLM outputs
- model comparison with pass rates, failure breakdown, and latency
- prompt version tracking
- repairable prompt retries

## Backend and Dashboard

The original API and frontend are still present:

- FastAPI backend in [`backend/app/main.py`](/c:/Eval-Driven%20LLM%20Pipeline%20Behavioral%20Contract%20System/llm-contracts/backend/app/main.py)
- React dashboard in [`frontend/src/pages/Dashboard.jsx`](/c:/Eval-Driven%20LLM%20Pipeline%20Behavioral%20Contract%20System/llm-contracts/frontend/src/pages/Dashboard.jsx)

API endpoints:

- `POST /trace`
- `GET /contracts`
- `GET /results`
- `GET /results/{trace_id}`
- `GET /results/stats`

The semantic evaluator now includes a deterministic fallback when no Groq key is available, so hallucination checks remain useful in local test runs.

## Quick Start

### Backend API

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Dashboard

```bash
cd frontend
npm install
npm run dev
```

### Demo Trace Pipeline

```bash
cd demo
pip install -r requirements.txt
python run_demo.py
```

### Tests

```bash
cd backend
pytest tests -v
```

## Tech Stack

- FastAPI
- SQLAlchemy async + Postgres/SQLite
- LangGraph + Groq for semantic judging when configured
- React + Vite
- JSON suite runner with local artifact storage

## Why This Is Different

This project is not just "evaluate outputs." The differentiator is behavioral contracts plus auto-repair:

- define what must hold
- detect exactly how behavior failed
- tighten the prompt automatically
- re-run and measure whether quality improved
