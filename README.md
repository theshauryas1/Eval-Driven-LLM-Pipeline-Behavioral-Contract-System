# Eval-Driven LLM Pipeline Behavioral Contract System

Evaluation-driven LLM quality platform for testing outputs, detecting hallucinations and policy violations, and tracking regressions over time.

## What It Does

This project gives you two product surfaces:

- a contract-based `llmtest` CLI for running repeatable LLM evaluation suites
- a FastAPI + React dashboard for trace ingestion, live inspection, and contract monitoring

## Why It Matters

Most LLM apps fail silently. This system makes failures explicit by checking every output against behavioral contracts such as citation requirements, PII leakage, and context faithfulness.

## Demo

- API demo endpoint: `POST /trace/demo`
- Frontend dashboard: deploy the Vite app on Vercel and point `VITE_API_URL` at the Render backend
- Suggested live flow: run the demo endpoint, then open the Trace Inspector to show the failed contracts and reasoning trace

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
- `POST /trace/demo`
- `GET /contracts`
- `GET /results`
- `GET /results/{trace_id}`
- `GET /results/stats`

The semantic evaluator includes:

- deterministic fallback when no Groq key is available
- retry handling for Groq rate limits
- bounded in-process concurrency for burst control
- in-memory prompt-response caching for repeated semantic checks

The trace ingestion endpoints also include a simple in-memory per-IP rate limiter for demo deployments.

## Quick Start

### Backend API

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Instant demo:

```bash
curl -X POST http://127.0.0.1:8000/trace/demo ^
  -H "Content-Type: application/json" ^
  -d "{\"scenario\":\"hallucination\"}"
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

## Deployment

### Render Backend

The repo already includes Render config in [render.yaml](/c:/Eval-Driven%20LLM%20Pipeline%20Behavioral%20Contract%20System/llm-contracts/render.yaml). A legacy Railway config still exists in [backend/railway.json](/c:/Eval-Driven%20LLM%20Pipeline%20Behavioral%20Contract%20System/llm-contracts/backend/railway.json), but Render is the current backend target.

Use these settings:

```bash
Root Directory: backend
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Recommended environment variables:

```bash
CORS_ORIGIN=https://your-frontend-domain.vercel.app,https://your-preview-domain.vercel.app
DATABASE_URL=postgresql+asyncpg://user:password@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant
GROQ_FALLBACK_MODEL=llama-3.3-70b-versatile
GROQ_MAX_RETRIES=3
GROQ_RETRY_BASE_DELAY=2
GROQ_MAX_CONCURRENCY=2
CONTRACTS_YAML_PATH=contracts/example_contracts.yaml
TRACE_RATE_LIMIT_MAX_REQUESTS=5
TRACE_RATE_LIMIT_WINDOW_SECONDS=60
```

Notes:

- `DATABASE_URL` is optional for first deploy because the app falls back to SQLite
- `GROQ_API_KEY` is optional because semantic evaluation has a deterministic fallback
- the current rate limiter is in-memory and per-process, which is appropriate for demos and single-instance deploys
- for multi-instance production traffic, move rate limiting to Redis or your API gateway

### Vercel Frontend

Deploy `frontend/` and set:

```bash
VITE_API_URL=https://your-backend-name.onrender.com
```

The repo already includes SPA rewrites in [frontend/vercel.json](/c:/Eval-Driven%20LLM%20Pipeline%20Behavioral%20Contract%20System/llm-contracts/frontend/vercel.json).

### Neon Database

Create a Neon Postgres database and paste its pooled `postgresql+asyncpg://...?...sslmode=require` connection string into Render as `DATABASE_URL`.

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
