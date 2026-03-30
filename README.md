# Eval-Driven LLM Pipeline Behavioral Contract System

A deployable system where teams **define what their LLM pipeline must always do** and the system automatically detects when it stops doing that — with a live dashboard showing the history of violations.

[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/Eval%20Agent-LangGraph-7C3AED?style=flat)](https://www.langchain.com/langgraph)
[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?style=flat&logo=react)](https://vitejs.dev)
[![Database](https://img.shields.io/badge/DB-Neon%20Postgres-00E5CC?style=flat)](https://neon.tech)
[![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-orange?style=flat)](https://console.groq.com)

---

## Architecture

```
Frontend (React/Vite → Vercel)
        │ REST API
        ▼
Backend (FastAPI → Railway)
   ├── POST /trace        ← receives LLM pipeline traces
   ├── GET  /contracts    ← lists active contracts + pass rates
   └── GET  /results      ← paginated traces + eval details
        │
        ▼
Contract Evaluator Engine
   ├── Structural  — deterministic rule checks (citations, length)
   ├── Pattern     — regex checks (PII: email, phone)
   └── Semantic    — LangGraph 3-step faithfulness judge
                       extract_claims → match_to_context → flag_contradictions
                       (Groq / Llama 3.3 70B)
        │
        ▼
Neon Postgres (traces · contracts · eval_results)
```

## Contract YAML Format

Define behavioral specs any non-engineer can read:

```yaml
contracts:
  - id: always_cite_source
    type: structural
    description: Response must reference at least one retrieved chunk
    config:
      min_citations: 1

  - id: no_pii_email
    type: pattern
    description: Response must not contain email addresses
    config:
      pattern: '[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
      must_not_match: true

  - id: context_faithfulness
    type: semantic
    description: Claims in response must be grounded in retrieved context
```

Three contract types:
| Type | How it works | Speed |
|------|-------------|-------|
| `structural` | Deterministic rules (citation count, min length) | < 1ms |
| `pattern` | Compiled regex (PII, format enforcement) | < 1ms |
| `semantic` | LangGraph agent — 3-step LLM judge via Groq | ~2–5s |

## Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/theshauryas1/Eval-Driven-LLM-Pipeline-Behavioral-Contract-System.git
cd Eval-Driven-LLM-Pipeline-Behavioral-Contract-System

# Copy and fill in env vars
cp backend/.env.example backend/.env
# Edit backend/.env with your GROQ_API_KEY and DATABASE_URL (Neon)
```

### 2. Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 3. Run the dashboard

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### 4. Run the demo

```bash
cd demo
pip install -r requirements.txt
python run_demo.py
# Sends 3 normal queries + 3 deliberate failures to the contract system
# Watch violations appear on the dashboard in real time
```

### 5. Run unit tests

```bash
cd backend
pytest tests/ -v
# 25+ unit tests across all 3 evaluator types
```

## The Semantic Faithfulness Agent (Week 3)

The differentiator of this project is the **LangGraph 3-step reasoning agent** that judges whether LLM responses are grounded in retrieved context:

```
Node 1 — extract_claims
  "List every factual claim in this response, one per line."
         ↓
Node 2 — match_to_context
  "For each claim, find the best matching sentence in context.
   Rate similarity: high / medium / low / none."
         ↓
Node 3 — flag_contradictions
  "For each unsupported claim, explain why it is hallucinated."
         ↓
Output: { passed: bool, violations: [...], explanation: "..." }
```

Every step's output is stored as a JSON reasoning trace in Postgres and rendered inline in the Trace Inspector on the dashboard.

## Demo: 3 Deliberate Failures

Run `demo/inject_failures.py` to fire each contract:

| Failure | What happens | Contract fired |
|---------|-------------|----------------|
| PII Leak | Email appended to response | `no_pii_email` |
| Missing Citation | `[Source:...]` stripped | `always_cite_source` |
| Hallucination | Eiffel Tower → "London, 1754" | `context_faithfulness` |

## Deployment

| Component | Platform | Cost |
|-----------|----------|------|
| FastAPI backend | Railway | $0 (free $5/mo credit) |
| React dashboard | Vercel | $0 (free) |
| Postgres | Neon | $0 (free tier, 0.5GB) |
| LLM judge | Groq API | $0 (free tier) |

**Total monthly cost: $0**

## Project Structure

```
├── contracts/
│   └── example_contracts.yaml      # Contract definitions
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI entrypoint
│   │   ├── models.py               # SQLAlchemy ORM (traces, contracts, eval_results)
│   │   ├── database.py             # Async Postgres/SQLite connection
│   │   ├── api/
│   │   │   ├── traces.py           # POST /trace
│   │   │   ├── contracts.py        # GET /contracts
│   │   │   └── results.py          # GET /results, /results/{id}, /results/stats
│   │   └── evaluators/
│   │       ├── engine.py           # Orchestrates all evaluators
│   │       ├── contract_loader.py  # YAML parser + Pydantic validation
│   │       ├── structural.py       # Deterministic structural checks
│   │       ├── pattern.py          # Regex PII/format checks
│   │       └── semantic.py         # LangGraph 3-step faithfulness judge
│   ├── tests/                      # 25+ unit tests
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── Dashboard.jsx       # Contract cards + sparklines
│       │   ├── TracesPage.jsx      # Trace list + inspector modal
│       │   └── RegressionPage.jsx  # Pass-rate line charts
│       └── components/
│           └── Sidebar.jsx
└── demo/
    ├── rag_pipeline.py             # Instrumented LangChain RAG app
    ├── inject_failures.py          # 3 deliberate failure modes
    └── run_demo.py                 # Full demo orchestrator
```

## Tech Stack

- **FastAPI** — async Python backend with BackgroundTasks for non-blocking eval
- **SQLAlchemy (async) + asyncpg** — Postgres ORM, falls back to SQLite for local dev
- **LangGraph** — multi-step agent graph for semantic evaluation
- **Groq / Llama 3.3 70B** — fast free LLM inference for the faithfulness judge
- **React + Vite + Recharts** — dashboard with live charts and trace inspector
- **Neon** — serverless Postgres with no spin-down on free tier

---

> Built as a portfolio project demonstrating eval-driven LLM engineering.
> The LangGraph semantic agent is the technical differentiator — every interview walkthrough starts there.
