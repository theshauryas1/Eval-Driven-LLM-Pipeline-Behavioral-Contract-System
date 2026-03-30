"""
Demo orchestrator — run all queries then inject all 3 failures.

Usage:
    cd demo/
    python run_demo.py

This will:
  1. Run 3 normal RAG queries (should all pass contracts)
  2. Inject PII leak → no_pii_email fires
  3. Inject missing citation → always_cite_source fires
  4. Inject hallucinated claim → context_faithfulness fires (requires GROQ_API_KEY)

Then open: http://localhost:5173 to see all violations on the dashboard.
"""
import time
from rag_pipeline import build_pipeline, run_query
from inject_failures import inject_pii_leak, inject_no_citation, inject_hallucination

if __name__ == "__main__":
    print("=" * 60)
    print("  LLM Contract System — Demo Script")
    print("  Make sure the FastAPI backend is running (uvicorn app.main:app)")
    print("=" * 60)

    print("\nBuilding RAG pipeline…")
    chain, retriever = build_pipeline()

    print("\n[Phase 1] Normal queries — expect all contracts to PASS")
    run_query(chain, retriever, "Where is the Eiffel Tower located?")
    time.sleep(1)
    run_query(chain, retriever, "What is machine learning?")
    time.sleep(1)
    run_query(chain, retriever, "When was Python created?")

    print("\n\n[Phase 2] Failure injection — watch contracts FAIL on dashboard")
    time.sleep(2)
    inject_pii_leak(chain, retriever)
    time.sleep(2)
    inject_no_citation(chain, retriever)
    time.sleep(2)
    inject_hallucination(chain, retriever)

    print("\n\n" + "=" * 60)
    print("  Demo complete!")
    print("  → Dashboard: http://localhost:5173")
    print("  → Traces: http://localhost:5173/traces")
    print("  → Regression: http://localhost:5173/regression")
    print("=" * 60)
