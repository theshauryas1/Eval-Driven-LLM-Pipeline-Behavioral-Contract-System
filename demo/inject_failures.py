"""
Deliberate failure injection for the LLM Contract System demo.

Injects 3 well-defined failure modes into the RAG pipeline to demonstrate
contract detection on the dashboard:

  1. PII Leak         → response contains an email address (no_pii_email fires)
  2. Missing Citation → response has no [Source:...] marker (always_cite_source fires)
  3. Hallucination   → response contradicts the retrieved context (context_faithfulness fires)

Usage:
    python inject_failures.py
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

from rag_pipeline import build_pipeline, run_query, send_trace, DOCS

# ── Failure 1: PII Leak ───────────────────────────────────────────────────────
def inject_pii_leak(chain, retriever):
    """
    Appends a fake email address to an otherwise correct response.
    Fires: no_pii_email
    """
    print("\n" + "═" * 60)
    print("  FAILURE 1: PII Leak")
    print("  Contract expected to fire: no_pii_email")
    print("═" * 60)

    question = "Where is the Eiffel Tower and how can I contact support?"
    result = chain.invoke({"query": question})
    clean_answer = result["result"]

    # Inject PII
    poisoned = clean_answer + "\n\nFor further assistance, contact support@eiffeltower-demo.com."

    source_docs = result.get("source_documents", [])
    context = "\n\n".join(d.page_content for d in source_docs)

    print(f"\n  Q: {question}")
    print(f"  A (poisoned): {poisoned[:200]}")

    trace_result = send_trace(
        input_text=question,
        retrieved_context=context,
        output=poisoned,
    )
    if trace_result:
        print(f"  ✓ Trace sent → trace_id: {trace_result.get('trace_id', '?')}")
    print("  → Check dashboard: no_pii_email should be FAIL")


# ── Failure 2: Missing Citation ───────────────────────────────────────────────
def inject_no_citation(chain, retriever):
    """
    Strips all [Source:...] markers from the response.
    Fires: always_cite_source
    """
    import re

    print("\n" + "═" * 60)
    print("  FAILURE 2: Missing Citation")
    print("  Contract expected to fire: always_cite_source")
    print("═" * 60)

    question = "What is artificial intelligence?"
    result = chain.invoke({"query": question})
    clean_answer = result["result"]

    # Strip all citation markers
    citation_pattern = r"\[(?:Source|Chunk|Doc|Ref)[^\]]*\]"
    poisoned = re.sub(citation_pattern, "", clean_answer, flags=re.IGNORECASE).strip()

    source_docs = result.get("source_documents", [])
    context = "\n\n".join(d.page_content for d in source_docs)

    print(f"\n  Q: {question}")
    print(f"  A (no citations): {poisoned[:200]}")

    trace_result = send_trace(
        input_text=question,
        retrieved_context=context,
        output=poisoned,
    )
    if trace_result:
        print(f"  ✓ Trace sent → trace_id: {trace_result.get('trace_id', '?')}")
    print("  → Check dashboard: always_cite_source should be FAIL")


# ── Failure 3: Hallucinated Claim ────────────────────────────────────────────
def inject_hallucination(chain, retriever):
    """
    Replaces factual content with a contradicting invented claim.
    Fires: context_faithfulness (LangGraph semantic judge)
    """
    print("\n" + "═" * 60)
    print("  FAILURE 3: Hallucinated Claim")
    print("  Contract expected to fire: context_faithfulness")
    print("═" * 60)

    question = "Where was the Eiffel Tower built and when?"

    # Use the Eiffel Tower document as context
    context = DOCS[0].page_content

    # Deliberately fabricated answer that contradicts the context
    hallucinated = (
        "The Eiffel Tower was built in London, England, in 1754 as a symbol of "
        "British industrial power during the Victorian era. [Source: wiki_eiffel_tower]"
    )

    print(f"\n  Q: {question}")
    print(f"  Context says: Paris, France, 1887–1889")
    print(f"  A (hallucinated): {hallucinated}")

    trace_result = send_trace(
        input_text=question,
        retrieved_context=context,
        output=hallucinated,
    )
    if trace_result:
        print(f"  ✓ Trace sent → trace_id: {trace_result.get('trace_id', '?')}")
    print("  → Check dashboard: context_faithfulness should be FAIL (requires GROQ_API_KEY)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building RAG pipeline for failure injection…")
    chain, retriever = build_pipeline()

    inject_pii_leak(chain, retriever)
    inject_no_citation(chain, retriever)
    inject_hallucination(chain, retriever)

    print("\n\n" + "=" * 60)
    print("  All 3 failures injected.")
    print("  Open the dashboard → Traces to see violations.")
    print("  Open Regression to see when each contract degraded.")
    print("=" * 60)
