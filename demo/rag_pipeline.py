"""
Demo RAG Pipeline — instrumented with LLM Contract System.

A minimal LangChain retrieval-augmented generation app over 5 sample documents.
Every answer is sent to POST /trace for evaluation against all active contracts.

Usage:
    python rag_pipeline.py

Requirements:
    pip install langchain langchain-community langchain-groq faiss-cpu requests python-dotenv
"""
from __future__ import annotations

import os
import json
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings  # no API key needed for demo
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ── Sample documents ─────────────────────────────────────────────────────────
DOCS = [
    Document(
        page_content=(
            "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars "
            "in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of "
            "the 1889 World's Fair."
        ),
        metadata={"source": "wiki_eiffel_tower"},
    ),
    Document(
        page_content=(
            "Artificial intelligence (AI) is the simulation of human intelligence in machines "
            "programmed to think and learn. Machine learning is a subset of AI that allows "
            "systems to learn from data without explicit programming."
        ),
        metadata={"source": "wiki_ai"},
    ),
    Document(
        page_content=(
            "The Python programming language was created by Guido van Rossum and first released "
            "in 1991. Python's design philosophy emphasizes code readability and simplicity."
        ),
        metadata={"source": "wiki_python"},
    ),
    Document(
        page_content=(
            "Climate change refers to long-term shifts in global temperatures and weather patterns. "
            "Since the 1800s, human activities have been the main driver, primarily through "
            "burning fossil fuels."
        ),
        metadata={"source": "wiki_climate"},
    ),
    Document(
        page_content=(
            "The Great Wall of China is a series of fortifications built along the historical "
            "northern borders of ancient Chinese states. Construction began in the 7th century BC."
        ),
        metadata={"source": "wiki_great_wall"},
    ),
]

# ── Contract system endpoint ──────────────────────────────────────────────────
API_URL = os.getenv("CONTRACT_API_URL", "http://localhost:8000")
PIPELINE_ID = "demo-rag-pipeline"


def send_trace(input_text: str, retrieved_context: str, output: str) -> dict:
    """POST a trace to the contract evaluation system."""
    try:
        resp = requests.post(
            f"{API_URL}/trace",
            json={
                "pipeline_id": PIPELINE_ID,
                "input_text": input_text,
                "retrieved_context": retrieved_context,
                "output": output,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ⚠ Could not send trace to contract system: {e}")
        return {}


# ── RAG pipeline builder ──────────────────────────────────────────────────────
def build_pipeline():
    embeddings = FakeEmbeddings(size=128)  # deterministic fake embeddings for demo
    vectorstore = FAISS.from_documents(DOCS, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to backend/.env or set it in your environment."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        api_key=groq_api_key,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Use only the context below to answer the question. "
            "Always cite your source using [Source: <source_name>].\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain, retriever


def run_query(chain, retriever, question: str, override_output: str | None = None) -> str:
    """Run a RAG query, instrument to contract system, return answer."""
    print(f"\n{'─' * 60}")
    print(f"  Q: {question}")

    result = chain.invoke({"query": question})
    answer = override_output if override_output is not None else result["result"]
    source_docs = result.get("source_documents", [])
    context = "\n\n".join(d.page_content for d in source_docs)

    print(f"  A: {answer[:120]}{'…' if len(answer) > 120 else ''}")

    trace_result = send_trace(
        input_text=question,
        retrieved_context=context,
        output=answer,
    )
    if trace_result:
        print(f"  ✓ Trace sent → trace_id: {trace_result.get('trace_id', '?')}")

    return answer


if __name__ == "__main__":
    print("Building RAG pipeline…")
    chain, retriever = build_pipeline()

    print("\n=== Normal Queries ===")
    run_query(chain, retriever, "Where is the Eiffel Tower located?")
    run_query(chain, retriever, "What is machine learning?")
    run_query(chain, retriever, "When was Python first released?")

    print("\n\n✓ Normal queries complete. Check the dashboard for results.")
