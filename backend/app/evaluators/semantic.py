"""
Semantic evaluator for faithfulness and hallucination checks.

If Groq is configured, this uses the existing 3-step LangGraph judge.
Otherwise it falls back to a deterministic lexical-support check so local
test runs still produce meaningful failure diagnostics.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TypedDict

GROQ_AVAILABLE = False
try:
    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq
    from langgraph.graph import END, StateGraph

    GROQ_AVAILABLE = bool(os.getenv("GROQ_API_KEY"))
except ImportError:
    pass


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


@dataclass
class EvalResult:
    contract_id: str
    passed: bool
    explanation: str
    reasoning_trace: list[dict] = field(default_factory=list)


class AgentState(TypedDict):
    output: str
    context: str
    claims: list[str]
    matched: list[dict]
    verdict: dict


def _make_llm() -> "ChatGroq":
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOPWORDS
    }


def _split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(stripped) if part.strip()]
    return parts or [stripped]


def _extract_claims_fallback(output: str) -> list[str]:
    claims = []
    for sentence in _split_sentences(output):
        normalized = sentence.lower()
        if any(
            phrase in normalized
            for phrase in (
                "do not know",
                "don't know",
                "cannot determine",
                "can't determine",
                "not enough context",
                "insufficient context",
            )
        ):
            continue
        tokens = _tokenize(sentence)
        if len(tokens) >= 3:
            claims.append(sentence)
    return claims


def _similarity_score(claim: str, context_sentence: str) -> float:
    claim_tokens = _tokenize(claim)
    context_tokens = _tokenize(context_sentence)
    if not claim_tokens or not context_tokens:
        return 0.0
    return len(claim_tokens & context_tokens) / len(claim_tokens)


def _label_similarity(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    if score >= 0.25:
        return "low"
    return "none"


def _run_fallback(output: str, context: str) -> tuple[list[str], list[dict], dict]:
    claims = _extract_claims_fallback(output)
    context_sentences = _split_sentences(context)
    matched: list[dict] = []

    for claim in claims:
        best_sentence = ""
        best_score = 0.0
        for context_sentence in context_sentences:
            score = _similarity_score(claim, context_sentence)
            if score > best_score:
                best_score = score
                best_sentence = context_sentence

        matched.append(
            {
                "claim": claim,
                "best_match": best_sentence,
                "similarity": _label_similarity(best_score),
                "score": round(best_score, 2),
            }
        )

    low_support = [item for item in matched if item["similarity"] in {"low", "none"}]
    if not claims:
        verdict = {
            "passed": True,
            "violations": [],
            "explanation": "No factual claims detected in output.",
        }
    elif not context_sentences:
        verdict = {
            "passed": False,
            "violations": claims,
            "explanation": "No retrieved context provided to support factual claims.",
        }
    elif not low_support:
        verdict = {
            "passed": True,
            "violations": [],
            "explanation": "All detected claims have medium or high support in context.",
        }
    else:
        verdict = {
            "passed": False,
            "violations": [
                f"Claim '{item['claim']}' is only {item['similarity']} support."
                for item in low_support
            ],
            "explanation": f"{len(low_support)} unsupported claim(s) detected.",
        }

    return claims, matched, verdict


def extract_claims(state: AgentState) -> AgentState:
    llm = _make_llm()
    prompt = (
        "You are a claim extractor. Given the following LLM-generated response, "
        "list every distinct factual claim, one per line. Do not include opinions, "
        "hedges, or meta-statements - only verifiable factual claims.\n\n"
        f"Response:\n{state['output']}\n\n"
        "Output only the claims, one per line. If there are no factual claims, output NONE."
    )
    result = llm.invoke([HumanMessage(content=prompt)])
    raw = result.content.strip()
    if raw.upper() == "NONE" or not raw:
        claims = []
    else:
        claims = [line.lstrip("-* ").strip() for line in raw.splitlines() if line.strip()]
    state["claims"] = claims
    return state


def match_to_context(state: AgentState) -> AgentState:
    if not state["claims"]:
        state["matched"] = []
        return state

    llm = _make_llm()
    claims_str = "\n".join(f"- {claim}" for claim in state["claims"])
    prompt = (
        "You are a fact-checker. For each claim below, find the best matching sentence "
        "from the retrieved context. Rate similarity as: high, medium, low, or none.\n\n"
        f"Claims:\n{claims_str}\n\n"
        f"Retrieved Context:\n{state['context']}\n\n"
        "Output a JSON array where each element has "
        '{ "claim": "...", "best_match": "...", "similarity": "high|medium|low|none" }.'
    )
    result = llm.invoke([HumanMessage(content=prompt)])
    raw = result.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        matched = json.loads(raw)
    except json.JSONDecodeError:
        matched = [
            {"claim": claim, "best_match": "", "similarity": "none"}
            for claim in state["claims"]
        ]
    state["matched"] = matched
    return state


def flag_contradictions(state: AgentState) -> AgentState:
    low_support = [item for item in state["matched"] if item.get("similarity") in {"low", "none"}]
    if not low_support:
        state["verdict"] = {
            "passed": True,
            "violations": [],
            "explanation": "All factual claims are supported by the retrieved context.",
        }
        return state

    llm = _make_llm()
    items_str = "\n".join(
        f"- Claim: \"{item['claim']}\" | Context match: \"{item.get('best_match', 'N/A')}\" | Similarity: {item.get('similarity')}"
        for item in low_support
    )
    prompt = (
        "You are a faithfulness judge. The following claims from an LLM response "
        "are not sufficiently supported by retrieved context. For each one, write a "
        "single sentence explaining why it is hallucinated or unsupported.\n\n"
        f"{items_str}\n\n"
        "Return JSON with keys passed, violations, and explanation."
    )
    result = llm.invoke([HumanMessage(content=prompt)])
    raw = result.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        verdict = {
            "passed": False,
            "violations": [item["claim"] for item in low_support],
            "explanation": f"{len(low_support)} claim(s) not grounded in context.",
        }
    state["verdict"] = verdict
    return state


class SemanticEvaluator:
    """
    Evaluates semantic faithfulness contracts.

    When an API key is unavailable, a deterministic fallback still flags
    unsupported claims instead of returning a pass-through stub.
    """

    def __init__(self):
        self._graph = self._build_graph() if GROQ_AVAILABLE else None

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("extract_claims", extract_claims)
        graph.add_node("match_to_context", match_to_context)
        graph.add_node("flag_contradictions", flag_contradictions)
        graph.set_entry_point("extract_claims")
        graph.add_edge("extract_claims", "match_to_context")
        graph.add_edge("match_to_context", "flag_contradictions")
        graph.add_edge("flag_contradictions", END)
        return graph.compile()

    def evaluate(
        self,
        contract_id: str,
        config: dict,
        output: str,
        retrieved_context: str = "",
        **_kwargs,
    ) -> EvalResult:
        if not GROQ_AVAILABLE or self._graph is None:
            claims, matched, verdict = _run_fallback(output, retrieved_context)
            reasoning_trace = [
                {"step": "extract_claims", "result": claims},
                {"step": "match_to_context", "result": matched},
                {"step": "flag_contradictions", "result": verdict},
            ]
            return EvalResult(
                contract_id=contract_id,
                passed=verdict.get("passed", True),
                explanation=verdict.get("explanation", ""),
                reasoning_trace=reasoning_trace,
            )

        initial_state: AgentState = {
            "output": output,
            "context": retrieved_context,
            "claims": [],
            "matched": [],
            "verdict": {},
        }
        final_state = self._graph.invoke(initial_state)
        verdict = final_state.get("verdict", {})
        reasoning_trace = [
            {"step": "extract_claims", "result": final_state.get("claims", [])},
            {"step": "match_to_context", "result": final_state.get("matched", [])},
            {"step": "flag_contradictions", "result": verdict},
        ]
        return EvalResult(
            contract_id=contract_id,
            passed=verdict.get("passed", True),
            explanation=verdict.get("explanation", ""),
            reasoning_trace=reasoning_trace,
        )
