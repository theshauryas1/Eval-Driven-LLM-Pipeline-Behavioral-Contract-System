"""
Semantic evaluator — LangGraph 3-step faithfulness judge powered by Groq.

Week 1: Returns a stub result so the rest of the system can run without API keys.
Week 3: The full LangGraph agent replaces the stub.

The agent runs three nodes:
  1. extract_claims  — pull every factual claim from the LLM output
  2. match_to_context — find best matching sentence in retrieved context per claim
  3. flag_contradictions — identify claims not supported by context

Returns: EvalResult with structured JSON explanation containing the reasoning trace.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TypedDict

GROQ_AVAILABLE = False
try:
    from langgraph.graph import StateGraph, END
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    GROQ_AVAILABLE = bool(os.getenv("GROQ_API_KEY"))
except ImportError:
    pass


@dataclass
class EvalResult:
    contract_id: str
    passed: bool
    explanation: str
    reasoning_trace: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    output: str
    context: str
    claims: list[str]
    matched: list[dict]
    verdict: dict


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _make_llm() -> "ChatGroq":
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def extract_claims(state: AgentState) -> AgentState:
    llm = _make_llm()
    prompt = (
        "You are a claim extractor. Given the following LLM-generated response, "
        "list every distinct factual claim, one per line. Do not include opinions, "
        "hedges, or meta-statements — only verifiable factual claims.\n\n"
        f"Response:\n{state['output']}\n\n"
        "Output only the claims, one per line. If there are no factual claims, "
        "output NONE."
    )
    result = llm.invoke([HumanMessage(content=prompt)])
    raw = result.content.strip()
    if raw.upper() == "NONE" or not raw:
        claims = []
    else:
        claims = [line.strip("- •").strip() for line in raw.splitlines() if line.strip()]
    state["claims"] = claims
    return state


def match_to_context(state: AgentState) -> AgentState:
    if not state["claims"]:
        state["matched"] = []
        return state

    llm = _make_llm()
    claims_str = "\n".join(f"- {c}" for c in state["claims"])
    prompt = (
        "You are a fact-checker. For each claim below, find the best matching sentence "
        "from the retrieved context. Rate similarity as: high, medium, low, or none.\n\n"
        f"Claims:\n{claims_str}\n\n"
        f"Retrieved Context:\n{state['context']}\n\n"
        "Output a JSON array, where each element has:\n"
        '  { "claim": "...", "best_match": "...", "similarity": "high|medium|low|none" }\n'
        "Output ONLY the JSON array, no other text."
    )
    result = llm.invoke([HumanMessage(content=prompt)])
    raw = result.content.strip()
    # Strip markdown code fences if present
    raw = raw.strip("```json").strip("```").strip()
    try:
        matched = json.loads(raw)
    except json.JSONDecodeError:
        matched = [
            {"claim": c, "best_match": "", "similarity": "none"} for c in state["claims"]
        ]
    state["matched"] = matched
    return state


def flag_contradictions(state: AgentState) -> AgentState:
    low_support = [
        m for m in state["matched"] if m.get("similarity") in ("low", "none")
    ]

    if not low_support:
        state["verdict"] = {
            "passed": True,
            "violations": [],
            "explanation": (
                "All factual claims are supported by the retrieved context."
            ),
        }
        return state

    llm = _make_llm()
    items_str = "\n".join(
        f"- Claim: \"{m['claim']}\" | Context match: \"{m.get('best_match', 'N/A')}\" "
        f"| Similarity: {m.get('similarity')}"
        for m in low_support
    )
    prompt = (
        "You are a faithfulness judge. The following claims from an LLM response "
        "are either not found in the retrieved context or only weakly supported. "
        "For each, write ONE sentence explaining why it is unfaithful or hallucinated.\n\n"
        f"{items_str}\n\n"
        "Output a JSON object:\n"
        '{ "passed": false, "violations": ["explanation1", ...], "explanation": "summary" }\n'
        "Output ONLY the JSON, no other text."
    )
    result = llm.invoke([HumanMessage(content=prompt)])
    raw = result.content.strip().strip("```json").strip("```").strip()
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        verdict = {
            "passed": False,
            "violations": [m["claim"] for m in low_support],
            "explanation": f"{len(low_support)} claim(s) not grounded in context.",
        }
    state["verdict"] = verdict
    return state


# ---------------------------------------------------------------------------
# Public evaluator class
# ---------------------------------------------------------------------------

class SemanticEvaluator:
    """
    Evaluates semantic faithfulness contracts.

    If GROQ_API_KEY is set and langgraph/langchain-groq are installed,
    runs the full 3-step LangGraph agent. Otherwise returns a stub.
    """

    def __init__(self):
        self._graph = None
        if GROQ_AVAILABLE:
            self._graph = self._build_graph()

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
            return EvalResult(
                contract_id=contract_id,
                passed=True,
                explanation=(
                    "⚠ Semantic evaluation pending — set GROQ_API_KEY to enable "
                    "the LangGraph faithfulness judge."
                ),
                reasoning_trace=[],
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
