"""
Semantic evaluator for faithfulness and hallucination checks.

If Groq is configured, this uses the existing 3-step LangGraph judge.
Otherwise it falls back to a deterministic lexical-support check so local
test runs still produce meaningful failure diagnostics.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
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


logger = logging.getLogger(__name__)

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

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")
MAX_GROQ_RETRIES = max(1, int(os.getenv("GROQ_MAX_RETRIES", "3")))
GROQ_BASE_DELAY_SECONDS = float(os.getenv("GROQ_RETRY_BASE_DELAY", "2"))
GROQ_MAX_CONCURRENCY = max(1, int(os.getenv("GROQ_MAX_CONCURRENCY", "2")))

_groq_gate = threading.BoundedSemaphore(GROQ_MAX_CONCURRENCY)
_groq_cache: dict[tuple[str, str], str] = {}
_groq_cache_lock = threading.Lock()


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


def _make_llm(model_name: str | None = None) -> "ChatGroq":
    return ChatGroq(
        model=model_name or DEFAULT_GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), GROQ_BASE_DELAY_SECONDS)
            except ValueError:
                return None

    text = str(exc)
    match = re.search(r"retry-after[:= ]+(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return max(float(match.group(1)), GROQ_BASE_DELAY_SECONDS)
    return None


def _is_retryable_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def _invoke_with_resilience(prompt: str) -> str:
    cache_key = (DEFAULT_GROQ_MODEL, prompt)
    with _groq_cache_lock:
        cached = _groq_cache.get(cache_key)
    if cached is not None:
        return cached

    errors: list[str] = []
    models_to_try = [DEFAULT_GROQ_MODEL]
    if DEFAULT_GROQ_FALLBACK_MODEL != DEFAULT_GROQ_MODEL:
        models_to_try.append(DEFAULT_GROQ_FALLBACK_MODEL)

    for model_name in models_to_try:
        for attempt in range(1, MAX_GROQ_RETRIES + 1):
            with _groq_gate:
                try:
                    logger.info(
                        "Calling Groq semantic judge",
                        extra={"model": model_name, "attempt": attempt},
                    )
                    response = _make_llm(model_name).invoke([HumanMessage(content=prompt)])
                    content = response.content.strip()
                    with _groq_cache_lock:
                        _groq_cache[cache_key] = content
                    return content
                except Exception as exc:
                    errors.append(f"{model_name}: {exc}")
                    logger.warning(
                        "Groq semantic judge call failed",
                        extra={"model": model_name, "attempt": attempt, "error": str(exc)},
                    )
                    if not _is_retryable_rate_limit(exc) or attempt == MAX_GROQ_RETRIES:
                        break
                    delay = _extract_retry_after_seconds(exc) or (GROQ_BASE_DELAY_SECONDS * attempt)
                    time.sleep(delay)

    raise RuntimeError("Groq semantic evaluation failed after retries: " + " | ".join(errors))


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
    matched = _fallback_match_claims(claims, context)
    verdict = _build_verdict_from_matches(claims, matched, context)
    return claims, matched, verdict


def _fallback_match_claims(claims: list[str], context: str) -> list[dict]:
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

    return matched


def _build_verdict_from_matches(
    claims: list[str],
    matched: list[dict],
    context: str,
) -> dict:
    context_sentences = _split_sentences(context)

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

    return verdict


def _coerce_passed(value: object, violations: list[object]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "pass", "passed", "yes"}:
            return True
        if normalized in {"false", "fail", "failed", "no"}:
            return False
    if isinstance(value, list):
        return len(value) == 0 and len(violations) == 0
    if value is None:
        return len(violations) == 0
    return bool(value)


def _normalize_violations(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if value in (None, "", False):
        return []
    return [value]


def _normalize_verdict(verdict: object, low_support: list[dict] | None = None) -> dict:
    fallback_explanation = (
        f"{len(low_support or [])} claim(s) not grounded in context."
        if low_support is not None
        else "Semantic evaluation returned an invalid verdict."
    )
    fallback_violations = [item["claim"] for item in (low_support or [])]

    if not isinstance(verdict, dict):
        return {
            "passed": len(fallback_violations) == 0,
            "violations": fallback_violations,
            "explanation": fallback_explanation,
        }

    violations = _normalize_violations(verdict.get("violations"))
    if low_support is not None and (
        not violations
        or any(not isinstance(item, (str, dict)) for item in violations)
    ):
        violations = fallback_violations
    explanation = verdict.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = fallback_explanation

    return {
        "passed": _coerce_passed(verdict.get("passed"), violations),
        "violations": violations,
        "explanation": explanation,
    }


def _normalize_match_item(item: object, claim: str) -> dict:
    if not isinstance(item, dict):
        return {"claim": claim, "best_match": "", "similarity": "none"}

    similarity = str(item.get("similarity", "none")).strip().lower()
    if similarity not in {"high", "medium", "low", "none"}:
        similarity = "none"

    best_match = item.get("best_match", "")
    if not isinstance(best_match, str):
        best_match = ""

    item_claim = item.get("claim")
    if not isinstance(item_claim, str) or not item_claim.strip():
        item_claim = claim

    normalized = {
        "claim": item_claim,
        "best_match": best_match,
        "similarity": similarity,
    }

    score = item.get("score")
    if isinstance(score, (int, float)):
        normalized["score"] = round(float(score), 2)
    return normalized


def _claim_is_grounded_in_output(claim: str, output: str) -> bool:
    output_sentences = _split_sentences(output)
    if not output_sentences:
        return False
    best_score = max(_similarity_score(claim, sentence) for sentence in output_sentences)
    return best_score >= 0.5


def _normalize_claims(claims: list[str], output: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        if not isinstance(claim, str):
            continue
        cleaned = claim.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        if not _claim_is_grounded_in_output(cleaned, output):
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def _normalize_matched(claims: list[str], matched: object) -> list[dict]:
    if not isinstance(matched, list) or len(matched) != len(claims):
        return [
            {"claim": claim, "best_match": "", "similarity": "none"}
            for claim in claims
        ]

    return [
        _normalize_match_item(item, claim)
        for claim, item in zip(claims, matched)
    ]


def _should_replace_with_fallback(claims: list[str], matched: list[dict], fallback_matched: list[dict]) -> bool:
    if len(matched) != len(claims):
        return True

    llm_supported = sum(1 for item in matched if item["similarity"] in {"high", "medium"})
    fallback_supported = sum(
        1 for item in fallback_matched if item["similarity"] in {"high", "medium"}
    )

    if fallback_supported > llm_supported:
        return True

    for item, fallback_item in zip(matched, fallback_matched):
        if item["similarity"] == "none" and fallback_item["similarity"] in {"high", "medium"}:
            return True

    return False


def extract_claims(state: AgentState) -> AgentState:
    prompt = (
        "You are a claim extractor. Given the following LLM-generated response, "
        "list every distinct factual claim, one per line. Do not include opinions, "
        "hedges, or meta-statements - only verifiable factual claims.\n\n"
        f"Response:\n{state['output']}\n\n"
        "Output only the claims, one per line. If there are no factual claims, output NONE."
    )
    raw = _invoke_with_resilience(prompt)
    if raw.upper() == "NONE" or not raw:
        claims = []
    else:
        claims = [line.lstrip("-* ").strip() for line in raw.splitlines() if line.strip()]
    fallback_claims = _extract_claims_fallback(state["output"])
    normalized_claims = _normalize_claims(claims, state["output"])
    if not normalized_claims:
        normalized_claims = fallback_claims
    state["claims"] = normalized_claims
    return state


def match_to_context(state: AgentState) -> AgentState:
    if not state["claims"]:
        state["matched"] = []
        return state

    claims_str = "\n".join(f"- {claim}" for claim in state["claims"])
    prompt = (
        "You are a fact-checker. For each claim below, find the best matching sentence "
        "from the retrieved context. Rate similarity as: high, medium, low, or none.\n\n"
        f"Claims:\n{claims_str}\n\n"
        f"Retrieved Context:\n{state['context']}\n\n"
        "Output a JSON array where each element has "
        '{ "claim": "...", "best_match": "...", "similarity": "high|medium|low|none" }.'
    )
    raw = _invoke_with_resilience(prompt).removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        matched = json.loads(raw)
    except json.JSONDecodeError:
        matched = []

    normalized_matched = _normalize_matched(state["claims"], matched)
    fallback_matched = _fallback_match_claims(state["claims"], state["context"])

    if _should_replace_with_fallback(state["claims"], normalized_matched, fallback_matched):
        state["matched"] = fallback_matched
    else:
        state["matched"] = normalized_matched
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
    raw = _invoke_with_resilience(prompt).removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        verdict = {
            "passed": False,
            "violations": [item["claim"] for item in low_support],
            "explanation": f"{len(low_support)} claim(s) not grounded in context.",
        }
    state["verdict"] = _normalize_verdict(verdict, low_support=low_support)
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
        use_groq = GROQ_AVAILABLE and self._graph is not None and config.get("use_groq", True)

        if not use_groq:
            claims, matched, verdict = _run_fallback(output, retrieved_context)
            verdict = _normalize_verdict(verdict)
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

        try:
            final_state = self._graph.invoke(initial_state)
            verdict = _normalize_verdict(final_state.get("verdict", {}))
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
        except Exception as exc:
            logger.exception("Semantic Groq evaluation failed, falling back to deterministic evaluator")
            claims, matched, verdict = _run_fallback(output, retrieved_context)
            verdict = _normalize_verdict(verdict)
            reasoning_trace = [
                {"step": "groq_error", "result": str(exc)},
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
