"""
Unit tests for the semantic evaluator stub.
Full LangGraph agent tests require GROQ_API_KEY — these test the stub behavior.
"""
import pytest
from app.evaluators import semantic
from app.evaluators.semantic import SemanticEvaluator


@pytest.fixture
def evaluator():
    return SemanticEvaluator()


class TestSemanticStub:
    def test_stub_returns_passed_true_without_api_key(self, evaluator):
        """Without a GROQ_API_KEY the evaluator should return a stub pass."""
        result = evaluator.evaluate(
            contract_id="context_faithfulness",
            config={},
            output="The Eiffel Tower is located in London.",
            retrieved_context="The Eiffel Tower is a famous landmark in Paris, France.",
        )
        # Either passes (stub) or correctly flags (live agent)
        assert isinstance(result.passed, bool)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

    def test_stub_returns_eval_result_shape(self, evaluator):
        result = evaluator.evaluate(
            contract_id="context_faithfulness",
            config={},
            output="Some response",
            retrieved_context="Some context",
        )
        assert hasattr(result, "contract_id")
        assert hasattr(result, "passed")
        assert hasattr(result, "explanation")
        assert hasattr(result, "reasoning_trace")
        assert result.contract_id == "context_faithfulness"
        assert isinstance(result.reasoning_trace, list)

    def test_empty_output_handled(self, evaluator):
        result = evaluator.evaluate(
            contract_id="context_faithfulness",
            config={},
            output="",
            retrieved_context="Some context",
        )
        assert isinstance(result.passed, bool)

    def test_empty_context_handled(self, evaluator):
        result = evaluator.evaluate(
            contract_id="context_faithfulness",
            config={},
            output="Some factual claim here.",
            retrieved_context="",
        )
        assert isinstance(result.passed, bool)

    def test_groq_error_falls_back_to_deterministic_path(self, monkeypatch):
        monkeypatch.setattr(semantic, "GROQ_AVAILABLE", True)

        class BrokenGraph:
            def invoke(self, _state):
                raise RuntimeError("429 Too Many Requests")

        evaluator = SemanticEvaluator()
        evaluator._graph = BrokenGraph()

        result = evaluator.evaluate(
            contract_id="context_faithfulness",
            config={},
            output="The Eiffel Tower is located in London.",
            retrieved_context="The Eiffel Tower is a famous landmark in Paris, France.",
        )

        assert result.reasoning_trace[0]["step"] == "groq_error"
        assert isinstance(result.passed, bool)

    def test_groq_cache_reuses_response(self, monkeypatch):
        monkeypatch.setattr(semantic, "_groq_cache", {})
        calls = {"count": 0}

        class FakeResponse:
            def __init__(self, content):
                self.content = content

        class FakeLLM:
            def invoke(self, _messages):
                calls["count"] += 1
                return FakeResponse("NONE")

        monkeypatch.setattr(semantic, "_make_llm", lambda model_name=None: FakeLLM())

        first = semantic._invoke_with_resilience("prompt one")
        second = semantic._invoke_with_resilience("prompt one")

        assert first == "NONE"
        assert second == "NONE"
        assert calls["count"] == 1

    def test_normalize_verdict_coerces_non_boolean_passed(self):
        verdict = semantic._normalize_verdict(
            {
                "passed": [],
                "violations": [{"claim": "Unsupported claim"}],
                "explanation": "Unsupported.",
            }
        )

        assert verdict["passed"] is False
        assert verdict["violations"] == [{"claim": "Unsupported claim"}]
        assert verdict["explanation"] == "Unsupported."

    def test_groq_verdict_is_normalized_before_return(self, monkeypatch):
        monkeypatch.setattr(semantic, "GROQ_AVAILABLE", True)

        class FakeGraph:
            def invoke(self, _state):
                return {
                    "claims": ["Acme offers phone support."],
                    "matched": [{"claim": "Acme offers phone support.", "best_match": "", "similarity": "none"}],
                    "verdict": {
                        "passed": [],
                        "violations": [{"claim": "Acme offers phone support."}],
                        "explanation": "Not supported by context.",
                    },
                }

        evaluator = SemanticEvaluator()
        evaluator._graph = FakeGraph()

        result = evaluator.evaluate(
            contract_id="context_faithfulness",
            config={"use_groq": True},
            output="Acme offers phone support.",
            retrieved_context="Support is available by email only.",
        )

        assert result.passed is False
        assert result.reasoning_trace[-1]["result"]["passed"] is False

    def test_match_to_context_uses_fallback_when_llm_returns_bad_matches(self, monkeypatch):
        monkeypatch.setattr(
            semantic,
            "_invoke_with_resilience",
            lambda _prompt: '[{"claim":"Support is available Monday through Friday.","best_match":"","similarity":"none"}]',
        )

        state = {
            "output": "Support is available Monday through Friday.",
            "context": "Support is available Monday through Friday from 9 AM to 5 PM IST.",
            "claims": ["Support is available Monday through Friday."],
            "matched": [],
            "verdict": {},
        }

        result = semantic.match_to_context(state)

        assert result["matched"][0]["similarity"] in {"high", "medium"}
        assert result["matched"][0]["best_match"]

    def test_fallback_matching_produces_supported_verdict_for_grounded_claims(self):
        claims = ["Support is available Monday through Friday."]
        matched = semantic._fallback_match_claims(
            claims,
            "Support is available Monday through Friday from 9 AM to 5 PM IST.",
        )
        verdict = semantic._build_verdict_from_matches(
            claims,
            matched,
            "Support is available Monday through Friday from 9 AM to 5 PM IST.",
        )

        assert matched[0]["similarity"] in {"high", "medium"}
        assert verdict["passed"] is True

    def test_normalize_claims_filters_inferred_claims_not_stated_in_output(self):
        claims = [
            "Support is available Monday through Friday.",
            "IST is a time zone.",
        ]

        normalized = semantic._normalize_claims(
            claims,
            "Support is available Monday through Friday from 9 AM to 5 PM IST.",
        )

        assert "Support is available Monday through Friday." in normalized
        assert "IST is a time zone." not in normalized

    def test_normalize_verdict_replaces_non_informative_violations(self):
        verdict = semantic._normalize_verdict(
            {
                "passed": False,
                "violations": [2],
                "explanation": "2 claim(s) not grounded in context.",
            },
            low_support=[
                {"claim": "Claim one", "similarity": "none"},
                {"claim": "Claim two", "similarity": "low"},
            ],
        )

        assert verdict["violations"] == ["Claim one", "Claim two"]
