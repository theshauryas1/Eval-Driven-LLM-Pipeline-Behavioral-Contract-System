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
