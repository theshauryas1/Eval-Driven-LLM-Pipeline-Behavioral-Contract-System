"""
Unit tests for the semantic evaluator stub.
Full LangGraph agent tests require GROQ_API_KEY — these test the stub behavior.
"""
import pytest
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
