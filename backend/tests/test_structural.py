"""
Unit tests for the structural evaluator.
Run: pytest tests/test_structural.py -v
"""
import pytest
from app.evaluators.structural import StructuralEvaluator


@pytest.fixture
def evaluator():
    return StructuralEvaluator()


class TestMinCitations:
    def test_passes_with_one_citation(self, evaluator):
        result = evaluator.evaluate(
            contract_id="always_cite_source",
            config={"min_citations": 1},
            output="The capital of France is Paris. [Source: wiki_france]",
        )
        assert result.passed is True
        assert "1" in result.explanation

    def test_passes_with_multiple_citations(self, evaluator):
        result = evaluator.evaluate(
            contract_id="always_cite_source",
            config={"min_citations": 2},
            output="AI is transformative [Source: chunk_1]. It enables automation [Chunk 2].",
        )
        assert result.passed is True

    def test_fails_with_no_citations(self, evaluator):
        result = evaluator.evaluate(
            contract_id="always_cite_source",
            config={"min_citations": 1},
            output="The answer is 42.",
        )
        assert result.passed is False
        assert "0" in result.explanation

    def test_fails_when_too_few_citations(self, evaluator):
        result = evaluator.evaluate(
            contract_id="always_cite_source",
            config={"min_citations": 3},
            output="Here is one [Source: doc_1] answer.",
        )
        assert result.passed is False

    def test_recognizes_doc_and_ref_markers(self, evaluator):
        result = evaluator.evaluate(
            contract_id="always_cite_source",
            config={"min_citations": 1},
            output="Quantum entanglement [Doc: physics_101] is fascinating [Ref: einstein_1935].",
        )
        assert result.passed is True


class TestMinLength:
    def test_passes_long_response(self, evaluator):
        result = evaluator.evaluate(
            contract_id="response_not_empty",
            config={"min_length": 20},
            output="This is a sufficiently long response.",
        )
        assert result.passed is True

    def test_fails_short_response(self, evaluator):
        result = evaluator.evaluate(
            contract_id="response_not_empty",
            config={"min_length": 20},
            output="Short.",
        )
        assert result.passed is False

    def test_boundary_exact_length(self, evaluator):
        result = evaluator.evaluate(
            contract_id="response_not_empty",
            config={"min_length": 5},
            output="Hello",
        )
        assert result.passed is True

    def test_empty_string_fails(self, evaluator):
        result = evaluator.evaluate(
            contract_id="response_not_empty",
            config={"min_length": 1},
            output="",
        )
        assert result.passed is False


class TestCombinedConfig:
    def test_fails_when_either_rule_fails(self, evaluator):
        # Long enough but no citation
        result = evaluator.evaluate(
            contract_id="combined",
            config={"min_citations": 1, "min_length": 10},
            output="This is long enough but has no citation.",
        )
        assert result.passed is False

    def test_passes_when_all_rules_pass(self, evaluator):
        result = evaluator.evaluate(
            contract_id="combined",
            config={"min_citations": 1, "min_length": 10},
            output="Here is the answer [Source: chunk_1]. It is correct.",
        )
        assert result.passed is True

    def test_no_config_passes(self, evaluator):
        result = evaluator.evaluate(
            contract_id="no_rules",
            config={},
            output="Anything",
        )
        assert result.passed is True
