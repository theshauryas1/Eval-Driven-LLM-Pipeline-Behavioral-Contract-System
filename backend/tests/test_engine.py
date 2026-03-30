"""
Integration test for the EvaluationEngine across all 5 example contracts.
"""
import pytest
from pathlib import Path
from app.evaluators.engine import EvaluationEngine

CONTRACTS_PATH = str(
    Path(__file__).parents[2] / "contracts" / "example_contracts.yaml"
)


@pytest.fixture
def engine():
    return EvaluationEngine(contracts_yaml_path=CONTRACTS_PATH)


class TestEngineIntegration:
    def test_loads_five_contracts(self, engine):
        assert len(engine.contracts) == 5

    def test_clean_trace_mostly_passes(self, engine):
        results = engine.run(
            output=(
                "Artificial intelligence enables computers to perform tasks "
                "that typically require human intelligence. [Source: ai_overview_chunk_1]"
            ),
            retrieved_context=(
                "AI, or Artificial Intelligence, refers to the simulation of human "
                "intelligence processes by machines."
            ),
        )
        assert len(results) == 5
        by_id = {r.contract_id: r for r in results}

        assert by_id["always_cite_source"].passed is True
        assert by_id["response_not_empty"].passed is True
        assert by_id["no_pii_email"].passed is True
        assert by_id["no_pii_phone"].passed is True

    def test_pii_email_leak_fires(self, engine):
        results = engine.run(
            output=(
                "Contact our support at help@acme.com. "
                "AI uses machine learning [Source: chunk_1]."
            ),
            retrieved_context="AI uses machine learning algorithms.",
        )
        by_id = {r.contract_id: r for r in results}
        assert by_id["no_pii_email"].passed is False

    def test_no_citation_fires(self, engine):
        results = engine.run(
            output="Machine learning is a subset of artificial intelligence.",
            retrieved_context="ML is a type of AI.",
        )
        by_id = {r.contract_id: r for r in results}
        assert by_id["always_cite_source"].passed is False

    def test_short_response_fires(self, engine):
        results = engine.run(
            output="Yes. [Source: x]",
            retrieved_context="Some context.",
        )
        by_id = {r.contract_id: r for r in results}
        assert by_id["response_not_empty"].passed is False

    def test_all_results_have_required_fields(self, engine):
        results = engine.run(
            output="Some response [Source: chunk_1].",
            retrieved_context="Context text.",
        )
        for r in results:
            assert hasattr(r, "contract_id")
            assert hasattr(r, "passed")
            assert hasattr(r, "explanation")
            assert hasattr(r, "contract_type")
            assert r.contract_type in ("structural", "pattern", "semantic")
