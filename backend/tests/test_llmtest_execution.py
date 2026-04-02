from pathlib import Path

import pytest

from app.evaluators import semantic
from app.llmtest.execution import ExecutionEngine
from app.llmtest.loader import load_models, load_suites


TESTS_DIR = Path(__file__).parents[2] / "tests"


@pytest.fixture(autouse=True)
def disable_live_groq_for_execution_tests(monkeypatch):
    monkeypatch.setattr(semantic, "GROQ_AVAILABLE", False)


def test_execution_engine_runs_all_suites_and_models():
    suites = [(str(path), suite) for path, suite in load_suites(str(TESTS_DIR))]
    models = load_models(str(TESTS_DIR), inline_suites=[suite for _, suite in suites])

    report = ExecutionEngine().run(suites=suites, models=models, command="run")

    assert len(report.model_summaries) == 3
    assert len(report.results) == 18

    summary_by_model = {summary.model_id: summary for summary in report.model_summaries}
    assert summary_by_model["gpt-4o-mock"].pass_rate == 100.0
    assert summary_by_model["mistral-mock"].failed_tests > 0
    assert "hallucination" in summary_by_model["mistral-mock"].failure_breakdown


def test_failure_objects_are_structured():
    suites = [(str(path), suite) for path, suite in load_suites(str(TESTS_DIR))]
    models = [model for model in load_models(str(TESTS_DIR), inline_suites=[suite for _, suite in suites]) if model.id == "mistral-mock"]

    report = ExecutionEngine().run(suites=suites, models=models, command="compare")
    failures = [
        failure
        for result in report.results
        for failure in result.to_dict()["failures"]
    ]

    assert failures
    assert all("failure_type" in failure for failure in failures)
    assert any(failure["failure_type"] == "hallucination" for failure in failures)
