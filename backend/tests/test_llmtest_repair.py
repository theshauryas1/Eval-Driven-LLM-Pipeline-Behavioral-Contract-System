from pathlib import Path

from app.llmtest.loader import load_models, load_suites
from app.llmtest.repair import AutoRepairEngine


TESTS_DIR = Path(__file__).parents[2] / "tests"


def test_auto_repair_improves_mistral_mock(tmp_path):
    suites = [(str(path), suite) for path, suite in load_suites(str(TESTS_DIR))]
    models = load_models(str(TESTS_DIR), inline_suites=[suite for _, suite in suites])

    result = AutoRepairEngine(tmp_path).repair(
        suites=suites,
        models=models,
        baseline_model_id="mistral-mock",
        max_attempts=1,
    )

    baseline = result["baseline_report"]
    repaired = result["repaired_report"]

    baseline_pass_rate = baseline.model_summaries[0].pass_rate
    repaired_pass_rate = repaired.model_summaries[0].pass_rate

    assert repaired_pass_rate > baseline_pass_rate
    assert result["improved"] is True
    assert result["fixed_suite_paths"]
