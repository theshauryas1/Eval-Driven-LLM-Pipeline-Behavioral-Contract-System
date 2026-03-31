from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .execution import ExecutionEngine, RunReport
from .schemas import ModelSpec, PromptSpec, SuiteFile


def _next_version(version: str) -> str:
    if version.startswith("v") and version[1:].isdigit():
        return f"v{int(version[1:]) + 1}"
    return f"{version}_next"


class AutoRepairEngine:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.execution_engine = ExecutionEngine()

    def repair(
        self,
        suites: list[tuple[str, SuiteFile]],
        models: list[ModelSpec],
        baseline_model_id: str | None = None,
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        baseline_models = (
            [model for model in models if model.id == baseline_model_id]
            if baseline_model_id
            else models[:1]
        )
        if not baseline_models:
            raise ValueError("Baseline model was not found in the loaded model catalog")

        baseline_report = self.execution_engine.run(
            suites=suites,
            models=baseline_models,
            command="fix:baseline",
        )

        best_report = baseline_report
        best_suites = suites
        for _ in range(max_attempts):
            candidate_suites = self._apply_repairs(best_suites, best_report)
            candidate_report = self.execution_engine.run(
                suites=candidate_suites,
                models=baseline_models,
                command="fix:retry",
            )
            if self._score(candidate_report) > self._score(best_report):
                best_report = candidate_report
                best_suites = candidate_suites
            else:
                break

        fixed_paths = self._write_fixed_suites(best_suites, baseline_report, best_report)
        return {
            "baseline_report": baseline_report,
            "repaired_report": best_report,
            "improved": self._score(best_report) > self._score(baseline_report),
            "fixed_suite_paths": fixed_paths,
        }

    def _score(self, report: RunReport) -> float:
        return sum(summary.pass_rate for summary in report.model_summaries)

    def _apply_repairs(
        self,
        suites: list[tuple[str, SuiteFile]],
        report: RunReport,
    ) -> list[tuple[str, SuiteFile]]:
        failures_by_suite: dict[str, list[dict[str, Any]]] = {}
        for result in report.results:
            if result.status == "fail":
                failures_by_suite.setdefault(result.suite_name, []).extend(
                    check.failure.to_dict()
                    for check in result.checks
                    if check.failure is not None
                )

        repaired: list[tuple[str, SuiteFile]] = []
        for suite_path, suite in suites:
            suite_copy = deepcopy(suite)
            suite_failures = failures_by_suite.get(suite.suite_name, [])
            if suite_failures:
                suite_copy.prompt = self._repair_prompt(suite_copy.prompt, suite_failures)
            repaired.append((suite_path, suite_copy))
        return repaired

    def _repair_prompt(
        self,
        prompt: PromptSpec,
        failures: list[dict[str, Any]],
    ) -> PromptSpec:
        instructions: list[str] = []
        seen: set[str] = set()
        for failure in failures:
            failure_type = failure["failure_type"]
            if failure_type == "correctness":
                expected = failure.get("metadata", {}).get("expected")
                line = (
                    f"Explicitly include '{expected}' in the final answer."
                    if expected
                    else "Answer the question directly and explicitly."
                )
            elif failure_type == "hallucination":
                line = "Use only the provided context. If the answer is unsupported, say you do not know."
            elif failure_type == "length":
                limit = failure.get("metadata", {}).get("limit")
                line = (
                    f"Respect the length contract exactly (limit: {limit})."
                    if limit
                    else "Respect the required answer length."
                )
            elif failure_type == "safety":
                line = "Do not include emails, phone numbers, or other sensitive details."
            else:
                line = failure.get("suggestion") or "Follow the declared behavioral contracts exactly."

            if line not in seen:
                instructions.append(line)
                seen.add(line)

        extra_block = "\n\nRepair instructions:\n- " + "\n- ".join(instructions)
        return PromptSpec(
            id=prompt.id,
            version=_next_version(prompt.version),
            template=prompt.template + extra_block,
            metadata=prompt.metadata,
        )

    def _write_fixed_suites(
        self,
        suites: list[tuple[str, SuiteFile]],
        baseline_report: RunReport,
        repaired_report: RunReport,
    ) -> list[str]:
        if self._score(repaired_report) <= self._score(baseline_report):
            return []

        fixes_dir = self.workspace_root / ".llmtest" / "fixes"
        fixes_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for _, suite in suites:
            fixed_path = fixes_dir / f"{suite.suite_name}.{suite.prompt.version}.json"
            with fixed_path.open("w", encoding="utf-8") as handle:
                json.dump(suite.model_dump(), handle, indent=2)
            paths.append(str(fixed_path))
        return paths
