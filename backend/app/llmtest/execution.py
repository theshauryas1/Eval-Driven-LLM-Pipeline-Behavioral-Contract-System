from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import httpx

from .evaluation import (
    CheckResult,
    FailureDetail,
    aggregate_failure_counts,
    evaluate_constraint,
    evaluate_expected,
)
from .schemas import ModelSpec, PromptSpec, SuiteFile, TestCase


@dataclass
class TestRunResult:
    suite_name: str
    test_name: str
    model_id: str
    prompt_id: str
    prompt_version: str
    output: str
    status: str
    latency_ms: float
    checks: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "test_name": self.test_name,
            "model_id": self.model_id,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "output": self.output,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
            "checks": [check.to_dict() for check in self.checks],
            "failures": [check.failure.to_dict() for check in self.checks if check.failure],
        }


@dataclass
class ModelRunSummary:
    model_id: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    avg_latency_ms: float
    failure_breakdown: dict[str, int]

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return round(self.passed_tests / self.total_tests * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": self.pass_rate,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "failure_breakdown": self.failure_breakdown,
        }


@dataclass
class RunReport:
    run_id: str
    command: str
    created_at: str
    suite_paths: list[str]
    prompt_versions: list[dict[str, Any]]
    model_summaries: list[ModelRunSummary]
    results: list[TestRunResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "command": self.command,
            "created_at": self.created_at,
            "suite_paths": self.suite_paths,
            "prompt_versions": self.prompt_versions,
            "model_summaries": [summary.to_dict() for summary in self.model_summaries],
            "results": [result.to_dict() for result in self.results],
        }


class BaseModelAdapter:
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def generate(self, prompt_text: str, test_case: TestCase) -> str:
        raise NotImplementedError


class MockModelAdapter(BaseModelAdapter):
    def generate(self, prompt_text: str, test_case: TestCase) -> str:
        repair_responses = self.spec.settings.get("repair_responses", {})
        if "Repair instructions:" in prompt_text and test_case.test_name in repair_responses:
            return repair_responses[test_case.test_name]
        if test_case.test_name in self.spec.responses:
            return self.spec.responses[test_case.test_name]
        if self.spec.default_response:
            return self.spec.default_response.format(
                input=test_case.input,
                context=test_case.context,
                prompt=prompt_text,
            )
        return prompt_text


class EchoModelAdapter(BaseModelAdapter):
    def generate(self, prompt_text: str, test_case: TestCase) -> str:
        return prompt_text


class OpenAICompatibleAdapter(BaseModelAdapter):
    def generate(self, prompt_text: str, test_case: TestCase) -> str:
        base_url = self.spec.settings.get("base_url")
        api_key_env = self.spec.settings.get("api_key_env", "OPENAI_API_KEY")
        model_name = self.spec.model_name or self.spec.settings.get("model")
        if not base_url or not model_name:
            raise ValueError(
                f"Model '{self.spec.id}' requires settings.base_url and model_name"
            )

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable '{api_key_env}' is required for model '{self.spec.id}'"
            )

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.spec.settings.get("temperature", 0),
        }

        with httpx.Client(timeout=self.spec.settings.get("timeout", 30)) as client:
            response = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        return body["choices"][0]["message"]["content"]


def build_adapter(spec: ModelSpec) -> BaseModelAdapter:
    if spec.provider == "mock":
        return MockModelAdapter(spec)
    if spec.provider == "echo":
        return EchoModelAdapter(spec)
    if spec.provider == "openai_compatible":
        return OpenAICompatibleAdapter(spec)
    raise ValueError(f"Unsupported provider '{spec.provider}'")


def render_prompt(prompt: PromptSpec, test_case: TestCase, override: str | None = None) -> str:
    template = override or prompt.template
    return template.format(input=test_case.input, context=test_case.context)


class ExecutionEngine:
    def run(
        self,
        suites: list[tuple[str, SuiteFile]],
        models: list[ModelSpec],
        command: str,
    ) -> RunReport:
        run_id = str(uuid.uuid4())
        created_at = datetime.now(tz=timezone.utc).isoformat()
        results: list[TestRunResult] = []
        prompt_versions = [
            {
                "suite_name": suite.suite_name,
                "prompt_id": suite.prompt.id,
                "prompt_version": suite.prompt.version,
            }
            for _, suite in suites
        ]

        for model in models:
            adapter = build_adapter(model)
            for suite_path, suite in suites:
                for test_case in suite.tests:
                    prompt_text = render_prompt(
                        suite.prompt,
                        test_case,
                        override=model.prompt_override,
                    )
                    start = perf_counter()
                    try:
                        output = adapter.generate(prompt_text, test_case)
                    except Exception as exc:
                        latency_ms = (perf_counter() - start) * 1000
                        results.append(
                            TestRunResult(
                                suite_name=suite.suite_name,
                                test_name=test_case.test_name,
                                model_id=model.id,
                                prompt_id=suite.prompt.id,
                                prompt_version=suite.prompt.version,
                                output="",
                                status="fail",
                                latency_ms=latency_ms,
                                checks=[
                                    CheckResult(
                                        name="execution",
                                        passed=False,
                                        evaluator="execution.adapter",
                                        failure=FailureDetail(
                                            status="fail",
                                            failure_type="execution",
                                            reason=str(exc),
                                            confidence=1.0,
                                            evaluator="execution.adapter",
                                            suggestion="Check the model adapter configuration and credentials.",
                                        ),
                                    )
                                ],
                            )
                        )
                        continue
                    latency_ms = (perf_counter() - start) * 1000
                    checks: list[CheckResult] = []
                    expected_result = evaluate_expected(output, test_case.expected)
                    if expected_result is not None:
                        checks.append(expected_result)
                    for constraint in test_case.constraints:
                        checks.append(
                            evaluate_constraint(output, test_case.context, test_case, constraint)
                        )

                    status = "pass" if all(check.passed for check in checks) else "fail"
                    results.append(
                        TestRunResult(
                            suite_name=suite.suite_name,
                            test_name=test_case.test_name,
                            model_id=model.id,
                            prompt_id=suite.prompt.id,
                            prompt_version=suite.prompt.version,
                            output=output,
                            status=status,
                            latency_ms=latency_ms,
                            checks=checks,
                        )
                    )

        model_summaries = self._summarize_by_model(results, models)
        return RunReport(
            run_id=run_id,
            command=command,
            created_at=created_at,
            suite_paths=[suite_path for suite_path, _ in suites],
            prompt_versions=prompt_versions,
            model_summaries=model_summaries,
            results=results,
        )

    def _summarize_by_model(
        self,
        results: list[TestRunResult],
        models: list[ModelSpec],
    ) -> list[ModelRunSummary]:
        summaries: list[ModelRunSummary] = []
        for model in models:
            model_results = [result for result in results if result.model_id == model.id]
            checks = [check for result in model_results for check in result.checks]
            failure_breakdown = aggregate_failure_counts(checks)
            total = len(model_results)
            passed = sum(1 for result in model_results if result.status == "pass")
            failed = total - passed
            avg_latency = (
                sum(result.latency_ms for result in model_results) / total
                if total
                else 0.0
            )
            summaries.append(
                ModelRunSummary(
                    model_id=model.id,
                    total_tests=total,
                    passed_tests=passed,
                    failed_tests=failed,
                    avg_latency_ms=avg_latency,
                    failure_breakdown=failure_breakdown,
                )
            )
        return summaries
