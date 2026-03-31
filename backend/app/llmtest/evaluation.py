from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.evaluators.semantic import SemanticEvaluator
from app.evaluators.structural import CITATION_PATTERN

from .plugins import load_callable
from .schemas import ConstraintSpec, ExpectedOutput, TestCase


EMAIL_PATTERN = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
PHONE_PATTERN = r"(\+?\d[\d\s\-().]{7,}\d)"


@dataclass
class FailureDetail:
    status: str
    failure_type: str
    reason: str
    confidence: float
    evaluator: str
    suggestion: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_type": self.failure_type,
            "reason": self.reason,
            "confidence": round(self.confidence, 2),
            "evaluator": self.evaluator,
            "suggestion": self.suggestion,
            "metadata": self.metadata,
        }


@dataclass
class CheckResult:
    name: str
    passed: bool
    evaluator: str
    failure: FailureDetail | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "evaluator": self.evaluator,
            "failure": self.failure.to_dict() if self.failure else None,
            "details": self.details,
        }


def _pass_result(name: str, evaluator: str, **details: Any) -> CheckResult:
    return CheckResult(name=name, passed=True, evaluator=evaluator, details=details)


def _fail_result(
    name: str,
    evaluator: str,
    failure_type: str,
    reason: str,
    confidence: float,
    suggestion: str | None = None,
    **details: Any,
) -> CheckResult:
    return CheckResult(
        name=name,
        passed=False,
        evaluator=evaluator,
        failure=FailureDetail(
            status="fail",
            failure_type=failure_type,
            reason=reason,
            confidence=confidence,
            evaluator=evaluator,
            suggestion=suggestion,
            metadata=details,
        ),
        details=details,
    )


def evaluate_expected(output: str, expected: ExpectedOutput | None) -> CheckResult | None:
    if expected is None:
        return None

    output_cmp = output if expected.case_sensitive else output.lower()
    value_text = str(expected.value)
    value_cmp = value_text if expected.case_sensitive else value_text.lower()

    if expected.type == "contains":
        if value_cmp in output_cmp:
            return _pass_result("expected", "expected.contains", match=value_text)
        return _fail_result(
            "expected",
            "expected.contains",
            "correctness",
            f"Expected output to contain '{value_text}'",
            0.98,
            suggestion=f"Explicitly include '{value_text}' in the answer.",
            expected=value_text,
        )

    if expected.type == "equals":
        if output_cmp.strip() == value_cmp.strip():
            return _pass_result("expected", "expected.equals", match=value_text)
        return _fail_result(
            "expected",
            "expected.equals",
            "correctness",
            f"Expected exact match with '{value_text}'",
            0.99,
            suggestion="Match the expected answer exactly.",
            expected=value_text,
            actual=output,
        )

    if expected.type == "regex":
        flags = 0 if expected.case_sensitive else re.IGNORECASE
        if re.search(value_text, output, flags):
            return _pass_result("expected", "expected.regex", pattern=value_text)
        return _fail_result(
            "expected",
            "expected.regex",
            "format",
            f"Expected output to match regex '{value_text}'",
            0.95,
            suggestion="Adjust the answer to satisfy the required output format.",
            pattern=value_text,
        )

    return _fail_result(
        "expected",
        "expected.unknown",
        "configuration",
        f"Unsupported expected type '{expected.type}'",
        1.0,
    )


def evaluate_constraint(
    output: str,
    context: str,
    test_case: TestCase,
    constraint: ConstraintSpec,
) -> CheckResult:
    if constraint.type == "max_length":
        limit = int(constraint.value)
        actual = len(output.strip())
        if actual <= limit:
            return _pass_result(constraint.type, "constraint.length", limit=limit, actual=actual)
        return _fail_result(
            constraint.type,
            "constraint.length",
            "length",
            f"Response length {actual} exceeds maximum {limit}",
            0.99,
            suggestion=f"Keep the final answer under {limit} characters.",
            limit=limit,
            actual=actual,
        )

    if constraint.type == "min_length":
        limit = int(constraint.value)
        actual = len(output.strip())
        if actual >= limit:
            return _pass_result(constraint.type, "constraint.length", limit=limit, actual=actual)
        return _fail_result(
            constraint.type,
            "constraint.length",
            "length",
            f"Response length {actual} is below minimum {limit}",
            0.99,
            suggestion=f"Provide at least {limit} characters of useful detail.",
            limit=limit,
            actual=actual,
        )

    if constraint.type == "contains_citation":
        required = int(constraint.value or 1)
        matches = CITATION_PATTERN.findall(output)
        if len(matches) >= required:
            return _pass_result(
                constraint.type,
                "constraint.citation",
                required=required,
                found=len(matches),
            )
        return _fail_result(
            constraint.type,
            "constraint.citation",
            "format",
            f"Found {len(matches)} citations but required at least {required}",
            0.96,
            suggestion="Cite the supporting source in the final answer.",
            required=required,
            found=len(matches),
        )

    if constraint.type == "no_pii_email":
        matches = re.findall(EMAIL_PATTERN, output, re.IGNORECASE)
        if not matches:
            return _pass_result(constraint.type, "constraint.pii", match_count=0)
        return _fail_result(
            constraint.type,
            "constraint.pii",
            "safety",
            "Email address detected in output",
            0.99,
            suggestion="Do not include email addresses in the answer.",
            match_count=len(matches),
        )

    if constraint.type == "no_pii_phone":
        matches = re.findall(PHONE_PATTERN, output, re.IGNORECASE)
        if not matches:
            return _pass_result(constraint.type, "constraint.pii", match_count=0)
        return _fail_result(
            constraint.type,
            "constraint.pii",
            "safety",
            "Phone number detected in output",
            0.99,
            suggestion="Do not include phone numbers in the answer.",
            match_count=len(matches),
        )

    if constraint.type == "regex":
        pattern = str(constraint.value or constraint.config.get("pattern", ""))
        must_match = constraint.config.get("must_match", True)
        found = bool(re.search(pattern, output, re.IGNORECASE))
        if must_match and found:
            return _pass_result(constraint.type, "constraint.regex", pattern=pattern)
        if not must_match and not found:
            return _pass_result(constraint.type, "constraint.regex", pattern=pattern)
        return _fail_result(
            constraint.type,
            "constraint.regex",
            "format" if must_match else "safety",
            (
                f"Required pattern '{pattern}' not found"
                if must_match
                else f"Forbidden pattern '{pattern}' found"
            ),
            0.95,
            suggestion="Adjust the answer to satisfy the regex contract.",
            pattern=pattern,
            must_match=must_match,
        )

    if constraint.type == "no_hallucination":
        evaluator = SemanticEvaluator()
        verdict = evaluator.evaluate(
            contract_id="no_hallucination",
            config={},
            output=output,
            retrieved_context=context,
        )
        if verdict.passed:
            return _pass_result(
                constraint.type,
                "constraint.hallucination",
                reasoning_trace=verdict.reasoning_trace,
            )
        return _fail_result(
            constraint.type,
            "constraint.hallucination",
            "hallucination",
            verdict.explanation or "Unsupported claim detected",
            0.78 if not verdict.reasoning_trace else 0.88,
            suggestion="Use only the provided context and acknowledge missing information.",
            reasoning_trace=verdict.reasoning_trace,
        )

    if constraint.type == "custom":
        target = constraint.config.get("callable")
        if not target:
            return _fail_result(
                constraint.type,
                "constraint.custom",
                "configuration",
                "Custom evaluator is missing config.callable",
                1.0,
            )
        evaluator = load_callable(target)
        result = evaluator(
            output=output,
            context=context,
            test_case=test_case.model_dump(),
            constraint=constraint.model_dump(),
        )
        if result is True:
            return _pass_result(constraint.type, "constraint.custom")
        if result is False:
            return _fail_result(
                constraint.type,
                "constraint.custom",
                "custom",
                "Custom evaluator returned False",
                0.75,
            )
        if isinstance(result, dict):
            if result.get("passed", False):
                return _pass_result(
                    constraint.type,
                    "constraint.custom",
                    **{k: v for k, v in result.items() if k != "passed"},
                )
            return _fail_result(
                constraint.type,
                "constraint.custom",
                result.get("failure_type", "custom"),
                result.get("reason", "Custom evaluator reported a failure"),
                float(result.get("confidence", 0.75)),
                result.get("suggestion"),
                **{
                    k: v
                    for k, v in result.items()
                    if k
                    not in {"passed", "failure_type", "reason", "confidence", "suggestion"}
                },
            )
        return _fail_result(
            constraint.type,
            "constraint.custom",
            "custom",
            "Custom evaluator returned an unsupported result shape",
            1.0,
        )

    return _fail_result(
        constraint.type,
        "constraint.unknown",
        "configuration",
        f"Unsupported constraint type '{constraint.type}'",
        1.0,
    )


def aggregate_failure_counts(checks: list[CheckResult]) -> dict[str, int]:
    counter = Counter(
        check.failure.failure_type
        for check in checks
        if check.failure is not None
    )
    return dict(counter)
