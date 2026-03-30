"""
Pattern evaluator — regex-based checks for PII and format requirements.

Supported config keys:
  - pattern (str): Python regex pattern to test against the output
  - must_not_match (bool): if True, fails when pattern IS found (PII detection)
  - must_match (bool): if True, fails when pattern is NOT found (format enforcement)
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class EvalResult:
    contract_id: str
    passed: bool
    explanation: str


class PatternEvaluator:
    """Evaluates pattern contracts using compiled regex."""

    def evaluate(
        self,
        contract_id: str,
        config: dict,
        output: str,
        **_kwargs,
    ) -> EvalResult:
        pattern_str = config.get("pattern")
        if not pattern_str:
            return EvalResult(
                contract_id=contract_id,
                passed=True,
                explanation="No pattern configured — skipped.",
            )

        try:
            regex = re.compile(pattern_str, re.IGNORECASE)
        except re.error as exc:
            return EvalResult(
                contract_id=contract_id,
                passed=False,
                explanation=f"Invalid regex pattern '{pattern_str}': {exc}",
            )

        matches = regex.findall(output)
        found = len(matches) > 0

        must_not_match = config.get("must_not_match", False)
        must_match = config.get("must_match", False)

        if must_not_match:
            if found:
                # Redact matches partially for privacy in logs
                redacted = [m[:4] + "***" if len(m) > 4 else "***" for m in matches[:3]]
                return EvalResult(
                    contract_id=contract_id,
                    passed=False,
                    explanation=(
                        f"✗ Forbidden pattern found {len(matches)} time(s). "
                        f"Matched (redacted): {redacted}"
                    ),
                )
            return EvalResult(
                contract_id=contract_id,
                passed=True,
                explanation="✓ No forbidden pattern matches found in response.",
            )

        if must_match:
            if not found:
                return EvalResult(
                    contract_id=contract_id,
                    passed=False,
                    explanation=f"✗ Required pattern '{pattern_str}' not found in response.",
                )
            return EvalResult(
                contract_id=contract_id,
                passed=True,
                explanation=f"✓ Required pattern found {len(matches)} time(s).",
            )

        # Default: just report whether pattern was found
        return EvalResult(
            contract_id=contract_id,
            passed=True,
            explanation=f"Pattern found: {found} ({len(matches)} match(es)).",
        )
