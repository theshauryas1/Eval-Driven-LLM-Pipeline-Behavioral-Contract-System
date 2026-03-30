"""
Structural evaluator — deterministic rule-based checks.

Supported config keys:
  - min_citations (int): response must contain at least N citation markers
  - min_length (int): response must be at least N characters
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Citation patterns: [Source: ...], [Chunk ...], [Doc ...], [Ref ...]
CITATION_PATTERN = re.compile(
    r"\[(?:Source|Chunk|Doc|Ref)[^\]]*\]",
    re.IGNORECASE,
)


@dataclass
class EvalResult:
    contract_id: str
    passed: bool
    explanation: str


class StructuralEvaluator:
    """Evaluates structural contracts — no LLM calls, fully deterministic."""

    def evaluate(
        self,
        contract_id: str,
        config: dict,
        output: str,
        retrieved_context: str = "",
        **_kwargs,
    ) -> EvalResult:
        checks = []

        # --- min_citations ---
        if "min_citations" in config:
            min_n = int(config["min_citations"])
            citations = CITATION_PATTERN.findall(output)
            n_found = len(citations)
            passed = n_found >= min_n
            if passed:
                checks.append(
                    f"✓ Found {n_found} citation(s) (required ≥ {min_n}): {citations[:3]}"
                )
            else:
                checks.append(
                    f"✗ Found {n_found} citation(s) but required ≥ {min_n}. "
                    "Response does not reference any retrieved chunks."
                )

        # --- min_length ---
        if "min_length" in config:
            min_len = int(config["min_length"])
            actual_len = len(output.strip())
            passed_len = actual_len >= min_len
            if passed_len:
                checks.append(
                    f"✓ Response length {actual_len} chars (required ≥ {min_len})"
                )
            else:
                checks.append(
                    f"✗ Response length {actual_len} chars is below minimum {min_len}"
                )

        if not checks:
            return EvalResult(
                contract_id=contract_id,
                passed=True,
                explanation="No structural rules configured.",
            )

        all_passed = all(c.startswith("✓") for c in checks)
        return EvalResult(
            contract_id=contract_id,
            passed=all_passed,
            explanation=" | ".join(checks),
        )
