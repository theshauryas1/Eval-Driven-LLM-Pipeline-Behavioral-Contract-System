"""
Evaluation engine — orchestrates all evaluators against active contracts.

Usage:
    engine = EvaluationEngine(contracts_yaml_path="path/to/contracts.yaml")
    results = engine.run(
        output="...",
        retrieved_context="...",
        input_text="..."
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract_loader import Contract, load_contracts
from .structural import StructuralEvaluator
from .structural import EvalResult as StructuralResult
from .pattern import PatternEvaluator
from .pattern import EvalResult as PatternResult
from .semantic import SemanticEvaluator
from .semantic import EvalResult as SemanticResult


class EvaluationResult:
    """Unified result object across all evaluator types."""

    def __init__(
        self,
        contract_id: str,
        contract_type: str,
        passed: bool,
        explanation: str,
        reasoning_trace: list[dict] | None = None,
    ):
        self.contract_id = contract_id
        self.contract_type = contract_type
        self.passed = passed
        self.explanation = explanation
        self.reasoning_trace = reasoning_trace or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "contract_type": self.contract_type,
            "passed": self.passed,
            "explanation": self.explanation,
            "reasoning_trace": self.reasoning_trace,
        }

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"<EvaluationResult [{status}] {self.contract_id}: {self.explanation[:80]}>"


class EvaluationEngine:
    """
    Loads contracts from YAML and routes each to the correct evaluator.
    Stateless — can be called concurrently for multiple traces.
    """

    def __init__(self, contracts_yaml_path: str | Path | None = None):
        self.contracts: list[Contract] = load_contracts(contracts_yaml_path)
        self._structural = StructuralEvaluator()
        self._pattern = PatternEvaluator()
        self._semantic = SemanticEvaluator()

    def run(
        self,
        output: str,
        retrieved_context: str = "",
        input_text: str = "",
    ) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []

        for contract in self.contracts:
            result = self._evaluate_one(
                contract=contract,
                output=output,
                retrieved_context=retrieved_context,
                input_text=input_text,
            )
            results.append(result)

        return results

    def _evaluate_one(
        self,
        contract: Contract,
        output: str,
        retrieved_context: str,
        input_text: str,
    ) -> EvaluationResult:
        kwargs = dict(
            contract_id=contract.id,
            config=contract.config,
            output=output,
            retrieved_context=retrieved_context,
            input_text=input_text,
        )

        if contract.type == "structural":
            raw = self._structural.evaluate(**kwargs)
            return EvaluationResult(
                contract_id=raw.contract_id,
                contract_type="structural",
                passed=raw.passed,
                explanation=raw.explanation,
            )

        elif contract.type == "pattern":
            raw = self._pattern.evaluate(**kwargs)
            return EvaluationResult(
                contract_id=raw.contract_id,
                contract_type="pattern",
                passed=raw.passed,
                explanation=raw.explanation,
            )

        elif contract.type == "semantic":
            raw = self._semantic.evaluate(**kwargs)
            return EvaluationResult(
                contract_id=raw.contract_id,
                contract_type="semantic",
                passed=raw.passed,
                explanation=raw.explanation,
                reasoning_trace=raw.reasoning_trace,
            )

        else:
            return EvaluationResult(
                contract_id=contract.id,
                contract_type=contract.type,
                passed=False,
                explanation=f"Unknown contract type '{contract.type}'",
            )
