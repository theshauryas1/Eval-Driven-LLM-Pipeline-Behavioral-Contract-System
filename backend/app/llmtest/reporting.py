from __future__ import annotations

import json
from pathlib import Path

from .execution import RunReport


class ReportStore:
    def __init__(self, workspace_root: Path):
        self.base_dir = workspace_root / ".llmtest" / "reports"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, report: RunReport) -> Path:
        path = self.base_dir / f"{report.run_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        latest_path = self.base_dir / "latest.json"
        with latest_path.open("w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        return path

    def load(self, report_path: str | None = None) -> dict:
        path = Path(report_path) if report_path else self.base_dir / "latest.json"
        if not path.exists():
            raise FileNotFoundError(f"Report not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def format_model_summary(report: dict) -> str:
    lines = [f"Run {report['run_id']} at {report['created_at']}"]
    for summary in report.get("model_summaries", []):
        breakdown = ", ".join(
            f"{failure_type}={count}"
            for failure_type, count in sorted(summary.get("failure_breakdown", {}).items())
        ) or "none"
        lines.append(
            f"{summary['model_id']}: {summary['pass_rate']}% pass "
            f"({summary['passed_tests']}/{summary['total_tests']}), "
            f"avg latency {summary['avg_latency_ms']} ms, failures: {breakdown}"
        )

    failed = [result for result in report.get("results", []) if result["status"] == "fail"]
    if failed:
        lines.append("Worst-performing tests:")
        ranked = sorted(
            failed,
            key=lambda item: len(item.get("failures", [])),
            reverse=True,
        )[:5]
        for item in ranked:
            failure_types = ", ".join(
                failure["failure_type"] for failure in item.get("failures", [])
            )
            lines.append(
                f"- {item['model_id']}::{item['suite_name']}::{item['test_name']} -> {failure_types}"
            )
    return "\n".join(lines)
