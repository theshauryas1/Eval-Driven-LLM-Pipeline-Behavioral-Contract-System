from __future__ import annotations

import argparse
from pathlib import Path

from app.llmtest.execution import ExecutionEngine
from app.llmtest.loader import load_models, load_suites
from app.llmtest.prompt_history import PromptHistoryStore
from app.llmtest.reporting import ReportStore, format_model_summary
from app.llmtest.repair import AutoRepairEngine


def _load_runtime(args) -> tuple[list[tuple[str, object]], list[object], Path]:
    suite_entries = [(str(path), suite) for path, suite in load_suites(args.path)]
    models = load_models(args.path, args.models, [suite for _, suite in suite_entries])
    workspace_root = Path(args.workspace_root or Path.cwd())
    return suite_entries, models, workspace_root


def run_command(args) -> int:
    suites, models, workspace_root = _load_runtime(args)
    report = ExecutionEngine().run(suites=suites, models=models, command="run")
    store = ReportStore(workspace_root)
    prompt_store = PromptHistoryStore(workspace_root)

    for suite_path, suite in suites:
        for summary in report.model_summaries:
            prompt_store.register_prompt(
                prompt=suite.prompt,
                suite_name=suite.suite_name,
                source_path=suite_path,
                metrics={summary.model_id: summary.pass_rate},
            )

    report_path = store.save(report)
    print(format_model_summary(report.to_dict()))
    print(f"Report saved to {report_path}")
    return 1 if args.fail_on_error and any(result.status == "fail" for result in report.results) else 0


def compare_command(args) -> int:
    suites, models, workspace_root = _load_runtime(args)
    report = ExecutionEngine().run(suites=suites, models=models, command="compare")
    report_path = ReportStore(workspace_root).save(report)
    print(format_model_summary(report.to_dict()))
    print(f"Comparison saved to {report_path}")
    return 0


def report_command(args) -> int:
    report = ReportStore(Path(args.workspace_root or Path.cwd())).load(args.report_path)
    print(format_model_summary(report))
    return 0


def fix_command(args) -> int:
    suites, models, workspace_root = _load_runtime(args)
    result = AutoRepairEngine(workspace_root).repair(
        suites=suites,
        models=models,
        baseline_model_id=args.model,
        max_attempts=args.max_attempts,
    )

    baseline = result["baseline_report"]
    repaired = result["repaired_report"]
    report_store = ReportStore(workspace_root)
    baseline_path = report_store.save(baseline)
    repaired_path = report_store.save(repaired)

    print("Baseline:")
    print(format_model_summary(baseline.to_dict()))
    print()
    print("After repair:")
    print(format_model_summary(repaired.to_dict()))
    print(f"Baseline report saved to {baseline_path}")
    print(f"Repaired report saved to {repaired_path}")
    if result["fixed_suite_paths"]:
        print("Fixed suite files:")
        for path in result["fixed_suite_paths"]:
            print(f"- {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmtest", description="Contract-based LLM test runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler, help_text in (
        ("run", run_command, "Run test suites"),
        ("compare", compare_command, "Compare models across test suites"),
        ("fix", fix_command, "Auto-repair prompts and retry"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("path", help="Path to a suite JSON file or suite directory")
        command.add_argument("--models", help="Path to models.json", default=None)
        command.add_argument("--workspace-root", default=None, help="Directory for .llmtest artifacts")
        if name == "run":
            command.add_argument("--fail-on-error", action="store_true", help="Exit non-zero when any test fails")
        if name == "fix":
            command.add_argument("--model", default=None, help="Baseline model id to optimize against")
            command.add_argument("--max-attempts", type=int, default=1, help="Number of repair iterations")
        command.set_defaults(func=handler)

    report_parser = subparsers.add_parser("report", help="Show the latest stored report")
    report_parser.add_argument("--report-path", default=None, help="Optional explicit report JSON path")
    report_parser.add_argument("--workspace-root", default=None, help="Directory containing .llmtest artifacts")
    report_parser.set_defaults(func=report_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
