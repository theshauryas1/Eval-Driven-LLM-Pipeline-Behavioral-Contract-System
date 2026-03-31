from pathlib import Path

from app.cli import build_parser


TESTS_DIR = Path(__file__).parents[2] / "tests"


def test_cli_parser_accepts_run_command():
    parser = build_parser()
    args = parser.parse_args(["run", str(TESTS_DIR)])

    assert args.command == "run"
    assert args.path == str(TESTS_DIR)


def test_cli_parser_accepts_fix_command():
    parser = build_parser()
    args = parser.parse_args(["fix", str(TESTS_DIR), "--model", "mistral-mock"])

    assert args.command == "fix"
    assert args.model == "mistral-mock"
