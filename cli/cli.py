import click
from pathlib import Path
import json
from .schema import parse_test_suite
from .execution import execute_tests
from .evaluators import evaluate_results
from .auto_repair import repair_loop

@click.group()
def cli():
    """llmtest: Local testing framework for LLM Behavioral Contracts."""
    pass

@cli.command()
@click.argument('tests_dir', type=click.Path(exists=True))
@click.option('--model', default='llama-3.3-70b-versatile', help='Model to test against')
def run(tests_dir, model):
    """Run test suites in the specified directory."""
    click.echo(f"🔍 Parsing test suites from {tests_dir}...")
    
    all_tests = []
    p = Path(tests_dir)
    files = [p] if p.is_file() else list(p.glob('*.json'))
    
    for f in files:
        all_tests.extend(parse_test_suite(f))
        
    click.echo(f"✅ Found {len(all_tests)} tests. Executing...")
    
    execution_results = execute_tests(all_tests, models=[model])
    eval_report = evaluate_results(execution_results)
    
    _print_report(eval_report)

@cli.command()
@click.argument('tests_dir', type=click.Path(exists=True))
@click.argument('model_a')
@click.argument('model_b')
def compare(tests_dir, model_a, model_b):
    """Compare performance between two models."""
    click.echo(f"⚖️ Comparing {model_a} vs {model_b} on {tests_dir}...")
    
    all_tests = []
    p = Path(tests_dir)
    files = [p] if p.is_file() else list(p.glob('*.json'))
    for f in files:
        all_tests.extend(parse_test_suite(f))
        
    execution_results = execute_tests(all_tests, models=[model_a, model_b])
    eval_report = evaluate_results(execution_results)
    
    _print_report(eval_report)

@cli.command()
@click.argument('tests_dir', type=click.Path(exists=True))
def fix(tests_dir):
    """Run the Auto-Repair loop on failing tests."""
    click.echo(f"🔧 Running Auto-Repair loop on failing tests in {tests_dir}...")
    repair_loop(Path(tests_dir))

def _print_report(report: dict):
    click.echo("\n📊 === FAILURE TAXONOMY DASHBOARD ===")
    print(json.dumps(report, indent=2))
    click.echo("=====================================\n")

if __name__ == '__main__':
    cli()
