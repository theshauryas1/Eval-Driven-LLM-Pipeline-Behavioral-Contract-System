"""pytest configuration — adds backend/ to sys.path so imports resolve."""
import sys
from pathlib import Path

# Ensure 'app' package is importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))
