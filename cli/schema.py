import json
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import List, Any, Optional

class ConstraintDef(BaseModel):
    type: str
    value: Optional[Any] = None

class ExpectedDef(BaseModel):
    type: str # 'contains', 'not_contains', 'exact'
    value: str

class TestCase(BaseModel):
    test_name: str
    input: str
    expected: ExpectedDef
    constraints: List[ConstraintDef] = []

def parse_test_suite(file_path: Path) -> List[TestCase]:
    """Parse a JSON test suite into Pydantic models."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cases = []
    for item in data:
        try:
            cases.append(TestCase(**item))
        except ValidationError as e:
            print(f"Failed to parse test case: {e}")
    return cases
