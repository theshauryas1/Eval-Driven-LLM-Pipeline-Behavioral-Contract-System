"""
Contract loader — parses YAML contract definitions and returns typed contract objects.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


class Contract(BaseModel):
    id: str
    type: str  # structural | pattern | semantic
    description: str
    config: dict[str, Any] = {}

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"structural", "pattern", "semantic"}
        if v not in allowed:
            raise ValueError(f"Contract type must be one of {allowed}, got '{v}'")
        return v


def load_contracts(yaml_path: str | Path | None = None) -> list[Contract]:
    """Load contracts from YAML file. Falls back to env var CONTRACTS_YAML_PATH."""
    if yaml_path is None:
        yaml_path = os.getenv("CONTRACTS_YAML_PATH", "contracts/example_contracts.yaml")

    path = Path(yaml_path)
    if not path.is_absolute():
        candidate_paths = [
            Path.cwd() / path,
            Path(__file__).parents[2] / path,
            Path(__file__).parents[3] / path,
        ]
        path = next((candidate for candidate in candidate_paths if candidate.exists()), candidate_paths[1])

    if not path.exists():
        raise FileNotFoundError(f"Contracts YAML not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    contracts_raw = raw.get("contracts", [])
    contracts = [Contract(**c) for c in contracts_raw]
    return contracts
