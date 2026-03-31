from __future__ import annotations

import json
from pathlib import Path

from .schemas import ModelCatalog, ModelSpec, SuiteFile


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_suite_file(path: Path) -> SuiteFile:
    return SuiteFile.model_validate(_load_json(path))


def load_suites(path_str: str) -> list[tuple[Path, SuiteFile]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Suite path does not exist: {path}")

    if path.is_file():
        return [(path, load_suite_file(path))]

    suite_paths = sorted(
        candidate
        for candidate in path.glob("*.json")
        if candidate.name.lower() != "models.json"
    )
    if not suite_paths:
        raise FileNotFoundError(f"No suite JSON files found in: {path}")

    return [(suite_path, load_suite_file(suite_path)) for suite_path in suite_paths]


def load_models(
    suite_path: str,
    explicit_models_path: str | None = None,
    inline_suites: list[SuiteFile] | None = None,
) -> list[ModelSpec]:
    if explicit_models_path:
        catalog = ModelCatalog.model_validate(_load_json(Path(explicit_models_path)))
        return catalog.models

    base_path = Path(suite_path)
    if base_path.is_dir():
        models_file = base_path / "models.json"
        if models_file.exists():
            catalog = ModelCatalog.model_validate(_load_json(models_file))
            return catalog.models

    models: list[ModelSpec] = []
    for suite in inline_suites or []:
        models.extend(suite.models)

    if not models:
        raise FileNotFoundError(
            "No model catalog found. Add tests/models.json or pass --models."
        )

    return models
