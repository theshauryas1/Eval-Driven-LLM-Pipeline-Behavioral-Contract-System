from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schemas import PromptSpec


class PromptHistoryStore:
    def __init__(self, workspace_root: Path):
        self.base_dir = workspace_root / ".llmtest"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.base_dir / "prompt_versions.json"

    def _load(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        with self.history_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save(self, history: list[dict[str, Any]]) -> None:
        with self.history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    def register_prompt(
        self,
        prompt: PromptSpec,
        suite_name: str,
        source_path: str,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        history = self._load()
        prompt_hash = hashlib.sha256(prompt.template.encode("utf-8")).hexdigest()
        record = {
            "prompt_id": prompt.id,
            "version": prompt.version,
            "suite_name": suite_name,
            "source_path": source_path,
            "template_hash": prompt_hash,
            "template": prompt.template,
            "metrics": metrics or {},
        }
        exists = any(
            item["prompt_id"] == prompt.id
            and item["version"] == prompt.version
            and item["template_hash"] == prompt_hash
            and item.get("suite_name") == suite_name
            for item in history
        )
        if not exists:
            history.append(record)
            self._save(history)
            return record

        for item in history:
            if (
                item["prompt_id"] == prompt.id
                and item["version"] == prompt.version
                and item["template_hash"] == prompt_hash
                and item.get("suite_name") == suite_name
            ):
                item["metrics"] = {**item.get("metrics", {}), **(metrics or {})}
                self._save(history)
                return item

        return record
