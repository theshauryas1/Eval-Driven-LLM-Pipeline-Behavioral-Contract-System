from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Callable


def load_callable(target: str) -> Callable:
    if ":" not in target:
        raise ValueError(
            "Custom evaluator must be '<module>:<callable>' or '<path.py>:<callable>'"
        )

    module_ref, callable_name = target.split(":", 1)

    if module_ref.endswith(".py") or Path(module_ref).exists():
        module_path = Path(module_ref).resolve()
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load plugin module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)

    try:
        return getattr(module, callable_name)
    except AttributeError as exc:
        raise ImportError(
            f"Custom evaluator '{callable_name}' not found in '{module_ref}'"
        ) from exc
