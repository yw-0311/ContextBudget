from __future__ import annotations
from pathlib import Path
import os
import pickle
from typing import Any

_DEBUG_DIR = Path(".debug_cache")
_DEBUG_DIR.mkdir(exist_ok=True)

def _dump_args(name: str, payload: Any) -> Path:
    path = _DEBUG_DIR / f"{name}.pkl"
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def _load_args(name: str) -> Any:
    path = _DEBUG_DIR / f"{name}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)
