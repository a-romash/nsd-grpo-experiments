"""Torch-free helpers for files written during a run."""
from __future__ import annotations

import json
from pathlib import Path


def read_step_times(run_dir: str | Path) -> dict[int, dict]:
    """step -> record; after a resume the LAST record of a repeated step wins."""
    path = Path(run_dir) / "step_times.jsonl"
    records: dict[int, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                records[rec["step"]] = rec
    return records
