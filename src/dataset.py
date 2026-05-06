from __future__ import annotations

import json
from pathlib import Path

from src.schemas import StreamTask


def load_stream_tasks(path: str, limit: int | None = None) -> list[StreamTask]:
    tasks: list[StreamTask] = []
    data_path = Path(path)

    with data_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                tasks.append(StreamTask(**obj))
            except Exception as exc:
                raise ValueError(f"Failed to parse {data_path} line {line_number}: {exc}") from exc
            if limit is not None and len(tasks) >= limit:
                break

    return tasks


def load_gaia_level1_tasks(path: str, limit: int | None = None) -> list[StreamTask]:
    """TODO: adapt GAIA Level 1 validation examples into the experiment pipeline.

    GAIA tasks include tool use, browsing, and short unambiguous answers. This stub
    intentionally returns StreamTask-compatible objects later so the baseline,
    swarm, evaluator, and result writer do not need to be rewritten.
    """
    raise NotImplementedError(
        "GAIA Level 1 loading is a planned next step. "
        "Add an adapter that maps GAIA records to the benchmark task contract."
    )
