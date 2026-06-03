"""AutoHF — Model ranking engine (Phase 2 preparation).

Will rank models based on:
- Downloads & popularity
- Task compatibility
- Model size / inference speed
- Community ratings
"""

from __future__ import annotations

import math

from loguru import logger

from autohf.core.config import ModelInfo


def rank_models(
    models: list[ModelInfo],
    task_type: str,
    prefer_small: bool = False,
) -> list[ModelInfo]:
    """Score and sort model candidates.

    Placeholder for Phase 2 — currently sorts by downloads.
    """
    if not models:
        return []

    # Simple ranking by downloads for now
    ranked = sorted(models, key=lambda m: m.downloads, reverse=True)

    logger.info(
        "Ranked {} models. Top: {} ({:,} downloads)",
        len(ranked),
        ranked[0].model_id if ranked else "none",
        ranked[0].downloads if ranked else 0,
    )

    return ranked
