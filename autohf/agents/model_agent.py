"""AutoHF — Model search agent (Phase 2 preparation).

Patterns extracted from:
- HF Hub: list_models with pipeline_tag, sort
- AutoGen: Agent pattern with callable interface
"""

from __future__ import annotations

from typing import Optional

from huggingface_hub import HfApi
from loguru import logger

from autohf.core.config import ModelInfo


class ModelAgent:
    """Agent responsible for model discovery on HF Hub.

    Prepared for Phase 2 where models will be automatically selected
    and fine-tuned based on the task and dataset.
    """

    def __init__(self) -> None:
        self.api = HfApi()
        self.search_history: list[list[ModelInfo]] = []

    def __call__(self, task_type: str, limit: int = 10) -> list[ModelInfo]:
        """Search for models matching a task type."""
        models = self.search(task_type, limit)
        self.search_history.append(models)
        return models

    def search(
        self,
        task_type: str,
        limit: int = 10,
        library: Optional[str] = None,
    ) -> list[ModelInfo]:
        """Search HF Hub for models matching a pipeline tag.

        Uses HF Hub's pipeline_tag for precise filtering.
        """
        logger.info("Searching models — task='{}', limit={}", task_type, limit)

        try:
            kwargs = {
                "pipeline_tag": task_type,
                "sort": "downloads",
                "limit": limit,
            }
            if library:
                kwargs["library"] = library

            results = self.api.list_models(**kwargs)

            models = []
            for m in results:
                models.append(
                    ModelInfo(
                        model_id=m.id,
                        pipeline_tag=m.pipeline_tag,
                        downloads=m.downloads or 0,
                        likes=m.likes or 0,
                        library_name=getattr(m, "library_name", None),
                    )
                )

            logger.info("Found {} models for task '{}'.", len(models), task_type)
            return models

        except Exception as e:
            logger.warning("Model search failed: {}", e)
            return []

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get detailed info about a specific model."""
        try:
            info = self.api.model_info(model_id)
            return ModelInfo(
                model_id=info.id,
                pipeline_tag=info.pipeline_tag,
                downloads=info.downloads or 0,
                likes=info.likes or 0,
                library_name=getattr(info, "library_name", None),
            )
        except Exception as e:
            logger.warning("Could not fetch model info for '{}': {}", model_id, e)
            return None

    def reset(self) -> None:
        self.search_history.clear()
