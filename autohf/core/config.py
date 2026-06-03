"""AutoHF — Configuration models.

Inspired by AutoGluon's presets system and LangGraph's state management.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Presets — inspired by AutoGluon tabular_presets_dict
# ---------------------------------------------------------------------------

class Preset(str, Enum):
    """Training presets mirroring AutoGluon's preset system."""
    QUICK_PROTOTYPE = "quick_prototype"      # Fast prototyping (~60s)
    MEDIUM_QUALITY = "medium_quality"        # Default, good balance
    HIGH_QUALITY = "high_quality"            # Longer training, better results
    BEST_QUALITY = "best_quality"            # Maximum quality, long training
    OPTIMIZE_FOR_DEPLOYMENT = "optimize_for_deployment"  # Small model, fast inference


PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "quick_prototype": {
        "time_limit": 60,
        "presets": "medium_quality",
        "max_dataset_rows": 10_000,
        "max_datasets_to_search": 10,
        "hyperparameters": {"GBM": {}, "RF": {}},
    },
    "medium_quality": {
        "time_limit": 300,
        "presets": "medium_quality",
        "max_dataset_rows": 50_000,
        "max_datasets_to_search": 20,
    },
    "high_quality": {
        "time_limit": 600,
        "presets": "high_quality",
        "max_dataset_rows": 100_000,
        "max_datasets_to_search": 30,
    },
    "best_quality": {
        "time_limit": 3600,
        "presets": "best_quality",
        "max_dataset_rows": 200_000,
        "max_datasets_to_search": 50,
    },
    "optimize_for_deployment": {
        "time_limit": 300,
        "presets": "good_quality",
        "max_dataset_rows": 50_000,
        "keep_only_best": True,
        "save_space": True,
    },
}


# ---------------------------------------------------------------------------
# Problem types — inspired by AutoGluon core/constants
# ---------------------------------------------------------------------------

class ProblemType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


# Metric mapping — inspired by AutoGluon core/metrics
TASK_TO_PROBLEM_TYPE: dict[str, str] = {
    "text-classification": "auto",   # auto-detect binary vs multiclass
    "token-classification": "multiclass",
    "question-answering": "multiclass",
    "summarization": "regression",   # ROUGE-like metric
    "translation": "regression",
    "text-generation": "regression",
    "fill-mask": "multiclass",
    "text2text-generation": "regression",
    "zero-shot-classification": "multiclass",
}

# Recommended eval metrics per task — from AutoGluon's defaults
TASK_EVAL_METRICS: dict[str, str] = {
    "text-classification": "accuracy",
    "token-classification": "accuracy",
    "question-answering": "accuracy",
    "summarization": "accuracy",
    "translation": "accuracy",
    "text-generation": "accuracy",
    "fill-mask": "accuracy",
    "text2text-generation": "accuracy",
    "zero-shot-classification": "accuracy",
}

# AutoGluon hyperparameter presets by task
TASK_HYPERPARAMETERS: dict[str, dict] = {
    "text-classification": {
        "GBM": [
            {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
            {},
            {
                "learning_rate": 0.03,
                "num_leaves": 128,
                "feature_fraction": 0.9,
                "min_data_in_leaf": 3,
                "ag_args": {"name_suffix": "Large", "priority": 0},
            },
        ],
        "CAT": {},
        "XGB": {},
        "RF": [
            {"criterion": "gini", "ag_args": {"name_suffix": "Gini"}},
            {"criterion": "entropy", "ag_args": {"name_suffix": "Entr"}},
        ],
        "XT": [
            {"criterion": "gini", "ag_args": {"name_suffix": "Gini"}},
            {"criterion": "entropy", "ag_args": {"name_suffix": "Entr"}},
        ],
        "KNN": [
            {"weights": "uniform", "ag_args": {"name_suffix": "Unif"}},
            {"weights": "distance", "ag_args": {"name_suffix": "Dist"}},
        ],
    },
}


# ---------------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------------

class AutoHFConfig(BaseModel):
    """Central configuration for AutoHF pipeline.

    Inspired by AutoGluon's TabularPredictor parameters and preset system.
    """

    # --- Preset ---
    preset: str = Field(
        default="medium_quality",
        description="AutoHF preset. Overrides individual settings.",
    )

    # --- Dataset discovery ---
    max_datasets_to_search: int = Field(
        default=20,
        description="Maximum number of datasets to fetch from HF Hub for ranking.",
    )
    max_dataset_rows: int = Field(
        default=50_000,
        description="Cap the number of rows loaded from the dataset (for speed).",
    )

    # --- Training (AutoGluon) ---
    time_limit: int = Field(
        default=300,
        description="AutoGluon training time limit in seconds.",
    )
    presets: str = Field(
        default="medium_quality",
        description="AutoGluon presets: 'best_quality', 'high_quality', 'good_quality', 'medium_quality'.",
    )
    eval_metric: Optional[str] = Field(
        default=None,
        description="Evaluation metric. Auto-detected from task if None.",
    )
    hyperparameters: Optional[dict] = Field(
        default=None,
        description="AutoGluon hyperparameters dict. None = use defaults.",
    )

    # --- Output ---
    output_dir: str = Field(
        default="./autohf_output",
        description="Directory to save trained models.",
    )

    # --- Deployment options ---
    keep_only_best: bool = Field(
        default=False,
        description="Delete all models except the best one after training.",
    )
    save_space: bool = Field(
        default=False,
        description="Remove training artifacts to save disk space.",
    )

    # --- Pipeline control ---
    router: str = Field(
        default="auto",
        description="Routing strategy for task detection: 'auto' (LLM if key set), 'keyword' (local keywords), 'openai' (force OpenAI), or 'gemma' (local Gemma model).",
    )
    auto_select_dataset: bool = Field(
        default=True,
        description="Automatically select the best dataset (vs. asking the user).",
    )
    skip_training: bool = Field(
        default=False,
        description="Only do dataset discovery, skip training.",
    )

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "AutoHFConfig":
        """Create config from a named preset, with optional overrides.

        Follows AutoGluon's `@apply_presets` pattern where preset values
        are set first, then user overrides take priority.
        """
        if preset_name not in PRESET_CONFIGS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(PRESET_CONFIGS.keys())}"
            )
        params = PRESET_CONFIGS[preset_name].copy()
        params["preset"] = preset_name
        params.update(overrides)
        return cls(**params)


# ---------------------------------------------------------------------------
# Pipeline state — inspired by LangGraph's StateGraph
# ---------------------------------------------------------------------------

class PipelineState(str, Enum):
    """Pipeline execution states, inspired by LangGraph state machines."""
    IDLE = "idle"
    DETECTING_TASK = "detecting_task"
    SEARCHING_DATASETS = "searching_datasets"
    RANKING_DATASETS = "ranking_datasets"
    LOADING_DATASET = "loading_dataset"
    PROFILING_DATASET = "profiling_dataset"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Pipeline data models
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    """Result of task detection."""

    task_type: str = Field(description="HF task tag, e.g. 'text-classification'.")
    task_label: str = Field(description="Human-readable task name.")
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords extracted for dataset search.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence of the task detection (0-1).",
    )
    problem_type: str = Field(
        default="auto",
        description="AutoGluon problem type: 'binary', 'multiclass', 'regression', or 'auto'.",
    )


class DatasetCandidate(BaseModel):
    """A candidate dataset discovered from HF Hub."""

    dataset_id: str
    downloads: int = 0
    likes: int = 0
    tags: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    score: float = Field(default=0.0, description="Ranking score (higher = better).")
    size_category: Optional[str] = Field(
        default=None,
        description="Size category tag from HF (e.g. '10K<n<100K').",
    )


class DatasetProfile(BaseModel):
    """Profile of a loaded dataset, ready for training."""

    dataset_id: str
    num_rows: int
    text_column: str
    label_column: str
    num_classes: int
    class_names: list[str] = Field(default_factory=list)
    sample_texts: list[str] = Field(
        default_factory=list,
        description="A few example texts from the dataset.",
    )
    label_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of labels in the dataset.",
    )
    avg_text_length: float = Field(
        default=0.0,
        description="Average character length of text column.",
    )


class ModelInfo(BaseModel):
    """Info about a model from HF Hub (for future model search)."""

    model_id: str
    pipeline_tag: Optional[str] = None
    downloads: int = 0
    likes: int = 0
    library_name: Optional[str] = None


class TrainResult(BaseModel):
    """Result returned after training completes."""

    task_type: str
    dataset_id: str
    model_path: str
    metrics: dict = Field(default_factory=dict)
    leaderboard: Optional[str] = Field(
        default=None,
        description="Top models leaderboard as formatted string.",
    )
    training_time: float = Field(description="Wall-clock training time in seconds.")
    dataset_profile: Optional[DatasetProfile] = None
    best_model_name: Optional[str] = Field(
        default=None,
        description="Name of the best performing model.",
    )
    num_models_trained: int = Field(
        default=0,
        description="Total number of models trained by AutoGluon.",
    )
    pipeline_state: PipelineState = Field(
        default=PipelineState.COMPLETED,
        description="Final state of the pipeline.",
    )
