"""AutoHF — AutoGluon training wrapper.

Patterns extracted from:
- AutoGluon TabularPredictor: fit(), evaluate(), leaderboard(), feature_importance()
- AutoGluon presets system: preset configs, hyperparameter configs
- AutoGluon metrics: Scorer pattern, eval_metric auto-selection
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from autohf.core.config import (
    AutoHFConfig,
    TrainResult,
    DatasetProfile,
    PipelineState,
    TASK_EVAL_METRICS,
    TASK_HYPERPARAMETERS,
)


# ---------------------------------------------------------------------------
# Eval metric mapping — from AutoGluon core/metrics
# ---------------------------------------------------------------------------

# AutoGluon's default metrics per problem type
_PROBLEM_TYPE_METRICS: dict[str, str] = {
    "binary": "accuracy",
    "multiclass": "accuracy",
    "regression": "root_mean_squared_error",
}

# Available classification metrics (from AutoGluon)
CLASSIFICATION_METRICS = [
    "accuracy", "balanced_accuracy", "log_loss",
    "f1", "f1_macro", "f1_micro", "f1_weighted",
    "roc_auc", "precision", "recall", "mcc",
]

# Available regression metrics
REGRESSION_METRICS = [
    "root_mean_squared_error", "mean_squared_error",
    "mean_absolute_error", "r2",
]


def get_default_metric(task_type: str) -> str:
    """Return the default evaluation metric for a task type."""
    return TASK_EVAL_METRICS.get(task_type, "accuracy")


# ---------------------------------------------------------------------------
# Training — mirrors AutoGluon TabularPredictor.fit() pattern
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    task_type: str,
    dataset_id: str,
    config: AutoHFConfig,
    dataset_profile: DatasetProfile | None = None,
) -> TrainResult:
    """Train a model using AutoGluon TabularPredictor.

    Follows AutoGluon's training pattern:
      1. Validate data (check minimum rows, class distribution)
      2. Train/test split (80/20 with stratification if possible)
      3. Configure predictor (label, path, eval_metric)
      4. Fit with time_limit, presets, hyperparameters
      5. Evaluate on test set
      6. Extract leaderboard (AutoGluon's signature feature)
      7. Optionally optimize for deployment
      8. Return TrainResult with all metadata
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise ImportError(
            "AutoGluon is required for training. Install it with:\n"
            "  pip install autohf[train]\n"
            "or:\n"
            "  pip install 'autogluon.tabular[all]'"
        )

    from sklearn.model_selection import train_test_split

    # --- Step 1: Validate & prepare data ---
    logger.info("Preparing data for training...")

    train_df = df[[text_column, label_column]].dropna().copy()
    logger.info("Training data: {} rows after dropping NaN.", len(train_df))

    if len(train_df) < 20:
        raise RuntimeError(
            f"Dataset too small for training ({len(train_df)} rows). "
            "Need at least 20 rows."
        )

    # Log class distribution (AutoGluon does this)
    class_counts = train_df[label_column].value_counts()
    logger.info("Class distribution:\n{}", class_counts.to_string())

    # Warn about class imbalance (AutoGluon pattern)
    if len(class_counts) >= 2:
        imbalance_ratio = class_counts.max() / class_counts.min()
        if imbalance_ratio > 10:
            logger.warning(
                "Severe class imbalance detected (ratio: {:.1f}:1). "
                "Consider using eval_metric='balanced_accuracy' or 'f1_macro'.",
                imbalance_ratio,
            )

    # --- Step 2: Train/test split ---
    can_stratify = _can_stratify(train_df, label_column)
    train_data, test_data = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df[label_column] if can_stratify else None,
    )

    logger.info(
        "Split: {} train / {} test (stratified={})",
        len(train_data),
        len(test_data),
        can_stratify,
    )

    # --- Step 3: Configure eval metric ---
    eval_metric = config.eval_metric or get_default_metric(task_type)
    logger.info("Eval metric: {}", eval_metric)

    # --- Step 4: Output directory ---
    output_path = Path(config.output_dir) / f"model_{dataset_id.replace('/', '_')}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 5: Build hyperparameters ---
    # Use task-specific hyperparameters if available (AutoGluon pattern)
    hyperparameters = config.hyperparameters
    if hyperparameters is None and task_type in TASK_HYPERPARAMETERS:
        hyperparameters = TASK_HYPERPARAMETERS[task_type]
        logger.info("Using task-specific hyperparameters for '{}'", task_type)

    # --- Step 6: Train (mirrors TabularPredictor.fit()) ---
    logger.info(
        "Starting AutoGluon training:\n"
        "  time_limit={}s\n"
        "  presets='{}'\n"
        "  eval_metric='{}'\n"
        "  hyperparameters={}",
        config.time_limit,
        config.presets,
        eval_metric,
        "custom" if hyperparameters else "default",
    )

    start_time = time.time()

    predictor = TabularPredictor(
        label=label_column,
        path=str(output_path),
        eval_metric=eval_metric,
        verbosity=2,
    )

    fit_kwargs = {
        "train_data": train_data,
        "time_limit": config.time_limit,
        "presets": config.presets,
    }
    if hyperparameters:
        fit_kwargs["hyperparameters"] = hyperparameters

    predictor.fit(**fit_kwargs)

    training_time = round(time.time() - start_time, 2)
    logger.success("Training completed in {:.1f}s.", training_time)

    # --- Step 7: Evaluate (AutoGluon's evaluate + leaderboard) ---
    logger.info("Evaluating on test set...")
    performance = predictor.evaluate(test_data)

    # Get full leaderboard (AutoGluon's signature feature)
    leaderboard_str = None
    num_models = 0
    best_model_name = None
    try:
        leaderboard_df = predictor.leaderboard(test_data, silent=True)
        num_models = len(leaderboard_df)
        leaderboard_str = leaderboard_df.head(10).to_string()
        if len(leaderboard_df) > 0:
            best_model_name = leaderboard_df.iloc[0]["model"]
    except Exception as e:
        logger.warning("Could not generate leaderboard: {}", e)

    # Build metrics dict
    metrics = {}
    if isinstance(performance, dict):
        metrics = performance
    else:
        metrics[eval_metric] = performance

    logger.success("Evaluation metrics: {}", metrics)
    if best_model_name:
        logger.success("Best model: {}", best_model_name)

    # --- Step 8: Optimize for deployment (AutoGluon pattern) ---
    if config.keep_only_best:
        try:
            predictor.delete_models(models_to_keep="best", dry_run=False)
            logger.info("Deleted all models except the best one.")
        except Exception as e:
            logger.warning("Could not delete models: {}", e)

    if config.save_space:
        try:
            predictor.save_space()
            logger.info("Saved space by removing training artifacts.")
        except Exception as e:
            logger.warning("Could not save space: {}", e)

    return TrainResult(
        task_type=task_type,
        dataset_id=dataset_id,
        model_path=str(output_path),
        metrics=metrics,
        leaderboard=leaderboard_str,
        training_time=training_time,
        dataset_profile=dataset_profile,
        best_model_name=best_model_name,
        num_models_trained=num_models,
        pipeline_state=PipelineState.COMPLETED,
    )


# ---------------------------------------------------------------------------
# Prediction utility — for using trained models
# ---------------------------------------------------------------------------

def load_predictor(model_path: str):
    """Load a previously trained AutoGluon predictor."""
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise ImportError("AutoGluon is required. Install: pip install autohf[train]")

    return TabularPredictor.load(model_path)


def predict(model_path: str, data: pd.DataFrame) -> pd.Series:
    """Make predictions using a trained model."""
    predictor = load_predictor(model_path)
    return predictor.predict(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _can_stratify(df: pd.DataFrame, label_column: str) -> bool:
    """Check if stratification is possible (each class needs ≥2 samples)."""
    try:
        value_counts = df[label_column].value_counts()
        return value_counts.min() >= 2
    except Exception:
        return False
