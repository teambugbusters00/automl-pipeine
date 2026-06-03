"""AutoHF — Dataset discovery and loading agent.

Patterns extracted from:
- HF Hub: list_datasets with filter, sort for search
- HF Datasets: load_dataset, load_dataset_builder for metadata inspection
- AutoGen: Agent class pattern with callable interface
- AutoGluon: Data validation and feature detection logic
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from huggingface_hub import HfApi
from loguru import logger

from autohf.core.config import AutoHFConfig, DatasetCandidate, DatasetProfile


# ---------------------------------------------------------------------------
# Agent class — inspired by AutoGen agent pattern
# ---------------------------------------------------------------------------

class DatasetAgent:
    """Agent responsible for dataset discovery and loading.

    Follows AutoGen's agent pattern: clear responsibility, callable interface.
    """

    def __init__(self, config: AutoHFConfig | None = None) -> None:
        self.config = config or AutoHFConfig()
        self.api = HfApi()
        self.search_history: list[list[DatasetCandidate]] = []

    def search(self, task_type: str, keywords: list[str]) -> list[DatasetCandidate]:
        """Search for datasets matching a task."""
        candidates = find_datasets(task_type, keywords, self.config)
        self.search_history.append(candidates)
        return candidates

    def load(self, dataset_id: str) -> tuple[pd.DataFrame, str, str]:
        """Load a dataset into a DataFrame."""
        return load_dataset_as_dataframe(dataset_id, self.config.max_dataset_rows)

    def profile(self, df: pd.DataFrame, dataset_id: str, text_col: str, label_col: str) -> DatasetProfile:
        """Profile a loaded dataset."""
        return profile_dataset(df, dataset_id, text_col, label_col)


# ---------------------------------------------------------------------------
# Dataset discovery — uses HF Hub API patterns
# ---------------------------------------------------------------------------

def find_datasets(
    task_type: str,
    keywords: list[str],
    config: AutoHFConfig | None = None,
) -> list[DatasetCandidate]:
    """Search HF Hub for datasets matching a task type.

    Uses multiple search strategies inspired by HF Hub's API:
      1. Filtered search with task tag + keywords (most specific)
      2. Broader search with task tag only (fallback)
      3. Keyword-only search (last resort)
    """
    if config is None:
        config = AutoHFConfig()

    api = HfApi()
    search_query = " ".join(keywords[:5])
    candidates: list[DatasetCandidate] = []
    seen_ids: set[str] = set()

    logger.info(
        "Searching HF Hub — task='{}', query='{}', limit={}",
        task_type,
        search_query,
        config.max_datasets_to_search,
    )

    # --- Strategy 1: Task filter + keyword search ---
    try:
        results = api.list_datasets(
            filter=task_type,
            search=search_query if search_query.strip() else None,
            sort="downloads",
            limit=config.max_datasets_to_search,
        )
        for ds in results:
            if ds.id not in seen_ids:
                seen_ids.add(ds.id)
                candidates.append(_make_candidate(ds))
    except Exception as e:
        logger.warning("Strategy 1 (filtered+keyword) failed: {}", e)

    # --- Strategy 2: Task filter only (broader) ---
    if len(candidates) < 5:
        try:
            logger.info("Broadening search (task filter only)...")
            results = api.list_datasets(
                filter=task_type,
                sort="downloads",
                limit=config.max_datasets_to_search,
            )
            for ds in results:
                if ds.id not in seen_ids:
                    seen_ids.add(ds.id)
                    candidates.append(_make_candidate(ds))
        except Exception as e:
            logger.warning("Strategy 2 (task-only) failed: {}", e)

    # --- Strategy 3: Keyword search only (no task filter) ---
    if len(candidates) < 3:
        try:
            logger.info("Last resort: keyword-only search...")
            results = api.list_datasets(
                search=search_query,
                sort="downloads",
                limit=config.max_datasets_to_search,
            )
            for ds in results:
                if ds.id not in seen_ids:
                    seen_ids.add(ds.id)
                    candidates.append(_make_candidate(ds))
        except Exception as e:
            logger.warning("Strategy 3 (keyword-only) failed: {}", e)

    logger.info("Total candidates found: {}", len(candidates))

    if not candidates:
        raise RuntimeError(
            f"No datasets found for task '{task_type}' with keywords {keywords}. "
            "Try a different or more specific task description."
        )

    return candidates


def _make_candidate(ds) -> DatasetCandidate:
    """Convert HF DatasetInfo to our DatasetCandidate model."""
    # Extract size category from tags if available
    size_cat = None
    tags = ds.tags or []
    for tag in tags:
        if tag.startswith("size_categories:"):
            size_cat = tag.replace("size_categories:", "")
            break

    return DatasetCandidate(
        dataset_id=ds.id,
        downloads=ds.downloads or 0,
        likes=ds.likes or 0,
        tags=tags,
        description=getattr(ds, "description", None),
        size_category=size_cat,
    )


# ---------------------------------------------------------------------------
# Model search — for future model discovery (Phase 2 prep)
# ---------------------------------------------------------------------------

def find_models(
    task_type: str,
    limit: int = 10,
) -> list[dict]:
    """Search HF Hub for models matching a task type.

    Uses pipeline_tag for precise filtering (from HF Hub API patterns).
    Prepared for Phase 2 model selection.
    """
    api = HfApi()

    try:
        models = api.list_models(
            pipeline_tag=task_type,
            sort="downloads",
            limit=limit,
        )
        return [
            {
                "model_id": m.id,
                "pipeline_tag": m.pipeline_tag,
                "downloads": m.downloads or 0,
                "likes": m.likes or 0,
            }
            for m in models
        ]
    except Exception as e:
        logger.warning("Model search failed: {}", e)
        return []


# ---------------------------------------------------------------------------
# Dataset loading — inspired by HF Datasets load patterns
# ---------------------------------------------------------------------------

# Common label column names (priority order)
_LABEL_NAMES = [
    "label", "labels", "target", "class", "sentiment", "category",
    "intent", "emotion", "tag", "ner_tags", "pos_tags",
    "output", "answer", "rating", "score",
]

# Common text column names (priority order)
_TEXT_NAMES = [
    "text", "sentence", "review", "content", "question", "context",
    "input", "document", "title", "body", "comment", "tweet",
    "sentence1", "premise", "hypothesis", "message", "description",
    "abstract", "article", "passage",
]


def load_dataset_as_dataframe(
    dataset_id: str,
    max_rows: int = 50_000,
) -> tuple[pd.DataFrame, str, str]:
    """Load a HF dataset into a pandas DataFrame.

    Uses load_dataset_builder for metadata inspection (HF Datasets pattern)
    before downloading the full dataset.

    Auto-detects the text and label columns using heuristics inspired by
    AutoGluon's feature type detection.
    """
    from datasets import load_dataset, get_dataset_config_names

    logger.info("Loading dataset '{}' (max {} rows)...", dataset_id, max_rows)

    try:
        # --- Inspect metadata first (HF Datasets pattern) ---
        configs = None
        try:
            configs = get_dataset_config_names(dataset_id)
            logger.debug("Available configs: {}", configs)
        except Exception:
            pass

        config_name = None
        if configs:
            for preferred in ["default", "plain_text", "en"]:
                if preferred in configs:
                    config_name = preferred
                    break
            if config_name is None:
                config_name = configs[0]

        # --- Load the train split ---
        ds = load_dataset(
            dataset_id,
            name=config_name,
            split="train",
        )

        # Cap rows
        if len(ds) > max_rows:
            ds = ds.shuffle(seed=42).select(range(max_rows))
            logger.info("Capped dataset to {} rows.", max_rows)

        df = ds.to_pandas()
        logger.info("Loaded DataFrame: {} rows × {} columns.", len(df), len(df.columns))

    except Exception as e:
        logger.error("Failed to load dataset '{}': {}", dataset_id, e)
        raise RuntimeError(f"Could not load dataset '{dataset_id}': {e}") from e

    # --- Detect label column ---
    label_col = _detect_column(df, _LABEL_NAMES, strategy="label")
    if label_col is None:
        raise RuntimeError(
            f"Could not detect label column in '{dataset_id}'. "
            f"Columns: {list(df.columns)}"
        )

    # --- Detect text column ---
    text_col = _detect_column(df, _TEXT_NAMES, strategy="text")
    if text_col is None:
        raise RuntimeError(
            f"Could not detect text column in '{dataset_id}'. "
            f"Columns: {list(df.columns)}"
        )

    logger.success(
        "Detected columns — text='{}', label='{}'",
        text_col,
        label_col,
    )

    return df, text_col, label_col


def profile_dataset(
    df: pd.DataFrame,
    dataset_id: str,
    text_col: str,
    label_col: str,
) -> DatasetProfile:
    """Create a profile summary of the loaded dataset.

    Inspired by AutoGluon's data profiling that happens before training.
    """
    unique_labels = df[label_col].dropna().unique()
    class_names = [str(c) for c in sorted(unique_labels)]

    # Sample texts
    samples = df[text_col].dropna().head(3).tolist()
    sample_texts = [str(s)[:200] for s in samples]

    # Label distribution
    label_dist = df[label_col].value_counts().head(20).to_dict()
    label_dist = {str(k): int(v) for k, v in label_dist.items()}

    # Avg text length
    avg_len = df[text_col].dropna().astype(str).str.len().mean()

    return DatasetProfile(
        dataset_id=dataset_id,
        num_rows=len(df),
        text_column=text_col,
        label_column=label_col,
        num_classes=len(unique_labels),
        class_names=class_names[:20],  # Cap at 20 class names
        sample_texts=sample_texts,
        label_distribution=label_dist,
        avg_text_length=round(avg_len, 1),
    )


# ---------------------------------------------------------------------------
# Column detection — inspired by AutoGluon feature metadata detection
# ---------------------------------------------------------------------------

def _detect_column(
    df: pd.DataFrame,
    known_names: list[str],
    strategy: str = "text",
) -> Optional[str]:
    """Detect a column by name heuristics, then by content analysis.

    Follows AutoGluon's approach of inferring feature types from data content
    when explicit metadata isn't available.
    """
    columns_lower = {c.lower(): c for c in df.columns}

    # 1. Exact name match (case-insensitive)
    for name in known_names:
        if name in columns_lower:
            return columns_lower[name]

    # 2. Partial name match
    for name in known_names:
        for col_lower, col_orig in columns_lower.items():
            if name in col_lower:
                return col_orig

    # 3. Content-based fallback
    if strategy == "label":
        return _detect_label_by_content(df)
    elif strategy == "text":
        return _detect_text_by_content(df)

    return None


def _detect_label_by_content(df: pd.DataFrame) -> Optional[str]:
    """Find label column by looking for low-cardinality columns.

    AutoGluon uses similar logic: columns with low cardinality relative
    to dataset size are likely categorical/label columns.
    """
    best_col = None
    best_ratio = float("inf")

    for col in df.columns:
        n_unique = df[col].nunique()
        n_total = len(df)

        if 2 <= n_unique <= 200:
            ratio = n_unique / n_total
            if ratio < best_ratio:
                best_ratio = ratio
                best_col = col

    if best_col:
        logger.debug("Label fallback: '{}' (cardinality ratio: {:.4f})", best_col, best_ratio)
    return best_col


def _detect_text_by_content(df: pd.DataFrame) -> Optional[str]:
    """Find text column by looking for the column with longest avg string.

    AutoGluon marks high-entropy string columns as 'text' features.
    """
    best_col = None
    best_avg_len = 0.0

    for col in df.columns:
        if df[col].dtype == object:
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > best_avg_len:
                best_avg_len = avg_len
                best_col = col

    if best_col:
        logger.debug("Text fallback: '{}' (avg length: {:.1f})", best_col, best_avg_len)
    return best_col
