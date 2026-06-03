"""AutoHF — Dataset ranking engine.

Patterns extracted from:
- AutoGluon: Weighted scoring for model selection
- HF Datasets: Tag-based search with relevance scoring
"""

from __future__ import annotations

import math

from loguru import logger

from autohf.core.config import DatasetCandidate


# ---------------------------------------------------------------------------
# Scoring weights — tuned for dataset quality signals
# ---------------------------------------------------------------------------

WEIGHT_DOWNLOADS = 0.35
WEIGHT_LIKES = 0.15
WEIGHT_RELEVANCE = 0.30
WEIGHT_QUALITY = 0.20


def rank_datasets(
    candidates: list[DatasetCandidate],
    keywords: list[str],
) -> list[DatasetCandidate]:
    """Score and sort dataset candidates.

    Scoring formula:
      score = (0.35 × normalized_downloads) +
              (0.15 × normalized_likes) +
              (0.30 × keyword_relevance) +
              (0.20 × quality_signals)

    Quality signals include:
      - Has description (0.3)
      - Has ≥3 tags (0.2)
      - Has >100 downloads (0.2)
      - Good size category (0.3)
    """
    if not candidates:
        logger.warning("No candidates to rank.")
        return []

    # Pre-compute log-scaled values
    log_downloads = [math.log1p(c.downloads) for c in candidates]
    log_likes = [math.log1p(c.likes) for c in candidates]

    max_dl = max(log_downloads) if log_downloads else 1.0
    max_lk = max(log_likes) if log_likes else 1.0

    # Avoid division by zero
    max_dl = max_dl if max_dl > 0 else 1.0
    max_lk = max_lk if max_lk > 0 else 1.0

    keyword_set = set(k.lower() for k in keywords)

    for i, candidate in enumerate(candidates):
        # --- Normalised downloads (0-1) ---
        norm_dl = log_downloads[i] / max_dl

        # --- Normalised likes (0-1) ---
        norm_lk = log_likes[i] / max_lk

        # --- Keyword relevance (enhanced Jaccard) ---
        tag_set = set(t.lower() for t in candidate.tags)
        id_words = set(
            candidate.dataset_id.lower()
            .replace("/", " ")
            .replace("-", " ")
            .replace("_", " ")
            .split()
        )
        # Also check description words
        desc_words = set()
        if candidate.description:
            desc_words = set(candidate.description.lower().split()[:50])

        combined = tag_set | id_words | desc_words

        if keyword_set and combined:
            intersection = keyword_set & combined
            union = keyword_set | combined
            relevance = len(intersection) / len(union)
            # Boost if keywords appear in the dataset ID (strong signal)
            id_boost = len(keyword_set & id_words) * 0.1
            relevance = min(1.0, relevance + id_boost)
        else:
            relevance = 0.0

        # --- Quality signals ---
        quality = 0.0

        # Description quality
        if candidate.description and len(candidate.description) > 20:
            quality += 0.3

        # Tag richness
        if len(candidate.tags) >= 3:
            quality += 0.2

        # Download threshold
        if candidate.downloads > 100:
            quality += 0.2

        # Size category bonus (prefer medium-sized datasets)
        if candidate.size_category:
            good_sizes = ["1K<n<10K", "10K<n<100K", "100K<n<1M"]
            if candidate.size_category in good_sizes:
                quality += 0.3
            elif candidate.size_category in ["n<1K"]:
                quality -= 0.1  # Penalise very small datasets

        # --- Composite score ---
        score = (
            WEIGHT_DOWNLOADS * norm_dl
            + WEIGHT_LIKES * norm_lk
            + WEIGHT_RELEVANCE * relevance
            + WEIGHT_QUALITY * quality
        )

        candidate.score = round(max(0, score), 4)

    # Sort descending
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)

    logger.info(
        "Ranked {} datasets. Top: {} (score={:.4f})",
        len(ranked),
        ranked[0].dataset_id if ranked else "none",
        ranked[0].score if ranked else 0,
    )

    return ranked
