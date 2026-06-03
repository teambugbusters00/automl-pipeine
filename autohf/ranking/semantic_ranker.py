"""AutoHF — Semantic ranking using Sentence Transformers and Qdrant.

Uses sentence-transformers for embedding generation and Qdrant (in-memory)
for vector similarity search, with Cross-Encoder reranking.
"""

from __future__ import annotations

import os
from typing import Optional
from loguru import logger

from autohf.core.config import DatasetCandidate

# Constants for models
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class SemanticRanker:
    """Ranks datasets using semantic search and Cross-Encoder reranking."""

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self._embedding_model = None
        self._reranker = None
        self._qdrant_client = None

    def _init_models(self) -> bool:
        """Initialise sentence-transformers and Qdrant client lazily.

        Returns:
            True if initialisation succeeded, False otherwise.
        """
        try:
            from sentence_transformers import SentenceTransformer
            from qdrant_client import QdrantClient

            if self._embedding_model is None:
                logger.info("Loading embedding model '{}'...", self.embedding_model_name)
                self._embedding_model = SentenceTransformer(self.embedding_model_name)

            if self._qdrant_client is None:
                logger.info("Initializing in-memory Qdrant client...")
                self._qdrant_client = QdrantClient(":memory:")

            return True
        except ImportError as e:
            logger.warning(
                "Search dependencies missing. Install autohf[search]. Error: {}", e
            )
            return False
        except Exception as e:
            logger.warning("Failed to initialize semantic ranker: {}", e)
            return False

    def _init_reranker(self) -> bool:
        """Initialise Cross-Encoder reranker model lazily."""
        try:
            from sentence_transformers import CrossEncoder

            if self._reranker is None:
                logger.info("Loading Cross-Encoder reranker '{}'...", self.reranker_model_name)
                self._reranker = CrossEncoder(self.reranker_model_name)
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.warning("Failed to load Cross-Encoder reranker: {}", e)
            return False

    def rank(
        self,
        candidates: list[DatasetCandidate],
        problem_statement: str,
        keywords: list[str],
    ) -> list[DatasetCandidate]:
        """Rank dataset candidates semantically.

        Falls back to keyword ranking if search dependencies are not available.
        """
        if not candidates:
            return []

        # Step 1: Check/Load models
        if not self._init_models():
            logger.info("Semantic ranker unavailable, falling back to keyword ranking.")
            from autohf.ranking.dataset_ranker import rank_datasets
            return rank_datasets(candidates, keywords)

        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            collection_name = "dataset_candidates"

            # Recreate collection
            self._qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE,
                ),
            )

            # Prepare documents to embed
            # We combine: ID, description, and tags into a single text representation
            documents = []
            for candidate in candidates:
                desc = candidate.description or ""
                tags_str = ", ".join(candidate.tags)
                doc_text = (
                    f"Dataset ID: {candidate.dataset_id}\n"
                    f"Description: {desc[:300]}\n"
                    f"Tags: {tags_str}"
                )
                documents.append(doc_text)

            # Embed documents
            logger.info("Computing embeddings for {} datasets...", len(candidates))
            embeddings = self._embedding_model.encode(documents, show_progress_bar=False)

            # Insert into Qdrant
            points = []
            for idx, candidate in enumerate(candidates):
                points.append(
                    PointStruct(
                        id=idx,
                        vector=embeddings[idx].tolist(),
                        payload={
                            "dataset_id": candidate.dataset_id,
                            "index": idx,
                        },
                    )
                )

            self._qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
            )

            # Embed problem statement and search
            logger.info("Searching vector DB for: '{}'...", problem_statement)
            query_vector = self._embedding_model.encode(problem_statement).tolist()
            search_response = self._qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=len(candidates),
            )
            search_results = search_response.points

            # Map search scores (cosine similarities) to candidates
            semantic_scores = {
                res.payload["dataset_id"]: res.score for res in search_results
            }

            # Normalize semantic scores to 0-1
            min_sem = min(semantic_scores.values()) if semantic_scores else 0.0
            max_sem = max(semantic_scores.values()) if semantic_scores else 1.0
            sem_range = max_sem - min_sem
            if sem_range == 0.0:
                sem_range = 1.0

            # Step 2: Cross-Encoder Reranking (pair-wise comparison)
            rerank_scores = {}
            has_reranker = self._init_reranker()
            if has_reranker:
                logger.info("Reranking top candidates using Cross-Encoder...")
                pairs = []
                for candidate in candidates:
                    desc = candidate.description or ""
                    doc_text = f"Dataset: {candidate.dataset_id}. {desc[:200]}"
                    pairs.append((problem_statement, doc_text))

                raw_rerank_scores = self._reranker.predict(pairs)
                for candidate, score in zip(candidates, raw_rerank_scores):
                    rerank_scores[candidate.dataset_id] = float(score)

                # Normalize rerank scores
                min_rr = min(rerank_scores.values()) if rerank_scores else 0.0
                max_rr = max(rerank_scores.values()) if rerank_scores else 1.0
                rr_range = max_rr - min_rr
                if rr_range == 0.0:
                    rr_range = 1.0
            else:
                logger.info("Cross-Encoder reranking skipped (models not available or failed to load).")

            # Pre-compute log scale downloads/likes
            import math
            log_downloads = [math.log1p(c.downloads) for c in candidates]
            log_likes = [math.log1p(c.likes) for c in candidates]
            max_dl = max(log_downloads) if log_downloads else 1.0
            max_lk = max(log_likes) if log_likes else 1.0
            max_dl = max_dl if max_dl > 0.0 else 1.0
            max_lk = max_lk if max_lk > 0.0 else 1.0

            # Composite scoring formula combining vector similarity, quality signals, and metadata
            for i, candidate in enumerate(candidates):
                norm_dl = log_downloads[i] / max_dl
                norm_lk = log_likes[i] / max_lk
                popularity_score = 0.25 * norm_dl + 0.10 * norm_lk

                cosine_sim = semantic_scores.get(candidate.dataset_id, 0.0)
                norm_cosine = (cosine_sim - min_sem) / sem_range

                if has_reranker:
                    rr_score = rerank_scores.get(candidate.dataset_id, 0.0)
                    norm_rr = (rr_score - min_rr) / rr_range
                else:
                    norm_rr = norm_cosine

                quality_score = 0.0
                if candidate.description and len(candidate.description) > 20:
                    quality_score += 0.4
                if len(candidate.tags) >= 3:
                    quality_score += 0.3
                if candidate.downloads > 100:
                    quality_score += 0.3

                # Final Composite score
                final_score = (
                    0.35 * norm_cosine
                    + 0.20 * norm_rr
                    + popularity_score
                    + 0.10 * quality_score
                )
                candidate.score = round(max(0.0, final_score), 4)

            # Sort descending by composite score
            ranked = sorted(candidates, key=lambda c: c.score, reverse=True)

            logger.info(
                "Ranked {} candidates semantically. Top: {} (score={:.4f})",
                len(ranked),
                ranked[0].dataset_id if ranked else "none",
                ranked[0].score if ranked else 0.0,
            )
            return ranked

        except Exception as e:
            logger.error("Semantic ranking failed due to an error: {}", e)
            logger.info("Falling back to keyword-based ranking.")
            from autohf.ranking.dataset_ranker import rank_datasets
            return rank_datasets(candidates, keywords)
