"""AutoHF — FastAPI Web Service.

Exposes AutoHF features over a HTTP REST API.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from autohf.core.autohf import AutoHF
from autohf.core.config import AutoHFConfig, DatasetCandidate


app = FastAPI(
    title="🚀 AutoHF API",
    description="One-line AutoML API: from idea to trained model using Hugging Face + AutoGluon",
    version="0.1.0",
)


# --- Request/Response Models ---

class SearchRequest(BaseModel):
    task_description: str = Field(
        ..., 
        example="Build an AI that detects fake reviews",
        description="Natural language prompt describing the ML task."
    )
    router: str = Field(
        "auto", 
        example="auto",
        description="Routing strategy: 'auto', 'keyword', 'openai', or 'gemma'."
    )
    top_n: int = Field(
        10, 
        ge=1, 
        le=50, 
        description="Number of top dataset candidates to return."
    )


class TrainRequest(BaseModel):
    task_description: str = Field(
        ...,
        example="sentiment analysis",
        description="Natural language prompt describing the ML task."
    )
    preset: str = Field(
        "medium_quality",
        example="quick_prototype",
        description="AutoHF preset: quick_prototype, medium_quality, high_quality, best_quality."
    )
    time_limit: Optional[int] = Field(
        None,
        description="Override the training time limit in seconds."
    )
    max_dataset_rows: Optional[int] = Field(
        None,
        description="Override the maximum dataset rows to load."
    )
    router: str = Field(
        "auto",
        description="Routing strategy: 'auto', 'keyword', 'openai', or 'gemma'."
    )


# --- Endpoints ---

@app.get("/")
def read_root():
    return {
        "message": "Welcome to AutoHF API!",
        "docs_url": "/docs",
        "status": "running"
    }


@app.post("/search", response_model=List[DatasetCandidate])
def search_datasets(request: SearchRequest):
    """Search and rank Hugging Face datasets for a given goal."""
    try:
        config = AutoHFConfig(router=request.router)
        hf = AutoHF(config=config)
        results = hf.search(request.task_description, top_n=request.top_n)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=Dict[str, Any])
def train_model(request: TrainRequest):
    """Run full AutoML pipeline (detect, search, rank, load, profile, train).

    Note: Training blocks the request until completion. Use a low time_limit for tests.
    """
    try:
        overrides = {"router": request.router}
        if request.time_limit is not None:
            overrides["time_limit"] = request.time_limit
        if request.max_dataset_rows is not None:
            overrides["max_dataset_rows"] = request.max_dataset_rows

        hf = AutoHF.from_preset(request.preset, **overrides)
        result = hf.train(request.task_description)
        
        return {
            "task_type": result.task_type,
            "dataset_id": result.dataset_id,
            "model_path": result.model_path,
            "best_model_name": result.best_model_name,
            "num_models_trained": result.num_models_trained,
            "training_time_seconds": result.training_time,
            "metrics": result.metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
def get_info():
    """Retrieve details on supported tasks and presets."""
    from autohf.agents.task_agent import list_supported_tasks
    from autohf.core.config import PRESET_CONFIGS
    
    return {
        "supported_tasks": list_supported_tasks(),
        "presets": PRESET_CONFIGS,
    }
