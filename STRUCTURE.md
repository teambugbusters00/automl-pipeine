# AutoHF Architecture & Structure

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         autohf (package)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────┐     ┌───────────────┐     ┌────────────────┐ │
│  │ CLI (Typer)   │     │ Python API    │     │ Background     │ │
│  │ main.py       │────▶│ AutoHF        │     │ Services       │ │
│  └───────────────┘     │ autohf.py     │     │ (future)       │ │
│         │              └─────────┬───────┘     └────────────────┘ │
│         │                        │                                │
│         │               ┌────────▼────────┐                     │
│         │               │  State Machine  │                     │
│         │               │  (PipelineState)│                     │
│         │               └────────┬────────┘                     │
│         │                        │                                │
│         └─────────────────────────┼───────────────────────────────┘
│                                   │
│       ┌───────────────────────────┼───────────────────────────┐
│       │                           │                             │
│  ┌────▼────┐              ┌───────▼──────────┐           ┌─────▼──────┐
│  │ Task    │              │ Dataset          │           │ Training   │
│  │ Agent   │              │ Agent            │           │ AutoGluon  │
│  │ task_   │              │ dataset_         │           │ autogluon_ │
│  │ agent.py│              │ agent.py         │           │ trainer.py │
│  └─────────┘              └────────┬───────┬───┘           └────────────┘
│                                   │       │
│          ┌────────────────────────┘       └──────────────────────────┐
│          │                        │                                  │
│  ┌───────▼──────────┐   ┌───────▼──────────┐              ┌──────────▼──────────┐
│  │ Ranking          │   │ Model Agent      │              │ Output              │
│  │ (datasets &      │   │ model_agent.py   │              │ predictor           │
│  │  models)         │   │ (stub/Phase 2)   │              │ model files         │
│  │                  │   │                  │              │                     │
│  │ dataset_ranker.py│   └────────────────────┘              └─────────────────────┘
│  │ semantic_ranker  │                                                        
│  │ model_ranker.py  │                                                        
│  └──────────────────┘                                                        
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │ Config           │                                                        │
│  │ config.py        │                                                        │
│  │ (Pydantic models)│                                                        │
│  └──────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline Flow (State Machine)

```
User Description
      │
      ▼
┌─────────────────┐
│   IDLE          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DETECTING_TASK  │──▶ TaskAgent (keyword/OpenAI/Gemma)
│                 │──▶ Returns: TaskInfo(task_type, keywords, confidence)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SEARCHING_DATASETS│──▶ DatasetAgent (HF Hub API)
│                 │──▶ Returns: List[DatasetCandidate]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RANKING_DATASETS │──▶ DatasetRanker/SemanticRanker
│                 │──▶ Returns: Ranked DatasetCandidate list
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LOADING_DATASET │──▶ DatasetAgent.load()
│                 │──▶ Returns: (DataFrame, text_col, label_col)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PROFILING_DATASET│──▶ profile_dataset()
│                 │──▶ Returns: DatasetProfile
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TRAINING        │──▶ autogluon_trainer.train_model()
│                 │──▶ Returns: TrainResult
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ COMPLETED       │
└─────────────────┘
```

## Module Details

| Module | File | Responsibilities | Key Functions/Classes |
|--------|------|-----------------|----------------------|
| `cli/main.py` | cli/main.py | Typer CLI entry point | `train()`, `search()`, `info()`, `chat()`, `route()`, `version()` commands |
| `core/autohf.py` | core/autohf.py | Main orchestrator, state machine coordinator | `AutoHF` class with `train()`, `search()` methods |
| `core/config.py` | core/config.py | Pydantic models for configuration and data | `AutoHFConfig`, `TaskInfo`, `DatasetCandidate`, `DatasetProfile`, `TrainResult`, `PipelineState`, `Preset` enum |
| `agents/task_agent.py` | agents/task_agent.py | Intent-to-task detection | `TaskAgent`, `detect_task()`, `list_supported_tasks()` |
| `agents/dataset_agent.py` | agents/dataset_agent.py | HF dataset search, loading, profiling | `DatasetAgent`, `find_datasets()`, `load_dataset_as_dataframe()`, `profile_dataset()` |
| `agents/model_agent.py` | agents/model_agent.py | Model search (Phase 2 stub) | `ModelAgent`, `search()`, `get_model_info()` |
| `agents/chat_agent.py` | agents/chat_agent.py | Local Gemma chat agent | `GemmaChatAgent`, `load()`, `generate()` |
| `ranking/dataset_ranker.py` | ranking/dataset_ranker.py | Keyword-based dataset scoring | `rank_datasets()`, `DatasetRanker` (scoring weights) |
| `ranking/semantic_ranker.py` | ranking/semantic_ranker.py | Semantic ranking with embeddings | `SemanticRanker`, embedding + cross-encoder reranking |
| `ranking/model_ranker.py` | ranking/model_ranker.py | Model ranking (stub/Phase 2) | `rank_models()` function |
| `training/autogluon_trainer.py` | training/autogluon_trainer.py | AutoGluon training wrapper | `train_model()`, `load_predictor()`, `predict()` |

## Hardcoded Values Analysis

### Configuration-based (configurable via `AutoHFConfig`)
These values can be overridden via presets or CLI arguments:
- `time_limit`: Default 300s (medium_quality preset)
- `presets`: Default "medium_quality"
- `max_dataset_rows`: Default 50,000
- `max_datasets_to_search`: Default 20
- `output_dir`: Default "./autohf_output"
- `router`: Default "auto" (keyword/OpenAI/Gemma detection)

### Truly Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `config.py:22-26` | Preset values | `QUICK_PROTOTYPE=60s`, `MEDIUM_QUALITY=300s`, `HIGH_QUALITY=600s`, `BEST_QUALITY=3600s` | Keep as presets; consider making min/max bounds configurable |
| `dataset_agent.py:200-212` | `_LABEL_NAMES` | Priority list for label column detection | Consider moving to config or allow override |
| `dataset_agent.py:207-212` | `_TEXT_NAMES` | Priority list for text column detection | Consider moving to config or allow override |
| `dataset_ranker.py:21-24` | `WEIGHT_*` | Scoring weights (downloads, likes, relevance, quality) | Consider making configurable |
| `semantic_ranker.py:16-17` | Model IDs | `DEFAULT_EMBEDDING_MODEL="all-MiniLM-L6-v2"`, `DEFAULT_RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"` | Already configurable via constructor params |
| `chat_agent.py:106` | `max_new_tokens=512` | Gemma generation token limit | Consider making configurable |
| `task_agent.py:242` | `model="gpt-4o-mini"` | OpenAI model for task detection | Already uses fast, cost-effective model; could be configurable |
| `autohf.py:142-139` | `random_state=42` | Train/test split seed | OK as-is for reproducibility |

### Architecture Patterns Used

1. **State Machine (LangGraph-inspired)**: `PipelineState` enum tracks pipeline progress
2. **Agent Pattern (AutoGen-inspired)**: Separate agents with callable interface
3. **AutoGluon Patterns**: Presets system, TabularPredictor wrapper, leaderboards
4. **OpenHands Patterns**: Retry logic in `_load_best_dataset`, autonomous execution
5. **HF Hub API Patterns**: Filtered search with task+keyword strategies

## Data Flow

```
1. User Input: "sentiment analysis" (CLI or API)
   
2. TaskAgent → TaskInfo:
   - task_type: "text-classification"
   - keywords: ["sentiment", "analysis", "sentiment", "classification"]
   - confidence: 1.0
   - problem_type: "auto"

3. DatasetAgent → DatasetCandidate[]:
   - Query HF Hub with filter + search
   - Multi-strategy: task+keywords → task-only → keywords-only
   - Results: 20 candidates (configurable)

4. DatasetRanker → Ranked DatasetCandidate[]:
   - Score = 0.35×downloads + 0.15×likes + 0.30×relevance + 0.20×quality
   - Sort descending, take top-N

5. Load Dataset → DataFrame:
   - Auto-detect text/label columns
   - Cap at max_dataset_rows
   - Return (df, text_col, label_col)

6. Profile Dataset → DatasetProfile:
   - Statistics: rows, classes, distribution
   - Samples, avg text length

7. AutoGluonTrainer → TrainResult:
   - 80/20 train/test split (stratified)
   - TabularPredictor.fit() with time_limit, presets
   - Evaluate, get leaderboard
   - Return: model_path, metrics, best_model, training_time

8. Output: Console display + saved model files
```