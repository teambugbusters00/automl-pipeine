# AutoHF

**One-line AutoML: from idea to trained model using Hugging Face + AutoGluon.**

AutoHF is an autonomous machine learning pipeline that takes a natural language description of a task (e.g., "sentiment analysis") and automatically finds the best datasets on Hugging Face, ranks them by quality, and trains a state-of-the-art model using AutoGluon.

---

## Features

- **Intent-to-Task:** Automatically detects ML task types (classification, regression, etc.) and keywords from natural language.
- **Autonomous Dataset Discovery:** Searches the Hugging Face Hub for relevant datasets using multi-strategy search.
- **Intelligent Ranking:** Ranks datasets based on quality signals like downloads, likes, and metadata completeness.
- **Automated Training:** Leverages AutoGluon to train high-quality models with minimal configuration.
- **Agentic Architecture:** Inspired by patterns from AutoGen, LangGraph, and OpenHands for robust state management and collaboration.
- **Interactive Gemma Chat:** Run a single prompt or start an interactive chat session with local Gemma models.

---

## Installation

```bash
# Basic installation
pip install autohf

# With training support (recommended)
pip install "autohf[train]"
```

---

## CLI Quick Start (Step-by-Step)

AutoHF provides a simple command-line interface:

### Step 1: Detect and Train a Model
To find the best datasets and train a model directly from a task description:
```bash
autohf train "sentiment analysis"
```
Or with custom presets and training limits:
```bash
autohf train "spam detection" --preset high_quality --time-limit 600
```

### Step 2: Search and Rank Datasets
If you only want to discover and rank the top Hugging Face datasets for your task without training:
```bash
autohf search "question answering"
```
You can also list top models for the task:
```bash
autohf search "question answering" --models
```

### Step 3: Interactive local Gemma Chat
To query or chat with a local Gemma model (such as `google/gemma-4-E2B-it`):
```bash
# Start an interactive multi-turn chat REPL session
autohf chat

# Or run a single prompt query directly
autohf chat "Explain AutoML in one sentence."
```
*Note: Make sure your `HF_TOKEN` environment variable is set to download the model.*

### Step 4: Show package info and supported task types
```bash
autohf info
```

---

## Python API Usage

```python
from autohf import AutoHF

# Initialize and train
hf = AutoHF.from_preset("medium_quality")
result = hf.train("customer review classification")

# Access results
print(f"Best model: {result.best_model_name}")
print(f"Accuracy: {result.metrics['accuracy']}")
print(f"Model saved at: {result.model_path}")
```

---

## License

MIT License. See `LICENSE` for details.
