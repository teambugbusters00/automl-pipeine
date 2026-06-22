"""AutoHF — One-line AutoML: from idea to trained model.

Patterns from AutoGluon, HF Hub, LangGraph, AutoGen, and OpenHands.

Usage::

    from autohf import AutoHF

    # Quick prototype
    result = AutoHF().train("sentiment analysis")

    # With preset (AutoGluon-inspired)
    result = AutoHF.from_preset("best_quality").train("NER")

    # Custom config
    from autohf import AutoHFConfig
    config = AutoHFConfig(time_limit=600, presets="high_quality")
    result = AutoHF(config=config).train("spam detection")

    # Search datasets only
    datasets = AutoHF().search("question answering", top_n=10)
"""

from autohf.core.autohf import AutoHF
from autohf.core.config import AutoHFConfig, TrainResult, DatasetCandidate, PipelineState
from autohf.agents.chat_agent import GemmaChatAgent

__version__ = "1.0.1"
__all__ = [
    "AutoHF",
    "AutoHFConfig",
    "TrainResult",
    "DatasetCandidate",
    "PipelineState",
    "GemmaChatAgent",
]
