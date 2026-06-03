"""AutoHF — Task detection agent.

Maps natural-language task descriptions to Hugging Face task tags using
keyword matching with fuzzy fallback.

Patterns extracted from:
- AutoGluon: problem_type detection, eval_metric selection
- HF Hub: pipeline_tag taxonomy
- AutoGen: clear agent interface with __call__
"""

from __future__ import annotations

import difflib
import re
from typing import Optional

from loguru import logger

from autohf.core.config import (
    TaskInfo,
    TASK_TO_PROBLEM_TYPE,
    TASK_EVAL_METRICS,
)


# ---------------------------------------------------------------------------
# Task → keyword mapping (expanded from HF Hub pipeline_tag taxonomy)
# ---------------------------------------------------------------------------

TASK_MAP: dict[str, dict] = {
    "text-classification": {
        "label": "Text Classification",
        "keywords": [
            "sentiment", "sentiment analysis", "classification", "classify",
            "spam", "spam detection", "toxic", "toxicity", "emotion",
            "review", "hate speech", "hate", "offensive", "sarcasm",
            "irony", "fake news", "fake review", "clickbait", "intent",
            "topic", "topic classification",
            "category", "categorize", "positive negative", "polarity",
            "opinion", "stance", "abuse", "cyberbullying",
            "product review", "movie review", "app review",
            "news classification", "language detection",
            "duplicate detection", "paraphrase detection",
            "natural language inference", "textual entailment",
        ],
        "aliases": ["sentiment-analysis", "nli"],
    },
    "token-classification": {
        "label": "Token Classification",
        "keywords": [
            "ner", "named entity", "named entity recognition",
            "entity extraction", "entity", "pos", "pos tagging",
            "part of speech", "chunking", "slot filling", "tagging",
            "token classification", "sequence labeling",
            "information extraction", "keyphrase extraction",
        ],
        "aliases": ["ner"],
    },
    "question-answering": {
        "label": "Question Answering",
        "keywords": [
            "question answering", "qa", "question answer",
            "reading comprehension", "squad", "extractive qa",
            "answer extraction", "comprehension",
            "open domain qa", "closed domain qa",
        ],
        "aliases": ["qa", "extractive-qa"],
    },
    "summarization": {
        "label": "Summarization",
        "keywords": [
            "summarize", "summarization", "summary", "abstract",
            "abstractive", "extractive summary", "tldr", "tl;dr",
            "condense", "digest", "brief", "headline generation",
        ],
        "aliases": [],
    },
    "translation": {
        "label": "Translation",
        "keywords": [
            "translate", "translation", "language pair", "bilingual",
            "multilingual", "machine translation", "mt",
            "english to french", "english to german", "english to spanish",
            "english to hindi", "english to chinese",
            "en to fr", "en to de", "en to es", "en to hi",
        ],
        "aliases": [],
    },
    "text-generation": {
        "label": "Text Generation",
        "keywords": [
            "text generation", "generate text", "gpt", "language model",
            "lm", "causal lm", "story", "story writing", "creative writing",
            "autocomplete", "next word", "chatbot", "dialogue",
            "conversational", "chat",
        ],
        "aliases": ["conversational"],
    },
    "fill-mask": {
        "label": "Fill Mask",
        "keywords": [
            "fill mask", "masked language", "mask prediction",
            "cloze", "mlm", "masked lm", "bert",
        ],
        "aliases": [],
    },
    "text2text-generation": {
        "label": "Text-to-Text Generation",
        "keywords": [
            "text to text", "text2text", "t5", "seq2seq",
            "paraphrase", "grammar correction", "rewrite",
            "style transfer", "simplification", "text simplification",
        ],
        "aliases": [],
    },
    "zero-shot-classification": {
        "label": "Zero-Shot Classification",
        "keywords": [
            "zero shot", "zero-shot", "zero shot classification",
        ],
        "aliases": [],
    },
    # --- Additional HF task types ---
    "image-classification": {
        "label": "Image Classification",
        "keywords": [
            "image classification", "image recognition",
            "image categorization", "visual classification",
            "photo classification",
        ],
        "aliases": [],
    },
    "tabular-classification": {
        "label": "Tabular Classification",
        "keywords": [
            "tabular classification", "tabular", "structured data",
            "csv classification", "table classification",
            "churn prediction", "credit scoring", "fraud detection",
            "customer segmentation",
        ],
        "aliases": [],
    },
    "tabular-regression": {
        "label": "Tabular Regression",
        "keywords": [
            "tabular regression", "price prediction",
            "house price", "stock prediction", "forecasting",
            "time series", "demand prediction", "sales prediction",
        ],
        "aliases": [],
    },
}

# Build a flat lookup: keyword → task_type
_KEYWORD_INDEX: dict[str, str] = {}
for _task, _info in TASK_MAP.items():
    for _kw in _info["keywords"]:
        _KEYWORD_INDEX[_kw.lower()] = _task
    for _alias in _info.get("aliases", []):
        _KEYWORD_INDEX[_alias.lower()] = _task

# All keywords sorted longest-first so longer matches win
_ALL_KEYWORDS = sorted(_KEYWORD_INDEX.keys(), key=len, reverse=True)

DEFAULT_TASK = "text-classification"


# ---------------------------------------------------------------------------
# Agent class — inspired by AutoGen's agent interface
# ---------------------------------------------------------------------------

class TaskAgent:
    """Agent that detects ML task type from natural language.

    Follows the agent pattern from Microsoft AutoGen:
    each agent has a clear responsibility and a callable interface.
    """

    def __init__(self, router: str = "auto") -> None:
        self.router = router
        self.history: list[TaskInfo] = []

    def __call__(self, description: str, router: Optional[str] = None) -> TaskInfo:
        """Detect task — callable interface like AutoGen agents."""
        result = detect_task(description, router or self.router)
        self.history.append(result)
        return result

    def reset(self) -> None:
        self.history.clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _detect_task_openai(description: str, force: bool = False) -> Optional[TaskInfo]:
    import os
    if "OPENAI_API_KEY" not in os.environ:
        if force:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required when forcing the 'openai' router."
            )
        return None
    try:
        from openai import OpenAI
        from autohf.core.config import TASK_TO_PROBLEM_TYPE
        
        client = OpenAI()
        
        allowed_tasks = list(TASK_MAP.keys())
        system_prompt = f"""You are an AI assistant that maps a natural language description of a machine learning goal to the most appropriate Hugging Face pipeline tag / task type.
        
Allowed Hugging Face task types:
{allowed_tasks}

TASK MAP details:
{ {k: v['label'] for k, v in TASK_MAP.items()} }

You MUST respond with a raw JSON object containing exactly these keys:
- "task_type": string, must be one of the allowed task types above.
- "task_label": string, the human readable label for the task.
- "keywords": list of strings, 3-5 search keywords to query dataset hub.
- "confidence": float between 0.0 and 1.0.
- "problem_type": string, one of "binary", "multiclass", "regression", or "auto".

For text-classification, set "problem_type" to "auto".
For translation, summarization, text-generation, text2text-generation, tabular-regression, set "problem_type" to "regression".
For token-classification, question-answering, fill-mask, zero-shot-classification, tabular-classification, set "problem_type" to "multiclass".

Example output:
{{
  "task_type": "text-classification",
  "task_label": "Text Classification",
  "keywords": ["reviews", "fake", "spam"],
  "confidence": 0.95,
  "problem_type": "auto"
}}
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Problem description: '{description}'"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        import json
        data = json.loads(response.choices[0].message.content)
        task_type = data.get("task_type", DEFAULT_TASK)
        if task_type not in TASK_MAP:
            task_type = DEFAULT_TASK
            
        task_meta = TASK_MAP[task_type]
        
        logger.success(
            "OpenAI detected task: '{}' → {} (confidence: {})",
            description,
            task_type,
            data.get("confidence", 1.0),
        )
        
        return TaskInfo(
            task_type=task_type,
            task_label=data.get("task_label", task_meta["label"]),
            keywords=data.get("keywords", _extract_search_keywords(description, task_type)),
            confidence=float(data.get("confidence", 0.9)),
            problem_type=data.get("problem_type", TASK_TO_PROBLEM_TYPE.get(task_type, "auto")),
        )
    except Exception as e:
        if force:
            raise
        logger.warning("OpenAI task detection failed: {}", e)
        return None


def _detect_task_gemma(description: str, force: bool = False) -> Optional[TaskInfo]:
    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        if force:
            raise ValueError(
                "HF_TOKEN environment variable is required when forcing the 'gemma' router."
            )
        return None
        
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from autohf.core.config import TASK_TO_PROBLEM_TYPE
        
        logger.info("Loading local Gemma model 'google/gemma-4-E2B-it'...")
        
        # Load processor and model
        processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it", token=token)
        model = AutoModelForImageTextToText.from_pretrained(
            "google/gemma-4-E2B-it", 
            token=token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        allowed_tasks = list(TASK_MAP.keys())
        prompt = f"""You are an AI assistant that maps a natural language description of a machine learning goal to the most appropriate Hugging Face pipeline tag / task type.
        
Allowed Hugging Face task types:
{allowed_tasks}

TASK MAP details:
{ {k: v['label'] for k, v in TASK_MAP.items()} }

You MUST respond with a raw JSON object containing exactly these keys:
- "task_type": string, must be one of the allowed task types above.
- "task_label": string, the human readable label for the task.
- "keywords": list of strings, 3-5 search keywords to query dataset hub.
- "confidence": float between 0.0 and 1.0.
- "problem_type": string, one of "binary", "multiclass", "regression", or "auto".

For text-classification, set "problem_type" to "auto".
For translation, summarization, text-generation, text2text-generation, tabular-regression, set "problem_type" to "regression".
For token-classification, question-answering, fill-mask, zero-shot-classification, tabular-classification, set "problem_type" to "multiclass".

Example output:
{{
  "task_type": "text-classification",
  "task_label": "Text Classification",
  "keywords": ["reviews", "fake", "spam"],
  "confidence": 0.95,
  "problem_type": "auto"
}}

Problem description: '{description}'
"""
        
        # Format prompt using template if possible
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        try:
            formatted_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted_text = prompt
            
        # Prepare inputs
        try:
            inputs = processor(text=formatted_text, return_tensors="pt")
        except Exception:
            # If processor requires images, pass a dummy black image
            from PIL import Image
            dummy_image = Image.new("RGB", (224, 224), color="black")
            inputs = processor(images=dummy_image, text=formatted_text, return_tensors="pt")
            
        # Move to model's device
        device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.info("Generating task classification with local Gemma model...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            
        # Extract the new output tokens
        in_len = inputs.get("input_ids", inputs.get("pixel_values", [])).shape[-1] if hasattr(inputs.get("input_ids", inputs.get("pixel_values", None)), "shape") else 0
        generated_ids_trimmed = [
            out_ids[in_len:] for out_ids in generated_ids
        ]
        if not generated_ids_trimmed or len(generated_ids_trimmed[0]) == 0:
            generated_ids_trimmed = generated_ids
            
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        logger.debug("Gemma raw output: {}", output_text)
        
        # Extract JSON from output
        json_match = re.search(r"({.*})", output_text, re.DOTALL)
        if json_match:
            output_text = json_match.group(1)
            
        import json
        data = json.loads(output_text)
        task_type = data.get("task_type", DEFAULT_TASK)
        if task_type not in TASK_MAP:
            task_type = DEFAULT_TASK
            
        task_meta = TASK_MAP[task_type]
        
        logger.success(
            "Gemma detected task: '{}' → {} (confidence: {})",
            description,
            task_type,
            data.get("confidence", 1.0),
        )
        
        return TaskInfo(
            task_type=task_type,
            task_label=data.get("task_label", task_meta["label"]),
            keywords=data.get("keywords", _extract_search_keywords(description, task_type)),
            confidence=float(data.get("confidence", 0.9)),
            problem_type=data.get("problem_type", TASK_TO_PROBLEM_TYPE.get(task_type, "auto")),
        )
    except Exception as e:
        if force:
            raise
        logger.warning("Gemma task detection failed: {}", e)
        return None


def detect_task(description: str, router: str = "auto") -> TaskInfo:
    """Detect the ML task type from a natural-language description.

    Algorithm:
      1. Try Gemma LLM-based task detection if router is gemma or auto
      2. Try OpenAI LLM-based task detection if requested and OPENAI_API_KEY is present
      3. Normalize input text
      4. Try exact substring match (longest keyword first)
      5. Fuzzy-match each keyword against the input
      6. Fall back to 'text-classification'

    Returns:
        TaskInfo with the detected task type, label, keywords, and confidence.
    """
    logger.info("Detecting task from: '{}' (router='{}')", description, router)
    
    if router != "keyword":
        # Try Gemma first if forced
        if router == "gemma":
            gemma_result = _detect_task_gemma(description, force=True)
            if gemma_result is not None:
                return gemma_result
                
        # Try OpenAI
        force_openai = (router == "openai")
        openai_result = _detect_task_openai(description, force=force_openai)
        if openai_result is not None:
            return openai_result

        # Fallback to Gemma for auto
        if router == "auto":
            gemma_result = _detect_task_gemma(description, force=False)
            if gemma_result is not None:
                return gemma_result

    normalized = _normalize(description)

    # --- Step 1: exact substring match ---
    for keyword in _ALL_KEYWORDS:
        if keyword in normalized:
            task_type = _KEYWORD_INDEX[keyword]
            task_meta = TASK_MAP[task_type]
            problem_type = TASK_TO_PROBLEM_TYPE.get(task_type, "auto")
            logger.success(
                "Exact match: '{}' → {} (confidence: 1.0)",
                keyword,
                task_type,
            )
            return TaskInfo(
                task_type=task_type,
                task_label=task_meta["label"],
                keywords=_extract_search_keywords(description, task_type),
                confidence=1.0,
                problem_type=problem_type,
            )

    # --- Step 2: fuzzy match ---
    best_score = 0.0
    best_task: str | None = None
    best_keyword = ""

    for keyword in _ALL_KEYWORDS:
        for window_size in range(len(keyword.split()), 0, -1):
            words = normalized.split()
            for i in range(len(words) - window_size + 1):
                window = " ".join(words[i : i + window_size])
                ratio = difflib.SequenceMatcher(None, keyword, window).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_task = _KEYWORD_INDEX[keyword]
                    best_keyword = keyword

    if best_task and best_score >= 0.65:
        task_meta = TASK_MAP[best_task]
        confidence = round(best_score, 2)
        problem_type = TASK_TO_PROBLEM_TYPE.get(best_task, "auto")
        logger.info(
            "Fuzzy match: '{}' ≈ '{}' → {} (confidence: {})",
            best_keyword,
            normalized,
            best_task,
            confidence,
        )
        return TaskInfo(
            task_type=best_task,
            task_label=task_meta["label"],
            keywords=_extract_search_keywords(description, best_task),
            confidence=confidence,
            problem_type=problem_type,
        )

    # --- Step 3: fallback ---
    logger.warning(
        "No confident match for '{}'. Falling back to '{}'.",
        description,
        DEFAULT_TASK,
    )
    fallback_meta = TASK_MAP[DEFAULT_TASK]
    return TaskInfo(
        task_type=DEFAULT_TASK,
        task_label=fallback_meta["label"],
        keywords=_extract_search_keywords(description, DEFAULT_TASK),
        confidence=0.3,
        problem_type="auto",
    )


def list_supported_tasks() -> list[dict[str, str]]:
    """Return a list of all supported tasks with their labels."""
    return [
        {"task_type": task, "label": info["label"]}
        for task, info in TASK_MAP.items()
    ]


def get_eval_metric(task_type: str) -> str:
    """Get the recommended eval metric for a task type.

    Mirrors AutoGluon's automatic metric selection logic.
    """
    return TASK_EVAL_METRICS.get(task_type, "accuracy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_search_keywords(description: str, task_type: str) -> list[str]:
    """Build a list of search keywords for HF Hub dataset search.

    Combines the raw description words with task-specific terms.
    """
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "but", "not", "that", "this", "it", "i", "we",
        "you", "he", "she", "they", "my", "your", "build", "create",
        "make", "train", "detect", "find", "ai", "model", "ml",
    }
    words = _normalize(description).split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    task_search_terms = {
        "text-classification": ["sentiment", "classification"],
        "token-classification": ["ner", "entities"],
        "question-answering": ["qa", "question"],
        "summarization": ["summarization", "summary"],
        "translation": ["translation", "parallel"],
        "text-generation": ["text", "generation"],
        "fill-mask": ["masked", "language"],
        "text2text-generation": ["text2text", "seq2seq"],
        "zero-shot-classification": ["nli", "zero-shot"],
        "image-classification": ["image", "classification"],
        "tabular-classification": ["tabular", "classification"],
        "tabular-regression": ["tabular", "regression"],
    }

    for term in task_search_terms.get(task_type, []):
        if term not in keywords:
            keywords.append(term)

    return keywords
