"""AutoHF — Gemma local chat agent.

Connected to google/gemma-4-E2B-it (or user-specified model) for local chat.
"""

from __future__ import annotations

import os
import re
from typing import Optional, List, Dict
from loguru import logger


class GemmaChatAgent:
    """Agent that runs local Gemma chat/generation.

    Loads google/gemma-4-E2B-it locally using HF transformers.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-E2B-it",
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.processor = None
        self.model = None

    def load(self) -> None:
        """Load the model and processor from Hugging Face."""
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "HF_TOKEN environment variable is required to load the Gemma model."
            )

        logger.info("Loading local Gemma model '{}'...", self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=token)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            token=token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        logger.success("Model '{}' loaded successfully.", self.model_id)

    def generate(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """Generate response given a user prompt and previous conversation history."""
        import torch

        if not self.model or not self.processor:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Re-construct messages format for chat template
        messages = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

        for turn in history:
            messages.append(
                {
                    "role": turn["role"],
                    "content": [{"type": "text", "text": turn["content"]}],
                }
            )

        messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )

        try:
            formatted_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted_text = prompt

        try:
            inputs = self.processor(text=formatted_text, return_tensors="pt")
        except Exception:
            # If processor requires images (multimodal model), pass a dummy black image
            from PIL import Image

            dummy_image = Image.new("RGB", (224, 224), color="black")
            inputs = self.processor(
                images=dummy_image, text=formatted_text, return_tensors="pt"
            )

        device = getattr(
            self.model,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)

        # Trim input tokens
        in_len = (
            inputs.get("input_ids", inputs.get("pixel_values", [])).shape[-1]
            if hasattr(
                inputs.get("input_ids", inputs.get("pixel_values", None)),
                "shape",
            )
            else 0
        )
        generated_ids_trimmed = [out_ids[in_len:] for out_ids in generated_ids]
        if not generated_ids_trimmed or len(generated_ids_trimmed[0]) == 0:
            generated_ids_trimmed = generated_ids

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()
