"""Unified model clients with normalized token/cost accounting."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any


# USD per million tokens, captured 2026-07-16 from provider pricing pages.
PRICES = {
    "openai:gpt-4o": (2.50, 10.00),
    "openai:gpt-4o-mini": (0.15, 0.60),
    "gemini:gemini-2.5-flash": (0.30, 2.50),
}


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False
    errors: int = 0

    def add(self, input_tokens: int, output_tokens: int, estimated: bool = False) -> None:
        self.calls += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.estimated = self.estimated or estimated


class ModelClient:
    def __init__(self, spec: str) -> None:
        try:
            self.provider, self.model = spec.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"model must be provider:model, got {spec!r}") from exc
        self.spec = spec
        self.usage = Usage()
        if self.provider == "gemini":
            from google import genai
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is required")
            self.client = genai.Client(api_key=key)
        elif self.provider == "openai":
            from openai import OpenAI
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required")
            self.client = OpenAI()
        elif self.provider == "bedrock":
            import boto3
            self.client = boto3.client("bedrock-runtime")
        else:
            raise ValueError(f"unsupported provider {self.provider!r}")

    def generate(self, prompt: str, *, json_schema: dict[str, Any] | None = None) -> str:
        for attempt in range(3):
            try:
                if self.provider == "gemini":
                    return self._gemini(prompt, json_schema)
                if self.provider == "openai":
                    return self._openai(prompt, json_schema)
                return self._bedrock(prompt)
            except Exception:
                self.usage.errors += 1
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise AssertionError("unreachable")

    def _gemini(self, prompt: str, schema: dict[str, Any] | None) -> str:
        from google.genai import types
        config: dict[str, Any] = {
            "temperature": 0.0,
            "max_output_tokens": 200,
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
        }
        if schema:
            config.update(response_mime_type="application/json", response_schema=schema)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config),
        )
        meta = response.usage_metadata
        self.usage.add(
            getattr(meta, "prompt_token_count", 0),
            getattr(meta, "candidates_token_count", 0) + getattr(meta, "thoughts_token_count", 0),
        )
        return (response.text or "").strip()

    def _openai(self, prompt: str, schema: dict[str, Any] | None) -> str:
        kwargs: dict[str, Any] = {}
        if schema:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=200, **kwargs,
        )
        self.usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
        return (response.choices[0].message.content or "").strip()

    def _bedrock(self, prompt: str) -> str:
        model_id = os.environ.get("BEDROCK_OPUS_MODEL_ID")
        if not model_id:
            raise RuntimeError(
                "bedrock:opus-4-8 requires BEDROCK_OPUS_MODEL_ID (the regional "
                "Bedrock inference profile/model ID available to your AWS account)"
            )
        response = self.client.converse(
            modelId=model_id, messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 200},
        )
        usage = response.get("usage", {})
        self.usage.add(usage.get("inputTokens", 0), usage.get("outputTokens", 0))
        return response["output"]["message"]["content"][0]["text"].strip()

    def cost(self) -> dict[str, Any]:
        rates = PRICES.get(self.spec)
        usd = None
        if rates:
            usd = (self.usage.input_tokens * rates[0] + self.usage.output_tokens * rates[1]) / 1_000_000
        return {**self.usage.__dict__, "usd": usd, "model": self.spec, "rates_per_million": rates}


def estimate_tokens(text: str) -> int:
    """Conservative provider-neutral estimate used only for the preflight quote."""
    return max(1, (len(text) + 3) // 4)
