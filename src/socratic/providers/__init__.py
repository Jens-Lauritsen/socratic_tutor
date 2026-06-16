"""LLM provider abstraction layer.

The system MUST NOT be hard‑coded to one AI provider.  Every provider
implements the same ``LLMProvider`` interface so the tutor engine can
swap backends without changing any teaching logic.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx

from socratic.models import ProviderConfig


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract interface that every LLM backend must implement."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._model = config.model or self.default_model()

    @staticmethod
    @abstractmethod
    def default_model() -> str:
        """Return the default model name for this provider."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a prompt and return the model's response text."""
        ...

    def _resolve_temperature(self, temperature: float | None) -> float:
        return temperature if temperature is not None else self.config.temperature

    def _resolve_max_tokens(self, max_tokens: int | None) -> int:
        return max_tokens if max_tokens is not None else self.config.max_tokens


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """OpenAI API (GPT-4o, GPT-4o-mini, etc.)."""

    @staticmethod
    def default_model() -> str:
        return "gpt-4o-mini"

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        from openai import AsyncOpenAI

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var.")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.base_url or None,
        )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._resolve_temperature(temperature),
            max_tokens=self._resolve_max_tokens(max_tokens),
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Anthropic API (Claude models)."""

    @staticmethod
    def default_model() -> str:
        return "claude-3-5-haiku-latest"

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        from anthropic import AsyncAnthropic

        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY env var."
            )

        client = AsyncAnthropic(api_key=api_key)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._resolve_max_tokens(max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = await client.messages.create(**kwargs)
        # Anthropic returns a list of content blocks
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""


# ---------------------------------------------------------------------------
# Ollama provider (local models)
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Ollama – local LLM server (default http://localhost:11434)."""

    @staticmethod
    def default_model() -> str:
        return "llama3.2"

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        base = self.config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        url = f"{base.rstrip('/')}/api/generate"

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._resolve_temperature(temperature),
                "num_predict": self._resolve_max_tokens(max_tokens),
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")


# ---------------------------------------------------------------------------
# Utility: strip chain-of-thought reasoning from model output
# ---------------------------------------------------------------------------

# Patterns that indicate the start of reasoning/thinking blocks
_REASONING_STARTS = [
    "Thinking.",
    "Let me think",
    "Let me analyze",
    "Let me break",
    "I need to",
    "First, I",
    "We are asked",
    "We are given",
    "The user wants",
    "Okay, so",
    "Alright,",
]

# Patterns that mark the transition from reasoning to the actual answer
_ANSWER_MARKERS = [
    "\n\n**Answer",
    "\n\nAnswer:",
    "\n\nMy response",
    "\n\nHere",
    "\n\nThe best",
    "\n\nSo my",
    "\n\nTutor:",
    "\n\nStudent:",
    '\n\n{"title"',  # JSON exercise
]


def _strip_reasoning(text: str) -> str:
    """Remove chain-of-thought reasoning from visible model output.

    Some reasoning models (DeepSeek, o1, etc.) output their internal
    deliberation before the actual response.  This function detects and
    removes that prefix so the user only sees the tutor's answer.
    """
    if not text:
        return text

    # If the text doesn't start like reasoning, return as-is
    starts_with_reasoning = any(
        text.lstrip().startswith(prefix) for prefix in _REASONING_STARTS
    )
    if not starts_with_reasoning:
        return text

    # Look for a transition marker that signals the end of reasoning
    for marker in _ANSWER_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            # Return everything from the marker onward
            return text[idx:].lstrip()

    # If we can't find a clear transition, try to find the last
    # substantial paragraph (one that looks like a tutor response)
    paragraphs = text.split("\n\n")
    # Reasoning paragraphs often start with numbers, bullets, or analysis
    # The actual answer is usually the last substantial paragraph
    for i in range(len(paragraphs) - 1, -1, -1):
        p = paragraphs[i].strip()
        if p and not p[0].isdigit() and not p.startswith(("*", "-", "1.", "2.")):
            if len(p) > 30:  # Must be substantial
                return p

    return text


# ---------------------------------------------------------------------------
# OpenCode provider (OpenAI-compatible gateway)
# ---------------------------------------------------------------------------

class OpenCodeProvider(LLMProvider):
    """OpenCode API – OpenAI-compatible gateway to many models.

    OpenCode provides access to GPT, Claude, Gemini, DeepSeek, and other
    models through a single API key.  The backend is OpenAI-compatible.

    Environment variables
    ---------------------
    OPENCODE_API_KEY : str
        Your OpenCode API key (sk-...).
    OPENCODE_BASE_URL : str, optional
        Override the default base URL.
    """

    # Free model that works without a paid plan.
    # Paid users can override via --model or OPENCODE_MODEL env var.
    @staticmethod
    def default_model() -> str:
        return os.getenv("OPENCODE_MODEL", "deepseek-v4-flash-free")

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        from openai import AsyncOpenAI

        api_key = self.config.api_key or os.getenv("OPENCODE_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenCode API key not provided. Set OPENCODE_API_KEY env var."
            )

        base_url = (
            self.config.base_url
            or os.getenv("OPENCODE_BASE_URL")
            or "https://opencode.ai/zen/v1"
        )

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Reasoning models (like DeepSeek) need extra tokens for their
        # internal chain-of-thought — otherwise the visible response may
        # be empty or truncated mid-thought.
        resolved_max = self._resolve_max_tokens(max_tokens)
        if resolved_max < 4096:
            resolved_max = 4096  # Reasoning models need headroom

        response = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._resolve_temperature(temperature),
            max_tokens=resolved_max,
        )
        content = response.choices[0].message.content or ""
        # Some reasoning models return the real answer in reasoning_content
        # when content is empty (e.g. when max_tokens is too small).
        if not content and hasattr(response.choices[0].message, "reasoning_content"):
            content = response.choices[0].message.reasoning_content or ""

        # Strip chain-of-thought reasoning that spills into the visible
        # response (common with DeepSeek models).  The actual answer
        # typically follows after the thinking block.
        content = _strip_reasoning(content)
        return content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_CLASSES: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "opencode": OpenCodeProvider,
}


def create_provider(config: ProviderConfig) -> LLMProvider:
    """Instantiate the correct LLMProvider subclass from config."""
    cls = _PROVIDER_CLASSES.get(config.provider.lower())
    if cls is None:
        raise ValueError(
            f"Unknown provider '{config.provider}'. "
            f"Available: {', '.join(_PROVIDER_CLASSES)}"
        )
    return cls(config)


def available_providers() -> list[str]:
    return list(_PROVIDER_CLASSES.keys())
