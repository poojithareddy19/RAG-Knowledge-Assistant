"""LLM provider abstraction.

One interface, three backends selected by config/env:

  * openai  — OpenAI Chat Completions (also works for OpenAI-compatible APIs)
  * gemini  — Google Generative AI
  * ollama  — local models via the Ollama HTTP API (no API key)

Each provider returns a normalized ``LLMResponse`` with text and token usage so
the rest of the system never branches on provider. Adding a provider is a new
subclass + one line in ``get_llm``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.config import get_config


@dataclass
class LLMResponse:
    text: str
    tokens: dict[str, int] = field(default_factory=dict)


class BaseLLM:
    def __init__(self) -> None:
        cfg = get_config().generation
        self.model = cfg.model
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_tokens = int(cfg.get("max_tokens", 700))

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:  # noqa: D401
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        from openai import OpenAI

        api_key = (get_config().secrets.openai_api_key or "").strip()
        if not api_key or api_key in {"...", "your_api_key"}:
            raise ValueError(
                "OPENAI_API_KEY is missing. Set it in .env when using GENERATION__PROVIDER=openai."
            )
        if not api_key.startswith("sk-"):
            raise ValueError(
                "OPENAI_API_KEY format is invalid. It should start with 'sk-'."
            )

        self.client = OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            tokens={
                "prompt": getattr(usage, "prompt_tokens", 0),
                "completion": getattr(usage, "completion_tokens", 0),
                "total": getattr(usage, "total_tokens", 0),
            },
        )


class GeminiLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        import google.generativeai as genai

        genai.configure(api_key=get_config().secrets.google_api_key)
        self._genai = genai

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        model = self._genai.GenerativeModel(
            self.model, system_instruction=system_prompt
        )
        resp = model.generate_content(
            user_prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )
        usage = getattr(resp, "usage_metadata", None)
        tokens = {}
        if usage is not None:
            tokens = {
                "prompt": getattr(usage, "prompt_token_count", 0),
                "completion": getattr(usage, "candidates_token_count", 0),
                "total": getattr(usage, "total_token_count", 0),
            }
        return LLMResponse(text=resp.text or "", tokens=tokens)


class OllamaLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        self.base_url = get_config().secrets.ollama_base_url.rstrip("/")

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        import requests

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": self.temperature},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            tokens={
                "prompt": data.get("prompt_eval_count", 0),
                "completion": data.get("eval_count", 0),
                "total": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
        )


_PROVIDERS = {"openai": OpenAILLM, "gemini": GeminiLLM, "ollama": OllamaLLM}


def get_llm() -> BaseLLM:
    provider = get_config().generation.provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Options: {list(_PROVIDERS)}"
        )
    return _PROVIDERS[provider]()
