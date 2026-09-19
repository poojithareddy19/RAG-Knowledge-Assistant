"""LLM provider abstraction.

One backend today: ollama, local models over the Ollama HTTP API, no API key.

There were OpenAI and Gemini subclasses here too. They were deleted rather than
kept warm, because neither had ever been run: no key was ever configured, the
SQL generator does not go through this abstraction at all, and an untested
provider is a claim of portability rather than portability. The shape is still
here, so adding one back is a subclass and one line in ``_PROVIDERS``, but it
should arrive with something that exercises it.

The provider returns a normalized ``LLMResponse`` with text and token usage so
the rest of the system never branches on provider.
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


_PROVIDERS = {"ollama": OllamaLLM}


def get_llm() -> BaseLLM:
    provider = get_config().generation.provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Options: {list(_PROVIDERS)}"
        )
    return _PROVIDERS[provider]()
