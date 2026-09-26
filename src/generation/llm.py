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

from src.monitoring.tracing import llm_span, record_ollama
from src.utils.config import get_config

DEFAULT_NUM_CTX = 6144
DEFAULT_KEEP_ALIVE = "30m"


def num_ctx() -> int:
    """The context window sent with every Ollama call.

    One value for every call site, not one per call: Ollama reloads the model
    whenever the requested window changes, which on this hardware costs more
    than the call itself.
    """
    try:
        return int((get_config().get("ollama", {}) or {}).get("num_ctx", DEFAULT_NUM_CTX))
    except Exception:
        return DEFAULT_NUM_CTX


def keep_alive() -> str | int:
    """How long Ollama holds the model after a call, sent with every call.

    Per request rather than as an Ollama setting, so it travels with the
    repository instead of living in one machine's environment.
    """
    try:
        value = (get_config().get("ollama", {}) or {}).get("keep_alive", DEFAULT_KEEP_ALIVE)
    except Exception:
        value = DEFAULT_KEEP_ALIVE

    return value if isinstance(value, int) else str(value)


def warm_up(models: list[str]) -> dict[str, float]:
    """Load each model into Ollama, returning how long each took in seconds.

    A request with no prompt loads the model and generates nothing. It sends
    the same num_ctx as a real call, because a load at a different window is
    thrown away by the first question and loaded again.
    """
    import time

    import requests

    base = get_config().secrets.ollama_base_url.rstrip("/")
    took = {}

    for model in models:
        started = time.perf_counter()

        with llm_span("warm_up", model) as span:
            resp = requests.post(
                f"{base}/api/generate",
                json={
                    "model": model,
                    "keep_alive": keep_alive(),
                    "options": {"num_ctx": num_ctx()},
                },
                timeout=300,
            )
            resp.raise_for_status()
            record_ollama(span, resp.json())

        took[model] = round(time.perf_counter() - started, 1)

    return took


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

        with llm_span(
            "answer",
            self.model,
            operation="chat",
            temperature=self.temperature,
        ) as span:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "keep_alive": keep_alive(),
                    "options": {
                        "temperature": self.temperature,
                        "num_ctx": num_ctx(),
                    },
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            record_ollama(span, data, prompt=f"{system_prompt}\n\n{user_prompt}")

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
