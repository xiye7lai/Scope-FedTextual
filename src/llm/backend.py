from __future__ import annotations

import json
import os
import time
import threading
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .cache import DiskCache


class LLMCallBudgetExceeded(RuntimeError):
    """Raised before an uncached call would exceed the configured hard budget."""


class LLMBackend(ABC):
    def __init__(self, model: str, cache_dir: str | Path, max_llm_calls: int | None = None):
        self.model = model
        self.cache = DiskCache(cache_dir)
        self.max_llm_calls = None if max_llm_calls is None else int(max_llm_calls)
        if self.max_llm_calls is not None and self.max_llm_calls < 0:
            raise ValueError("max_llm_calls must be non-negative or null")
        self.stats = {"requests": 0, "api_calls": 0, "api_attempts": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}
        self.cache_identity: dict[str, Any] = {}
        self._stats_lock = threading.Lock()
        self._uncached_inflight = 0

    def generate(self, prompt: str, purpose: str = "general", metadata: dict | None = None) -> str:
        metadata = metadata or {}
        payload = {
            "backend": type(self).__name__, "model": self.model, "prompt": prompt,
            "purpose": purpose, "metadata": metadata, "decoding": self.cache_identity,
        }
        with self._stats_lock:
            self.stats["requests"] += 1
        def call_with_budget():
            with self._stats_lock:
                if self.max_llm_calls is not None and self.stats["api_calls"] + self._uncached_inflight >= self.max_llm_calls:
                    raise LLMCallBudgetExceeded(
                        f"Uncached LLM call budget exhausted ({self.max_llm_calls}); cached calls remain usable"
                    )
                self._uncached_inflight += 1
            try:
                value = self._call(prompt, purpose, metadata)
            except Exception:
                with self._stats_lock:
                    self._uncached_inflight -= 1
                raise
            with self._stats_lock:
                self._uncached_inflight -= 1
                self.stats["api_calls"] += 1
            return value

        result = self.cache.get_or_call(payload, call_with_budget)
        if result.get("cached"):
            with self._stats_lock:
                self.stats["cache_hits"] += 1
        usage = result.get("usage", {})
        with self._stats_lock:
            self.stats["input_tokens"] += int(usage.get("input_tokens", max(1, len(prompt) // 4)))
            self.stats["output_tokens"] += int(usage.get("output_tokens", max(1, len(result.get("text", "")) // 4)))
        return str(result["text"])

    @abstractmethod
    def _call(self, prompt: str, purpose: str, metadata: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIResponsesBackend(LLMBackend):
    def __init__(self, cfg: dict, cache_dir: str | Path):
        model = cfg.get("model") or os.environ.get("OPENAI_MODEL")
        if not model:
            raise RuntimeError("Set llm.model or OPENAI_MODEL for the Responses backend")
        super().__init__(
            model,
            cache_dir,
            cfg.get("max_llm_calls"),
        )
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for the real backend")
        self.api_key = key
        base = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.url = cfg.get("responses_url") or base.rstrip("/") + "/responses"
        self.max_output_tokens = int(cfg.get("max_output_tokens", 700))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.timeout = int(cfg.get("timeout_seconds", 120))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.cache_identity = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }

    def _call(self, prompt: str, purpose: str, metadata: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
        }
        # Some Responses-compatible gateways reject temperature for reasoning models.
        if self.temperature != 0:
            body["temperature"] = self.temperature
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        retryable_codes = {408, 409, 429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            self.stats["api_attempts"] += 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code not in retryable_codes or attempt >= self.max_retries:
                    raise RuntimeError(f"Responses API HTTP {exc.code}: {detail[:1000]}") from exc
            except urllib.error.URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Responses API connection error: {exc}") from exc
            time.sleep(min(2 ** attempt, 8))
        text = data.get("output_text")
        if not text:
            chunks = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"}:
                        chunks.append(content.get("text", ""))
            text = "\n".join(chunks)
        if not text:
            raise RuntimeError(f"Responses API returned no text: {str(data)[:1000]}")
        usage = data.get("usage", {})
        return {"text": text, "usage": {
            "input_tokens": usage.get("input_tokens", max(1, len(prompt) // 4)),
            "output_tokens": usage.get("output_tokens", max(1, len(text) // 4)),
        }}


class VLLMOpenAIBackend(LLMBackend):
    """OpenAI Chat Completions client for a local vLLM server."""

    def __init__(self, cfg: dict, cache_dir: str | Path):
        model = cfg.get("model_name") or cfg.get("model")
        if not model:
            raise ValueError("llm.model_name (or llm.model) is required for vllm_openai")
        super().__init__(model, cache_dir, cfg.get("max_llm_calls"))
        base = cfg.get("base_url", "http://127.0.0.1:8000/v1")
        self.url = cfg.get("chat_completions_url") or base.rstrip("/") + "/chat/completions"
        key_env = cfg.get("api_key_env", "LOCAL_LLM_API_KEY")
        self.api_key = os.environ.get(key_env) or "local-vllm"
        self.temperature = float(cfg.get("temperature", 0.0))
        self.top_p = float(cfg.get("top_p", 1.0))
        self.top_k = int(cfg.get("top_k", -1))
        self.max_tokens = int(cfg.get("max_new_tokens", cfg.get("max_output_tokens", 512)))
        self.seed = int(cfg.get("seed", 13))
        self.timeout = int(cfg.get("timeout_seconds", 180))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.enable_thinking = bool(cfg.get("enable_thinking", False))
        self.extra_body = dict(cfg.get("extra_body", {}))
        chat_kwargs = dict(self.extra_body.get("chat_template_kwargs", {}))
        chat_kwargs["enable_thinking"] = self.enable_thinking
        self.extra_body["chat_template_kwargs"] = chat_kwargs
        self.cache_identity = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "enable_thinking": self.enable_thinking,
            "extra_body": self.extra_body,
        }

    def _call(self, prompt: str, purpose: str, metadata: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            **self.extra_body,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        retryable_codes = {408, 409, 429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            self.stats["api_attempts"] += 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code not in retryable_codes or attempt >= self.max_retries:
                    raise RuntimeError(f"vLLM Chat API HTTP {exc.code}: {detail[:1000]}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"vLLM Chat API connection error at {self.url}: {exc}") from exc
            time.sleep(min(2 ** attempt, 8))
        choices = data.get("choices", [])
        text = choices[0].get("message", {}).get("content") if choices else None
        if text is None:
            raise RuntimeError(f"vLLM Chat API returned no assistant content: {str(data)[:1000]}")
        usage = data.get("usage", {})
        return {"text": text, "usage": {
            "input_tokens": usage.get("prompt_tokens", max(1, len(prompt) // 4)),
            "output_tokens": usage.get("completion_tokens", max(1, len(text) // 4)),
        }}


def build_backend(cfg: dict, run_dir: str | Path) -> LLMBackend:
    backend_name = "mock" if cfg.get("_dry_run") else cfg.get("provider", cfg.get("backend", "mock"))
    shared_cache = cfg.get("shared_cache_dir")
    cache_dir = Path(shared_cache) if shared_cache else Path(run_dir) / cfg.get("cache_dir", "llm_cache")
    if backend_name == "mock":
        from .mock_backend import MockBackend
        return MockBackend(cfg.get("model", "mock-textgrad-v1"), cache_dir, cfg.get("max_llm_calls"))
    if backend_name == "openai_responses":
        return OpenAIResponsesBackend(cfg, cache_dir)
    if backend_name == "vllm_openai":
        return VLLMOpenAIBackend(cfg, cache_dir)
    raise ValueError(f"Unknown LLM backend: {backend_name}")
