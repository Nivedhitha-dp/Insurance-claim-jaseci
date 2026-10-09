"""Provider-agnostic LLM access.

Groq, Gemini, OpenRouter, Ollama, OpenAI and Anthropic all expose OpenAI-compatible chat completions with
tool calling, so one client class covers them. `LLMChain` adds failover: if the primary provider is rate
limited, errors, or rejects a request as too large, the *same* conversation continues on the next provider
(messages are plain OpenAI format, so they transfer as-is). The agent never notices except via an event.
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .config import settings

# provider -> (base_url, api-key env var, default agent model, default vision model)
PROVIDERS: dict[str, tuple[Optional[str], Optional[str], str, Optional[str]]] = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY",
               "gemini-2.5-flash", "gemini-2.5-flash"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "qwen/qwen3-235b-a22b", None),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "claude-sonnet-5-5", "claude-sonnet-5-5"),
    "openai": (None, "OPENAI_API_KEY", "gpt-4.1-mini", "gpt-4.1-mini"),
    "ollama": ("http://localhost:11434/v1", None, "qwen2.5:7b", None),
}


class LLMUnavailable(RuntimeError):
    pass


class ProviderFailed(RuntimeError):
    """This provider can't serve the request right now - the chain should try the next one."""


# Free-tier requests-per-minute we pace ourselves to (override with <PROVIDER>_RPM). Pacing client-side is
# far cheaper than hitting 429s: Gemini's free tier rejects bursts within the same minute.
DEFAULT_RPM = {"gemini": 8, "groq": 25, "openrouter": 15, "anthropic": 40, "openai": 60, "ollama": 1000}
_last_call: dict[str, float] = {}
MAX_RATE_LIMIT_WAIT = float(os.getenv("MAX_RATE_LIMIT_WAIT", "30"))  # longer waits -> fail over instead


def _pace(provider: str, model: str) -> None:
    """Quotas are metered per model, so each provider+model gets its own request spacing."""
    rpm = float(os.getenv(f"{provider.upper()}_RPM", DEFAULT_RPM.get(provider, 30)))
    key = f"{provider}:{model}"
    wait = _last_call.get(key, 0) + 60.0 / rpm - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[key] = time.monotonic()


def model_for(provider: str) -> str:
    return os.getenv(f"{provider.upper()}_MODEL") or PROVIDERS[provider][2]


def vision_model_for(provider: str) -> Optional[str]:
    v = os.getenv(f"{provider.upper()}_VISION_MODEL")
    return (v or None) if v is not None else PROVIDERS[provider][3]


class LLM:
    def __init__(self, provider: str, model: Optional[str] = None):
        from openai import OpenAI

        if provider not in PROVIDERS:
            raise LLMUnavailable(f"unknown provider {provider}")
        base_url, key_env, _, _ = PROVIDERS[provider]
        key = os.getenv(key_env) if key_env else "ollama"
        if not key:
            raise LLMUnavailable(f"{key_env} is not set")
        self.provider = provider
        self.model = model or model_for(provider)
        self.vision_model = vision_model_for(provider)
        self.client = OpenAI(api_key=key, base_url=base_url, max_retries=0,
                             timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", "45")))

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None, model: Optional[str] = None,
             patient: bool = True) -> Any:
        """One completion with retries for transient problems. patient=False (a fallback exists) gives up
        quickly on rate limits instead of sleeping, so the chain can switch provider."""
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        if self.provider != "gemini":  # Gemini's thought signatures mean nothing to (and may break) others
            messages = _without_provider_extras(messages)
        kwargs: dict[str, Any] = {"model": model or self.model, "messages": messages,
                                  "temperature": settings.temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        delay = 2.0
        for attempt in range(6):
            try:
                _pace(self.provider, kwargs["model"])
                return self.client.chat.completions.create(**kwargs).choices[0].message
            except RateLimitError as e:
                wait = _retry_after(e) or delay
                # Short per-minute limits: wait them out. Long ones (daily quota): fail over. The last provider
                # in the chain has nowhere to fail over to, so it waits out anything up to ~a minute and a half.
                limit = MAX_RATE_LIMIT_WAIT if not patient else max(MAX_RATE_LIMIT_WAIT, 90)
                if wait > limit or attempt >= (5 if patient else 2):
                    raise ProviderFailed(f"{self.provider} rate-limited (retry after ~{wait:.0f}s)") from e
                time.sleep(wait + 0.5)
                delay *= 2
            except APIStatusError as e:
                body = str(e)
                if "tool_use_failed" in body or "Failed to call a function" in body:
                    kwargs["messages"] = messages + [{"role": "user", "content":
                                                      "Your last tool call was malformed. Call the tool again with valid JSON arguments."}]
                    continue
                if e.status_code == 413 or "too large" in body.lower() or "context" in body.lower() and "length" in body.lower():
                    raise ProviderFailed(f"{self.provider}: request too large for this model/tier") from e
                if e.status_code >= 500 and attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise ProviderFailed(f"{self.provider} error {e.status_code}: {body[:160]}") from e
            except (APIConnectionError, APITimeoutError) as e:
                if attempt >= 1:
                    raise ProviderFailed(f"{self.provider} unreachable") from e
                time.sleep(delay)
                delay *= 2
        raise ProviderFailed(f"{self.provider}: retries exhausted")

    def vision(self, image: Path, question: str) -> str:
        if not self.vision_model:
            raise LLMUnavailable(f"{self.provider} has no vision model configured")
        b64 = base64.b64encode(image.read_bytes()).decode()
        mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
        msg = self.chat([{"role": "user", "content": [
            {"type": "text", "text": "You are reading a scanned insurance document. Transcribe exactly what is "
                                     "written that answers the question; if something is crossed out, say what "
                                     "was crossed out and what replaced it. If any character is unclear (e.g. 2 vs "
                                     "Z, 5 vs S, 0 vs O), say so explicitly. If unreadable, say so. Question: " + question},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
            model=self.vision_model)
        return (msg.content or "").strip()


class LLMChain:
    """Primary provider + ordered fallbacks, same interface as LLM."""

    def __init__(self, llms: list[LLM]):
        self.llms = llms
        self.active = 0
        self.switches: list[str] = []
        self.on_switch: Callable[[str], None] = lambda msg: None

    @property
    def provider(self) -> str:
        return self.llms[self.active].provider

    @property
    def model(self) -> str:
        return self.llms[self.active].model

    def describe(self) -> str:
        return " → ".join(f"{l.provider}/{l.model}" for l in self.llms)

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None) -> Any:
        while True:
            llm = self.llms[self.active]
            last = self.active == len(self.llms) - 1
            try:
                return llm.chat(messages, tools, patient=last)
            except ProviderFailed as e:
                if last:
                    raise
                self.active += 1
                self.switches.append(f"{llm.provider}/{llm.model}: {str(e)[:80]}")
                self.on_switch(f"{e} - switching to {self.llms[self.active].provider}/{self.llms[self.active].model}")

    def vision(self, image: Path, question: str) -> str:
        errors = []
        for llm in self.llms:  # best document reader first (the primary), others as backup
            if not llm.vision_model:
                continue
            try:
                return f"[{llm.provider}/{llm.vision_model}] " + llm.vision(image, question)
            except Exception as e:  # noqa: BLE001 - try the next provider
                errors.append(f"{llm.provider}: {str(e)[:80]}")
        raise LLMUnavailable("no vision model could read the image: " + "; ".join(errors))


def _without_provider_extras(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [{k: v for k, v in c.items() if k != "extra_content"} for c in m["tool_calls"]]}
        out.append(m)
    return out


def _retry_after(e) -> Optional[float]:
    """Seconds to wait: the Retry-After header (Groq/OpenAI) or Gemini's retryDelay ('17s') in the body."""
    import re

    try:
        h = e.response.headers.get("retry-after")
        if h:
            return float(h)
    except Exception:
        pass
    m = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s", str(e))
    return float(m.group(1)) if m else None


def get_llm() -> Optional[LLMChain]:
    """Build the chain from LLM_PROVIDER + LLM_FALLBACK. Entries are 'provider' (its default model) or
    'provider:model' - free tiers meter quota per model, so several models of one provider multiply the
    free capacity. Providers without a key are skipped."""
    if settings.provider == "none":
        return None
    entries = [settings.provider] + [p for p in settings.fallback_providers if p != settings.provider]
    llms = []
    for entry in entries:
        provider, _, model = entry.partition(":")
        try:
            llms.append(LLM(provider, model or None))
        except LLMUnavailable:
            continue
    return LLMChain(llms) if llms else None
