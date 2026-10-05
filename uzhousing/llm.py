"""Thin, defensive wrapper around multiple hosted LLM providers.

Five vendors are supported, picked from the model id or an explicit
``provider:`` prefix:

* ``anthropic`` — Claude via the Messages API
* ``openai`` — GPT/o-series via the Responses API
* ``groq`` — any Groq model via the OpenAI-compatible Chat Completions endpoint
* ``gemini`` — Google Gemini via its OpenAI-compatible endpoint
* ``openrouter`` — OpenRouter via its OpenAI-compatible API

A single normalised return shape is produced for every provider, so the rest
of the pipeline is provider-blind. The whole pipeline is designed to work
without any API key: every entry point here either returns a usable value or
raises :class:`LLMUnavailable`, which callers catch to fall back to
deterministic behaviour.

Backward compatibility: ``LLM(api_key, model)`` keeps the previous contract,
and ``provider_for(model)`` still routes ``claude-*`` to Anthropic and
``gpt-*`` / ``o*`` to OpenAI. New callers can either use a ``provider:model``
prefix (for example ``"groq:openai/gpt-oss-120b"``) or build via
``LLM.from_settings(settings)``, which also wires up the free-tier fallback
chain described in the project README.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)

# The current models support a 128k output ceiling, but every token of it is
# streamed and billed, so the pipeline caps itself well below that.
MAX_OUTPUT_TOKENS = 64_000

# Adaptive thinking is on by default on the current model generation and its
# tokens count against ``max_tokens``. A budget sized for the JSON alone is
# therefore spent on the reasoning, and the object is cut off before it closes —
# which surfaces as "model did not return parseable JSON" and silently drops the
# pipeline back to its deterministic fallbacks. Every structured call gets
# headroom for the thinking that precedes the answer.
JSON_MIN_TOKENS = 8_000


# Default model and base URL per provider. Base URLs point at each vendor's
# OpenAI-compatible endpoint; the Anthropic SDK and OpenAI's native Responses
# API do not use ``base_url`` from this table.
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic":  {"model": "claude-opus-5"},
    "openai":     {"model": "gpt-5"},
    "groq":       {"model": "openai/gpt-oss-120b",
                   "base_url": "https://api.groq.com/openai/v1"},
    "gemini":     {"model": "gemini-2.5-flash",
                   "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
    "openrouter": {"model": "openrouter/auto",
                   "base_url": "https://openrouter.ai/api/v1"},
}

# Which env var carries each provider's key.
ENV_KEY_FOR_PROVIDER: dict[str, str] = {
    "anthropic":  "ANTHROPIC_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "groq":       "GROQ_API_KEY",
    "gemini":     "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Human-friendly label per provider, used in run logs and the narrative's
# ``generated_by`` field.
PROVIDER_DISPLAY_NAME: dict[str, str] = {
    "anthropic":  "Claude",
    "openai":     "OpenAI",
    "groq":       "Groq",
    "gemini":     "Gemini",
    "openrouter": "OpenRouter",
}

# Preferred fallback order from most to least preferred FREE provider, with
# the two established paid providers at the end.
FALLBACK_ORDER: tuple[str, ...] = ("groq", "gemini", "openrouter", "openai", "anthropic")

# Known provider prefixes (case-insensitive) in model ids like
# ``"groq:llama3-8b"``.
_PROVIDER_PREFIXES = tuple(ENV_KEY_FOR_PROVIDER.keys())

# Legacy prefix routing: when no explicit ``provider:`` prefix is given, map
# bare model-id prefixes to the two providers the project started with.
_ANTHROPIC_PREFIXES = ("claude", "anthropic.")
_OPENAI_PREFIXES = ("gpt", "chatgpt", "o1", "o3", "o4", "o5")


def parse_model_id(spec: str) -> tuple[str | None, str]:
    """Split a ``provider:model`` spec into its two parts.

    The model keeps every colon it had after the first one, so an id such as
    ``"groq:openai/gpt-oss-120b"`` stays intact (only the leading provider
    tag is stripped). Returns ``(None, spec)`` when no known provider tag is
    present.
    """
    spec = (spec or "").strip()
    if ":" in spec:
        head, tail = spec.split(":", 1)
        if head.lower() in _PROVIDER_PREFIXES:
            return head.lower(), tail.strip()
    return None, spec


def provider_for(model: str) -> str:
    """Return the provider name for a model id.

    An explicit ``provider:model`` prefix wins; otherwise the bare model name
    is matched against the legacy prefix tables so existing ``claude-*`` and
    ``gpt-*`` ids still route correctly. Unknown ids default to Anthropic,
    which is what every id in this project was before OpenAI was added.
    """
    prefix, name = parse_model_id(model)
    if prefix:
        return prefix
    low = (name or "").strip().lower()
    if low.startswith(_ANTHROPIC_PREFIXES):
        return "anthropic"
    if low.startswith(_OPENAI_PREFIXES):
        return "openai"
    return "anthropic"


class LLMUnavailable(RuntimeError):
    """Raised when no LLM answer could be obtained."""


class _ProviderError(RuntimeError):
    """Internal: a provider call that could not be completed.

    ``retryable`` says whether a different provider (or the same one on
    another attempt) might succeed; it is the signal the fallback loop uses
    to decide whether to try the next provider.
    """

    def __init__(self, message: str, *, retryable: bool = False,
                 retry_after: float | None = None,
                 status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
def _retry_after_from_headers(exc: BaseException) -> float | None:
    try:
        headers = dict(getattr(getattr(exc, "response", None), "headers", {}) or {})
    except Exception:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _retryable_openai(exc: BaseException) -> tuple[bool, float | None]:
    """Classify an exception raised by the ``openai`` SDK."""
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a hard dependency
        return False, None

    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status in (408, 409, 425, 429, 500, 502, 503, 504, 529):
        return True, _retry_after_from_headers(exc)
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True, None
    # These are deterministic programmer / credential errors — never retry
    # them and never fall back to another provider on their account.
    non_retryable = (
        openai.AuthenticationError, openai.BadRequestError, openai.NotFoundError,
        openai.PermissionDeniedError, openai.UnprocessableEntityError,
    )
    if isinstance(exc, non_retryable):
        return False, None
    # A generic APIStatusError without a code we recognise: treat as retryable
    # so the fallback chain can try another provider.
    if isinstance(exc, openai.APIStatusError):
        return True, _retry_after_from_headers(exc)
    return False, None


def _retryable_anthropic(exc: BaseException) -> tuple[bool, float | None]:
    """Classify an exception raised by the ``anthropic`` SDK."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return False, None

    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status in (408, 409, 425, 429, 500, 502, 503, 504, 529):
        return True, _retry_after_from_headers(exc)
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True, None
    non_retryable = (
        anthropic.AuthenticationError, anthropic.BadRequestError,
        anthropic.NotFoundError, anthropic.PermissionDeniedError,
    )
    if isinstance(exc, non_retryable):
        return False, None
    if isinstance(exc, anthropic.APIStatusError):
        return True, _retry_after_from_headers(exc)
    return False, None


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------
class _ProviderAdapter:
    """One provider.  Build it, check ``.available``, then call it.

    Subclasses override :meth:`_build_client` to pick the right SDK and
    :meth:`complete` to call the right endpoint. All of them share the same
    normalised ``(text, stop_reason)`` return and ``last_usage`` dict so the
    outer :class:`LLM` need not care which vendor it is talking to.
    """

    def __init__(self, provider: str, api_key: str, model: str,
                 base_url: str | None = None) -> None:
        self.provider = provider
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None
        self._init_error: str | None = None
        self.last_usage: dict[str, int] = {}
        if not api_key:
            self._init_error = (
                f"no {ENV_KEY_FOR_PROVIDER.get(provider, 'API key')} set"
            )
            return
        try:
            self._client = self._build_client()
        except Exception as exc:  # pragma: no cover - import/credential issues
            self._init_error = f"{provider} client unavailable: {exc}"
            LOGGER.warning("LLM provider %s disabled: %s", provider, self._init_error)

    def _build_client(self) -> Any:  # overridden
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, prompt: str, system: str, max_tokens: int
                 ) -> tuple[str, str]:
        raise NotImplementedError

    def classify_error(self, exc: BaseException) -> tuple[bool, float | None]:
        """Return ``(retryable, retry_after_seconds)`` for a vendor error."""
        return False, None


class _AnthropicAdapter(_ProviderAdapter):
    """Claude via the Messages API."""

    def _build_client(self) -> Any:
        import anthropic
        return anthropic.Anthropic(api_key=self._api_key)

    def complete(self, prompt: str, system: str, max_tokens: int
                 ) -> tuple[str, str]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        with self._client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
        text = "".join(
            block.text for block in message.content
            if getattr(block, "type", "") == "text"
        )
        usage = getattr(message, "usage", None)
        self.last_usage = {
            key: int(getattr(usage, key, 0) or 0)
            for key in ("input_tokens", "output_tokens")
        }
        reason = str(getattr(message, "stop_reason", "") or "")
        return text, reason

    def classify_error(self, exc: BaseException) -> tuple[bool, float | None]:
        return _retryable_anthropic(exc)


class _OpenAIResponsesAdapter(_ProviderAdapter):
    """OpenAI via the Responses API.

    The Responses API is used rather than Chat Completions because it is what
    the current reasoning models are served on, and because it reports a
    cut-off answer explicitly. That signal is normalised onto Anthropic's
    spelling, ``max_tokens``, so :meth:`LLM.complete_json` has one stop
    reason to test rather than one per vendor.
    """

    def _build_client(self) -> Any:
        import openai
        return openai.OpenAI(api_key=self._api_key)

    def complete(self, prompt: str, system: str, max_tokens: int
                 ) -> tuple[str, str]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        if system:
            kwargs["instructions"] = system
        try:
            with self._client.responses.stream(**kwargs) as stream:
                response = stream.get_final_response()
        except AttributeError:
            # Older SDK with no Responses API: fall back to a non-streaming
            # Chat Completions call rather than crashing.
            return _OpenAICompatAdapter.complete(self, prompt, system, max_tokens)
        text = str(getattr(response, "output_text", "") or "")
        usage = getattr(response, "usage", None)
        self.last_usage = {
            key: int(getattr(usage, key, 0) or 0)
            for key in ("input_tokens", "output_tokens")
        }
        reason = str(getattr(getattr(response, "incomplete_details", None),
                             "reason", "") or "")
        status = str(getattr(response, "status", "") or "")
        if reason.startswith("max_") or (status == "incomplete" and not reason):
            return text, "max_tokens"
        return text, reason or "end_turn"

    def classify_error(self, exc: BaseException) -> tuple[bool, float | None]:
        return _retryable_openai(exc)


class _OpenAICompatAdapter(_ProviderAdapter):
    """Chat Completions against any OpenAI-compatible provider.

    Groq, Google (via its OpenAI-compatible endpoint) and OpenRouter all
    accept the same shape; the only difference is the ``base_url``. Using
    non-streaming here keeps the adapter small — the surrounding retry /
    fallback loop is what matters for free-tier reliability.
    """

    def _build_client(self) -> Any:
        import openai
        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return openai.OpenAI(**kwargs)

    def complete(self, prompt: str, system: str, max_tokens: int
                 ) -> tuple[str, str]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        # Most OpenAI-compatible providers accept ``max_tokens``; a few newer
        # ones prefer ``max_completion_tokens``. We send the compatible name;
        # if a server rejects it, that counts as a deterministic 400 and the
        # fallback chain moves on to the next provider.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        text = str(getattr(message, "content", "") or "")
        raw_reason = str(getattr(choice, "finish_reason", "") or "")
        if raw_reason == "length":
            reason = "max_tokens"
        elif raw_reason in ("stop", ""):
            reason = "end_turn"
        else:
            reason = raw_reason
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = {
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            }
        else:
            self.last_usage = {}
        return text, reason

    def classify_error(self, exc: BaseException) -> tuple[bool, float | None]:
        return _retryable_openai(exc)


_ADAPTER_FOR: dict[str, type[_ProviderAdapter]] = {
    "anthropic":  _AnthropicAdapter,
    "openai":     _OpenAIResponsesAdapter,
    "groq":       _OpenAICompatAdapter,
    "gemini":     _OpenAICompatAdapter,
    "openrouter": _OpenAICompatAdapter,
}


def _make_adapter(provider: str, api_key: str, model: str,
                  base_url: str | None = None) -> _ProviderAdapter:
    cls = _ADAPTER_FOR.get(provider)
    if cls is None:
        raise LLMUnavailable(f"unknown provider: {provider}")
    if base_url is None:
        base_url = PROVIDER_DEFAULTS.get(provider, {}).get("base_url")
    if not model:
        model = PROVIDER_DEFAULTS.get(provider, {}).get("model", "")
    return cls(provider, api_key, model, base_url=base_url)


# ---------------------------------------------------------------------------
# Normalised response
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    """A provider-independent view of one LLM call's result."""

    text: str
    stop_reason: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""
    fallback_used: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------
class LLM:
    """Front door to every supported vendor.

    Call signatures stay the same as the earlier two-vendor wrapper: pass an
    API key and a model id and the right SDK is picked for you. The new
    ``LLM.from_settings`` factory is preferred when a :class:`Settings`
    object is available, because it also builds the free-tier fallback
    chain described in the project README.
    """

    def __init__(self, api_key: str = "", model: str = "claude-opus-5",
                 *, provider: str | None = None, base_url: str | None = None,
                 fallbacks: list["LLM"] | None = None) -> None:
        prefix, bare = parse_model_id(model)
        self.provider = (provider or prefix or provider_for(model))
        self.model = bare if prefix else model
        if not self.model:
            self.model = PROVIDER_DEFAULTS.get(self.provider, {}).get("model", "")
        self._adapter: _ProviderAdapter | None = None
        self._last_error: str | None = None
        self._fallbacks: list[LLM] = list(fallbacks or [])
        self.last_usage: dict[str, int] = {}
        self.last_provider: str = ""
        self.last_model: str = ""
        self.last_fallback_used: bool = False
        try:
            self._adapter = _make_adapter(
                self.provider, api_key, self.model, base_url=base_url,
            )
        except Exception as exc:  # pragma: no cover
            self._last_error = str(exc)
        if self._adapter is not None and not self._adapter.available:
            self._last_error = self._adapter._init_error

    # The raw underlying SDK client. Legacy tests set this directly to swap
    # in a fake, so it has to stay as a mutable attribute even when the
    # adapter was built without a key.
    @property
    def _client(self) -> Any:
        return self._adapter._client if self._adapter is not None else None

    @_client.setter
    def _client(self, value: Any) -> None:
        if self._adapter is None:
            self._adapter = _make_adapter(self.provider, "ignored", self.model)
        self._adapter._client = value
        self._adapter._init_error = None
        self._last_error = None

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        if self._adapter is not None and self._adapter.available:
            return True
        return any(fb.available for fb in self._fallbacks)

    @property
    def status(self) -> str:
        if self._adapter is not None and self._adapter.available:
            return f"enabled ({self.provider}: {self.model})"
        for fb in self._fallbacks:
            if fb.available:
                return (
                    f"enabled via fallback ({fb.provider}: {fb.model}); "
                    f"primary disabled: {self._last_error}"
                )
        return f"disabled ({self._last_error or 'no provider configured'})"

    @property
    def fallback_providers(self) -> list[str]:
        """Names of the fallback providers currently wired up."""
        return [fb.provider for fb in self._fallbacks if fb.available]

    @property
    def display_name(self) -> str:
        """Human-friendly label, e.g. ``"Claude"`` or ``"Groq"``."""
        provider = self.last_provider or self.provider
        return PROVIDER_DISPLAY_NAME.get(provider, "LLM")

    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Any) -> "LLM":
        """Build an :class:`LLM` with primary provider plus the fallback chain.

        The chain is built from whichever of :data:`FALLBACK_ORDER`'s
        providers have an API key configured, skipping the one chosen as
        primary. Fallbacks are disabled entirely when
        ``settings.fallback_enabled`` is false.
        """
        provider = settings.provider
        api_key = settings.api_key_for(provider)
        model = settings.model_for(provider)
        base_url = settings.base_url_for(provider)
        primary = cls(
            api_key=api_key, model=model,
            provider=provider, base_url=base_url,
        )
        if not getattr(settings, "fallback_enabled", True):
            return primary
        chain: list[LLM] = []
        for name in FALLBACK_ORDER:
            if name == provider:
                continue
            key = settings.api_key_for(name)
            if not key:
                continue
            fallback = cls(
                api_key=key, model=settings.model_for(name),
                provider=name, base_url=settings.base_url_for(name),
            )
            if fallback.available:
                chain.append(fallback)
        primary._fallbacks = chain
        return primary

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def complete(self, prompt: str, system: str = "", max_tokens: int = 2000,
                 retries: int = 2) -> str:
        """Return plain text from the model, or raise :class:`LLMUnavailable`."""
        return self.complete_with_reason(prompt, system, max_tokens, retries)[0]

    def complete_with_reason(self, prompt: str, system: str = "",
                             max_tokens: int = 2000, retries: int = 2
                             ) -> tuple[str, str]:
        """Text plus the reason generation stopped.

        The stop reason matters: ``max_tokens`` means the answer was cut off
        mid-sentence rather than finished, which a caller parsing structured
        output needs to tell apart from a model that simply answered badly.
        """
        resp = self.complete_response(prompt, system, max_tokens, retries)
        return resp.text, resp.stop_reason

    def complete_response(self, prompt: str, system: str = "",
                          max_tokens: int = 2000, retries: int = 2
                          ) -> LLMResponse:
        """The full normalised response, including which provider served it."""
        if not self.available:
            raise LLMUnavailable(self._last_error or "LLM not configured")

        budget = min(max_tokens, MAX_OUTPUT_TOKENS)
        order: list[_ProviderAdapter] = []
        if self._adapter is not None and self._adapter.available:
            order.append(self._adapter)
        for fb in self._fallbacks:
            if fb._adapter is not None and fb._adapter.available:
                order.append(fb._adapter)
        if not order:
            raise LLMUnavailable(self._last_error or "no provider available")

        last_exc: BaseException | None = None
        for index, adapter in enumerate(order):
            try:
                text, reason = _call_with_retry(
                    adapter, prompt, system, budget, retries,
                )
            except _ProviderError as exc:
                last_exc = exc
                if exc.retryable:
                    LOGGER.warning(
                        "%s unavailable (%s); %s",
                        adapter.provider, exc,
                        _next_provider_message(order, index),
                    )
                    continue
                # Non-retryable means a deterministic client bug — surface it
                # right away rather than disguising it behind a fallback.
                raise LLMUnavailable(str(exc)) from exc
            self.last_usage = dict(adapter.last_usage)
            self.last_provider = adapter.provider
            self.last_model = adapter.model
            self.last_fallback_used = index > 0
            return LLMResponse(
                text=text.strip(),
                stop_reason=reason,
                provider=adapter.provider,
                model=adapter.model,
                input_tokens=adapter.last_usage.get("input_tokens", 0),
                output_tokens=adapter.last_usage.get("output_tokens", 0),
                finish_reason=reason,
                fallback_used=index > 0,
            )
        raise LLMUnavailable(
            f"all providers failed; last error: {last_exc}"
            if last_exc else "all providers exhausted"
        )

    # ------------------------------------------------------------------
    def complete_json(self, prompt: str, system: str = "",
                      max_tokens: int = 3000) -> Any:
        """Ask for JSON and parse it, tolerating fenced or chatty output.

        A truncated answer is retried once with a larger budget rather than
        reported as unparseable: an object that was cut off before its
        closing brace is a budget problem, not a model that cannot follow
        instructions.
        """
        guarded = (
            prompt
            + "\n\nRespond with valid JSON only. No prose, no markdown fences, no commentary."
        )
        budget = min(max(max_tokens, JSON_MIN_TOKENS), MAX_OUTPUT_TOKENS)

        while True:
            raw, stop_reason = self.complete_with_reason(
                guarded, system=system, max_tokens=budget,
            )
            parsed = extract_json(raw)
            if parsed is not None:
                return parsed
            if stop_reason != "max_tokens" or budget >= MAX_OUTPUT_TOKENS:
                raise LLMUnavailable(
                    "model did not return parseable JSON"
                    + (" (answer truncated at the output ceiling)"
                       if stop_reason == "max_tokens" else "")
                )
            previous, budget = budget, min(budget * 4, MAX_OUTPUT_TOKENS)
            LOGGER.warning(
                "JSON answer was truncated at %s tokens; retrying with %s",
                previous, budget,
            )


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------
def _next_provider_message(order: list[_ProviderAdapter], index: int) -> str:
    if index + 1 < len(order):
        return f"trying {order[index + 1].provider}..."
    return "no further providers available"


def _call_with_retry(adapter: _ProviderAdapter, prompt: str, system: str,
                     budget: int, retries: int) -> tuple[str, str]:
    """Call ``adapter.complete`` with exponential-backoff retry.

    Classifies each exception as retryable or deterministic through the
    adapter's own ``classify_error`` hook, so a 429 or a timeout gets another
    try while an invalid key or a bad request surfaces immediately.
    """
    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            text, reason = adapter.complete(prompt, system, budget)
        except Exception as exc:
            retryable, retry_after = adapter.classify_error(exc)
            if not retryable:
                raise _ProviderError(
                    f"{type(exc).__name__}: {exc}", retryable=False,
                ) from exc
            last_exc = exc
            LOGGER.warning(
                "%s call failed (attempt %s/%s): %s",
                adapter.provider, attempt + 1, retries + 1, exc,
            )
            if attempt < retries:
                time.sleep(retry_after if retry_after is not None
                          else _backoff(attempt))
                continue
            raise _ProviderError(
                f"{type(exc).__name__}: {exc}", retryable=True,
            ) from exc
        if text.strip():
            return text, reason
        # Empty response: retry a few times, then give up on this provider
        # without falling back — an empty answer is usually a soft bug the
        # caller handles (deterministic template fallback) rather than a
        # provider-level failure.
        last_exc = RuntimeError(
            "empty response"
            + (" (the token budget was spent before the answer began)"
               if reason == "max_tokens" else "")
        )
        if attempt < retries:
            time.sleep(_backoff(attempt))
            continue
        raise _ProviderError(str(last_exc), retryable=False)
    # Unreachable, but satisfies the type-checker.
    raise _ProviderError(                                      # pragma: no cover
        str(last_exc) if last_exc else "unknown error", retryable=False,
    )


def _backoff(attempt: int) -> float:
    """Exponential backoff with a conservative cap."""
    return min(1.5 * (2 ** attempt), 15.0)


# ---------------------------------------------------------------------------
def extract_json(text: str) -> Any | None:
    """Best-effort JSON extraction from an LLM response."""
    if not text:
        return None
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost balanced object / array in the string.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None
