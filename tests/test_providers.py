"""Tests for the multi-provider LLM abstraction.

Nothing here makes a real external API call: every provider is swapped for a
small in-process double that records what was asked for and replays canned
answers. Keeps the suite offline-friendly while covering provider selection,
free-tier fallback, retry handling, response normalisation and the deterministic
no-LLM path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uzhousing import llm as llm_mod  # noqa: E402
from uzhousing.config import Settings  # noqa: E402
from uzhousing.llm import (  # noqa: E402
    FALLBACK_ORDER,
    LLM,
    LLMUnavailable,
    PROVIDER_DEFAULTS,
    provider_for,
    parse_model_id,
)


# ---------------------------------------------------------------------------
# Low-level provider resolver
# ---------------------------------------------------------------------------
class TestProviderResolver:
    def test_claude_model_still_routes_to_anthropic(self):
        assert provider_for("claude-opus-5") == "anthropic"
        assert provider_for("claude-sonnet-5") == "anthropic"

    def test_openai_prefixes_still_route_to_openai(self):
        assert provider_for("gpt-5") == "openai"
        assert provider_for("gpt-6-astra") == "openai"
        assert provider_for("o4-mini") == "openai"

    def test_explicit_prefix_wins(self):
        assert provider_for("groq:openai/gpt-oss-120b") == "groq"
        assert provider_for("gemini:gemini-2.5-flash") == "gemini"
        assert provider_for("openrouter:any/model") == "openrouter"
        assert provider_for("openai:gpt-5") == "openai"
        assert provider_for("anthropic:claude-sonnet-5") == "anthropic"

    def test_parse_model_id_preserves_tail_colons(self):
        prefix, bare = parse_model_id("groq:openai/gpt-oss-120b")
        assert prefix == "groq"
        assert bare == "openai/gpt-oss-120b"

    def test_unknown_prefix_falls_through_to_anthropic(self):
        # Original behaviour: unknown ids default to Anthropic.
        assert provider_for("some-other-id") == "anthropic"


# ---------------------------------------------------------------------------
# Settings: provider selection and per-provider lookups
# ---------------------------------------------------------------------------
def _blank_settings(**overrides) -> Settings:
    """A Settings object with every API key empty, with specific overrides."""
    s = Settings()
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


class TestSettingsProvider:
    def test_explicit_llm_provider_wins_over_model(self):
        s = _blank_settings(
            llm_provider_setting="groq", model="claude-opus-5",
            groq_api_key="k",
        )
        assert s.provider == "groq"

    def test_llm_model_prefix_implies_provider(self):
        s = _blank_settings(
            llm_model_setting="gemini:gemini-2.5-flash",
            gemini_api_key="k",
        )
        assert s.provider == "gemini"

    def test_legacy_model_routes_to_anthropic(self):
        s = _blank_settings(model="claude-opus-5", anthropic_api_key="k")
        assert s.provider == "anthropic"

    def test_legacy_model_routes_to_openai(self):
        s = _blank_settings(model="gpt-5", openai_api_key="k")
        assert s.provider == "openai"

    def test_free_key_preferred_when_legacy_key_missing(self):
        # Classic .env but anthropic key missing: a free provider's key
        # should become the primary.
        s = _blank_settings(model="claude-opus-5", groq_api_key="k")
        assert s.provider == "groq"

    def test_no_keys_defaults_to_anthropic(self):
        s = _blank_settings()
        assert s.provider == "anthropic"

    def test_api_key_lookup_table(self):
        s = _blank_settings(
            anthropic_api_key="a", openai_api_key="b", groq_api_key="c",
            gemini_api_key="d", openrouter_api_key="e",
        )
        assert s.api_key_for("anthropic") == "a"
        assert s.api_key_for("openai") == "b"
        assert s.api_key_for("groq") == "c"
        assert s.api_key_for("gemini") == "d"
        assert s.api_key_for("openrouter") == "e"
        assert s.api_key_for("unknown") == ""

    def test_model_for_falls_back_to_defaults(self):
        s = _blank_settings()
        assert s.model_for("groq") == PROVIDER_DEFAULTS["groq"]["model"]
        assert s.model_for("gemini") == PROVIDER_DEFAULTS["gemini"]["model"]
        assert s.model_for("openrouter") == PROVIDER_DEFAULTS["openrouter"]["model"]

    def test_model_for_honours_overrides(self):
        s = _blank_settings(groq_model="custom-groq", gemini_model="custom-gemini",
                            openrouter_model="custom-router")
        assert s.model_for("groq") == "custom-groq"
        assert s.model_for("gemini") == "custom-gemini"
        assert s.model_for("openrouter") == "custom-router"

    def test_base_urls_for_free_providers_are_set(self):
        s = _blank_settings()
        assert s.base_url_for("groq") == "https://api.groq.com/openai/v1"
        assert s.base_url_for("gemini").startswith("https://generativelanguage.googleapis.com")
        assert s.base_url_for("openrouter") == "https://openrouter.ai/api/v1"
        # The two legacy providers do not need a base URL.
        assert s.base_url_for("anthropic") is None
        assert s.base_url_for("openai") is None

    def test_llm_enabled_true_with_only_free_key(self):
        s = _blank_settings(groq_api_key="k")
        assert s.llm_enabled is True

    def test_llm_enabled_false_without_keys(self):
        s = _blank_settings()
        assert s.llm_enabled is False

    def test_llm_enabled_false_when_fallback_disabled_and_no_primary(self):
        s = _blank_settings(groq_api_key="k", fallback_enabled=False,
                            llm_provider_setting="openai")
        # Primary is openai, no key for it, and fallback is disabled.
        assert s.llm_enabled is False


# ---------------------------------------------------------------------------
# Provider adapters: a shared fake client swaps in for the OpenAI SDK and the
# Anthropic SDK, so no network traffic ever leaves the box.
# ---------------------------------------------------------------------------
class _FakeChatCompletions:
    def __init__(self, replies, raises=None, model_calls=None):
        self._replies = list(replies)
        self._raises = list(raises or [])
        self.calls = model_calls if model_calls is not None else []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises:
            exc = self._raises.pop(0)
            if exc is not None:
                raise exc
        text, finish_reason = self._replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        )


class _FakeOpenAIClient:
    """Stand-in for openai.OpenAI used by the Chat Completions adapter."""

    def __init__(self, replies, raises=None):
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=_FakeChatCompletions(replies, raises=raises,
                                             model_calls=self.calls),
        )


def _fake_openai_llm(provider: str, replies, raises=None,
                     *, model: str = "fake-model") -> LLM:
    """Build an LLM for ``provider`` with its openai client swapped out."""
    base_url = PROVIDER_DEFAULTS.get(provider, {}).get("base_url")
    llm = LLM(api_key="dummy", model=model, provider=provider,
              base_url=base_url)
    llm._client = _FakeOpenAIClient(replies, raises=raises)
    return llm


# ---------------------------------------------------------------------------
# Provider selection via LLM
# ---------------------------------------------------------------------------
class TestLLMProviderSelection:
    def test_groq_adapter_uses_chat_completions(self):
        llm = _fake_openai_llm("groq", [("Groq says hi.", "stop")])
        text = llm.complete("Hello")
        assert text == "Groq says hi."
        call = llm._client.calls[0]
        assert call["model"] == "fake-model"
        assert call["messages"][-1]["content"] == "Hello"

    def test_gemini_adapter_uses_chat_completions(self):
        llm = _fake_openai_llm("gemini", [("Gemini output.", "stop")])
        assert llm.complete("hi") == "Gemini output."

    def test_openrouter_adapter_uses_chat_completions(self):
        llm = _fake_openai_llm("openrouter", [("Via OpenRouter.", "stop")])
        assert llm.complete("hi") == "Via OpenRouter."

    def test_openai_provider_uses_responses_api(self):
        # Preserve the existing OpenAI Responses-API behaviour. The adapter's
        # ``_client.responses.stream`` is the attribute the pipeline reaches
        # for, so a fake modelled on the real streaming context manager works.
        class _FakeStream:
            def __init__(self, text):
                self._text = text

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_response(self):
                return SimpleNamespace(
                    output_text=self._text,
                    usage=SimpleNamespace(input_tokens=5, output_tokens=7),
                    incomplete_details=None,
                    status="completed",
                )

        class _Responses:
            def __init__(self, text):
                self._text = text

            def stream(self, **_kwargs):
                return _FakeStream(self._text)

        llm = LLM(api_key="dummy", model="gpt-5", provider="openai")
        llm._client = SimpleNamespace(responses=_Responses("OpenAI output"))
        assert llm.complete("hello") == "OpenAI output"

    def test_anthropic_provider_keeps_messages_stream(self):
        # ``LLM(api_key="", model="claude-opus-5")`` goes to the Anthropic
        # adapter. The legacy test fixture monkey-patches ``_client`` with a
        # tiny recorder; the same fixture must keep working here.
        class _AnthroStream:
            def __init__(self, text):
                self._text = text

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text=self._text)],
                    stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=3, output_tokens=4),
                )

        class _Messages:
            def __init__(self, text):
                self._text = text

            def stream(self, **_kwargs):
                return _AnthroStream(self._text)

        llm = LLM(api_key="dummy", model="claude-opus-5")
        llm._client = SimpleNamespace(messages=_Messages("Claude output"))
        assert llm.complete("hi") == "Claude output"

    def test_missing_api_key_disables_llm(self):
        llm = LLM(api_key="", model="groq:openai/gpt-oss-120b")
        assert llm.available is False
        assert "GROQ_API_KEY" in llm.status
        with pytest.raises(LLMUnavailable):
            llm.complete("hi")


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------
class _RateLimitError(Exception):
    """Looks like an openai.RateLimitError to the error classifier."""

    def __init__(self):
        super().__init__("429 Too Many Requests")
        self.status_code = 429
        self.response = SimpleNamespace(status_code=429, headers={})


class _AuthError(Exception):
    def __init__(self):
        super().__init__("401 Unauthorized")
        self.status_code = 401
        self.response = SimpleNamespace(status_code=401, headers={})


class _Boom(Exception):
    """A deterministic programming bug: must NOT trigger provider fallback."""


class TestFallbackChain:
    def _primary_with_fallback(self, primary, fallbacks):
        primary._fallbacks = list(fallbacks)
        return primary

    def test_groq_rate_limit_falls_over_to_gemini(self, monkeypatch):
        groq = _fake_openai_llm(
            "groq",
            replies=[("ignored", "stop")],
            raises=[_RateLimitError(), _RateLimitError(), _RateLimitError()],
        )
        gemini = _fake_openai_llm(
            "gemini", replies=[("Gemini saved the day.", "stop")],
        )
        chain = self._primary_with_fallback(groq, [gemini])
        # Monkey-patch openai error classification to recognise our fakes.
        monkeypatch.setattr(llm_mod, "_retryable_openai",
                            _fake_retry_classifier)
        resp = chain.complete_response("hi")
        assert resp.text == "Gemini saved the day."
        assert resp.provider == "gemini"
        assert resp.fallback_used is True
        assert chain.last_provider == "gemini"
        assert chain.last_fallback_used is True

    def test_groq_then_gemini_then_openrouter(self, monkeypatch):
        groq = _fake_openai_llm(
            "groq", replies=[("x", "stop")],
            raises=[_RateLimitError()] * 3,
        )
        gemini = _fake_openai_llm(
            "gemini", replies=[("y", "stop")],
            raises=[_RateLimitError()] * 3,
        )
        openrouter = _fake_openai_llm(
            "openrouter", replies=[("Finally.", "stop")],
        )
        chain = self._primary_with_fallback(groq, [gemini, openrouter])
        monkeypatch.setattr(llm_mod, "_retryable_openai",
                            _fake_retry_classifier)
        resp = chain.complete_response("hi")
        assert resp.text == "Finally."
        assert resp.provider == "openrouter"

    def test_programming_bug_does_not_trigger_fallback(self, monkeypatch):
        # A random exception (not an API error) must NOT cause fallback —
        # it is almost always a bug the user needs to see.
        groq = _fake_openai_llm("groq", replies=[("x", "stop")],
                                raises=[_Boom("surprise")])
        gemini = _fake_openai_llm("gemini", replies=[("saved", "stop")])
        chain = self._primary_with_fallback(groq, [gemini])
        monkeypatch.setattr(llm_mod, "_retryable_openai",
                            _fake_retry_classifier)
        with pytest.raises(LLMUnavailable):
            chain.complete("hi")

    def test_auth_error_does_not_trigger_fallback(self, monkeypatch):
        groq = _fake_openai_llm("groq", replies=[("x", "stop")],
                                raises=[_AuthError()])
        gemini = _fake_openai_llm("gemini", replies=[("saved", "stop")])
        chain = self._primary_with_fallback(groq, [gemini])
        monkeypatch.setattr(llm_mod, "_retryable_openai",
                            _fake_retry_classifier)
        with pytest.raises(LLMUnavailable):
            chain.complete("hi")

    def test_from_settings_builds_the_chain(self, monkeypatch):
        s = _blank_settings(
            llm_provider_setting="groq",
            groq_api_key="g", gemini_api_key="e", openrouter_api_key="o",
        )
        llm = LLM.from_settings(s)
        assert llm.provider == "groq"
        assert llm.fallback_providers == ["gemini", "openrouter"]

    def test_from_settings_respects_disabled_fallback(self):
        s = _blank_settings(
            llm_provider_setting="groq",
            groq_api_key="g", gemini_api_key="e",
            fallback_enabled=False,
        )
        llm = LLM.from_settings(s)
        assert llm.fallback_providers == []

    def test_from_settings_leaves_order_in_preferred_sequence(self):
        s = _blank_settings(
            llm_provider_setting="openai", openai_api_key="o",
            groq_api_key="g", gemini_api_key="e", openrouter_api_key="r",
            anthropic_api_key="a",
        )
        llm = LLM.from_settings(s)
        # Primary is openai, fallbacks follow FALLBACK_ORDER minus openai,
        # minus any provider without a key.
        expected = [p for p in FALLBACK_ORDER if p != "openai"]
        assert llm.fallback_providers == expected


def _fake_retry_classifier(exc):
    """Classify our ``_RateLimitError`` / ``_AuthError`` / plain exceptions.

    Replaces :func:`uzhousing.llm._retryable_openai` under monkey-patch so the
    tests do not depend on the real ``openai`` exception hierarchy.
    """
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True, None
    if status in (500, 502, 503, 504):
        return True, None
    if status in (401, 403, 404, 400):
        return False, None
    return False, None


# ---------------------------------------------------------------------------
# Retry handling
# ---------------------------------------------------------------------------
class TestRetryBehaviour:
    def test_rate_limit_is_retried_within_the_same_provider(self, monkeypatch):
        groq = _fake_openai_llm(
            "groq", replies=[("recovered", "stop")],
            raises=[_RateLimitError(), None],
        )
        monkeypatch.setattr(llm_mod, "_retryable_openai",
                            _fake_retry_classifier)
        monkeypatch.setattr(llm_mod.time, "sleep", lambda _s: None)
        assert groq.complete("hi") == "recovered"
        # Two calls to the chat completions endpoint, one failure + one success.
        assert len(groq._client.calls) == 2

    def test_retry_respects_retry_after_header(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(llm_mod.time, "sleep", lambda s: sleeps.append(s))

        exc = _RateLimitError()
        exc.response = SimpleNamespace(status_code=429,
                                       headers={"retry-after": "0.4"})

        def classify(exc_):
            if getattr(exc_, "status_code", None) == 429:
                return True, llm_mod._retry_after_from_headers(exc_)
            return False, None

        monkeypatch.setattr(llm_mod, "_retryable_openai", classify)

        groq = _fake_openai_llm(
            "groq", replies=[("ok", "stop")], raises=[exc, None],
        )
        groq.complete("hi")
        assert sleeps and sleeps[0] == 0.4

    def test_retries_exhausted_raises(self, monkeypatch):
        groq = _fake_openai_llm(
            "groq", replies=[("x", "stop")],
            raises=[_RateLimitError()] * 10,
        )
        monkeypatch.setattr(llm_mod, "_retryable_openai",
                            _fake_retry_classifier)
        monkeypatch.setattr(llm_mod.time, "sleep", lambda _s: None)
        with pytest.raises(LLMUnavailable):
            groq.complete("hi")


# ---------------------------------------------------------------------------
# Normalised response shape
# ---------------------------------------------------------------------------
class TestResponseShape:
    def test_complete_response_carries_metadata(self):
        llm = _fake_openai_llm("groq", [("Hello world.", "stop")])
        resp = llm.complete_response("hi")
        assert resp.text == "Hello world."
        assert resp.provider == "groq"
        assert resp.model == "fake-model"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 20
        assert resp.total_tokens == 30
        assert resp.fallback_used is False
        assert resp.stop_reason == "end_turn"

    def test_length_finish_reason_is_normalised_to_max_tokens(self):
        llm = _fake_openai_llm("groq", [("partial", "length")])
        text, reason = llm.complete_with_reason("hi")
        assert text == "partial"
        assert reason == "max_tokens"


# ---------------------------------------------------------------------------
# Deterministic / no-LLM mode: the pipeline must still work with no key.
# ---------------------------------------------------------------------------
class TestNoLLMMode:
    def test_llm_unavailable_without_any_key(self):
        s = _blank_settings()
        llm = LLM.from_settings(s)
        assert llm.available is False
        with pytest.raises(LLMUnavailable):
            llm.complete("hi")

    def test_settings_llm_enabled_is_false_without_keys(self):
        s = _blank_settings()
        assert s.llm_enabled is False


# ---------------------------------------------------------------------------
# Backward-compatible CLI / legacy constructor
# ---------------------------------------------------------------------------
class TestBackwardCompatibility:
    def test_legacy_llm_constructor_without_key(self):
        # Existing tests call this exact signature; keep it working.
        llm = LLM(api_key="", model="stub-model")
        assert llm.available is False
        assert llm.provider == "anthropic"

    def test_legacy_claude_model_still_default_anthropic(self):
        llm = LLM(api_key="k", model="claude-opus-5")
        assert llm.provider == "anthropic"

    def test_legacy_gpt_model_routes_to_openai(self):
        llm = LLM(api_key="k", model="gpt-5")
        assert llm.provider == "openai"

    def test_cli_help_lists_new_flags(self):
        from uzhousing.cli import build_parser

        parser = build_parser()
        help_text = parser.format_help()
        assert "--provider" in help_text
        assert "--no-llm-fallback" in help_text
        assert "--model" in help_text

    def test_cli_accepts_new_flags(self):
        from uzhousing.cli import build_parser

        args = build_parser().parse_args([
            "--data", "x.json",
            "--provider", "groq",
            "--model", "groq:openai/gpt-oss-120b",
            "--no-llm-fallback",
        ])
        assert args.provider == "groq"
        assert args.model == "groq:openai/gpt-oss-120b"
        assert args.llm_fallback is False


# ---------------------------------------------------------------------------
# Full pipeline integration smoke: no real LLM call, just that fallback
# wiring survives a monkey-patched LLM substitute.
# ---------------------------------------------------------------------------
class TestPipelineIntegration:
    def test_from_settings_builds_chain_for_full_run(self, monkeypatch):
        s = _blank_settings(
            llm_provider_setting="groq",
            groq_api_key="g", gemini_api_key="e",
        )
        llm = LLM.from_settings(s)
        # Primary is groq (available because key is set, even though the
        # adapter will fail at call time without network); fallback chain
        # includes gemini.
        assert llm.provider == "groq"
        assert "gemini" in llm.fallback_providers
