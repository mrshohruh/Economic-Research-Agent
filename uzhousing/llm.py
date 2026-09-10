"""Thin, defensive wrapper around the Anthropic Messages API.

The whole pipeline is designed to work without an API key, so every entry point
here either returns a usable value or raises :class:`LLMUnavailable`, which
callers catch to fall back to deterministic behaviour.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

LOGGER = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """Raised when no LLM answer could be obtained."""


class LLM:
    """Small helper around ``anthropic.Anthropic``."""

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-5") -> None:
        self.model = model
        self._client = None
        self._last_error: str | None = None
        if not api_key:
            self._last_error = "no ANTHROPIC_API_KEY set"
            return
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
        except Exception as exc:  # pragma: no cover - import/credential issues
            self._last_error = f"anthropic client unavailable: {exc}"
            LOGGER.warning("LLM disabled: %s", self._last_error)

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def status(self) -> str:
        if self.available:
            return f"enabled ({self.model})"
        return f"disabled ({self._last_error})"

    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2000,
        temperature: float = 0.2,
        retries: int = 2,
    ) -> str:
        """Return plain text from the model, or raise :class:`LLMUnavailable`."""
        if not self.available:
            raise LLMUnavailable(self._last_error or "LLM not configured")

        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system:
                    kwargs["system"] = system
                response = self._client.messages.create(**kwargs)
                text = "".join(
                    block.text for block in response.content if getattr(block, "type", "") == "text"
                )
                if text.strip():
                    return text.strip()
                last_exc = RuntimeError("empty response")
            except Exception as exc:  # pragma: no cover - network dependent
                last_exc = exc
                LOGGER.warning("LLM call failed (attempt %s): %s", attempt + 1, exc)
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        raise LLMUnavailable(str(last_exc))

    # ------------------------------------------------------------------
    def complete_json(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 3000,
        temperature: float = 0.0,
    ) -> Any:
        """Ask for JSON and parse it, tolerating fenced or chatty output."""
        guarded = (
            prompt
            + "\n\nRespond with valid JSON only. No prose, no markdown fences, no commentary."
        )
        raw = self.complete(guarded, system=system, max_tokens=max_tokens, temperature=temperature)
        parsed = extract_json(raw)
        if parsed is None:
            raise LLMUnavailable("model did not return parseable JSON")
        return parsed


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
