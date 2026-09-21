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


class LLMUnavailable(RuntimeError):
    """Raised when no LLM answer could be obtained."""


class LLM:
    """Small helper around ``anthropic.Anthropic``."""

    def __init__(self, api_key: str = "", model: str = "claude-opus-5") -> None:
        self.model = model
        self._client = None
        self.last_usage: dict[str, int] = {}
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
        retries: int = 2,
    ) -> str:
        """Return plain text from the model, or raise :class:`LLMUnavailable`."""
        return self.complete_with_reason(prompt, system, max_tokens, retries)[0]

    def complete_with_reason(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2000,
        retries: int = 2,
    ) -> tuple[str, str]:
        """Text plus the reason generation stopped.

        The stop reason matters: ``max_tokens`` means the answer was cut off
        mid-sentence rather than finished, which a caller parsing structured
        output needs to tell apart from a model that simply answered badly.

        Requests are streamed. The output ceilings here are large enough that a
        single non-streaming request can exceed the SDK's HTTP timeout.

        Sampling parameters are deliberately not sent. ``temperature``, ``top_p``
        and ``top_k`` were removed from the Messages API for the current model
        generation (Sonnet 5, Opus 5 and later); passing one is rejected outright,
        which previously disabled every LLM step in the pipeline.
        """
        if not self.available:
            raise LLMUnavailable(self._last_error or "LLM not configured")

        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": min(max_tokens, MAX_OUTPUT_TOKENS),
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system:
                    kwargs["system"] = system
                with self._client.messages.stream(**kwargs) as stream:
                    message = stream.get_final_message()
                # Thinking blocks are returned with their text omitted by
                # default, so only the answer blocks are collected here.
                text = "".join(
                    block.text for block in message.content if getattr(block, "type", "") == "text"
                )
                stop_reason = str(getattr(message, "stop_reason", "") or "")
                if text.strip():
                    usage = getattr(message, "usage", None)
                    self.last_usage = {key: int(getattr(usage, key, 0) or 0) for key in ("input_tokens", "output_tokens")}
                    return text.strip(), stop_reason
                last_exc = RuntimeError(
                    "empty response"
                    + (" (the token budget was spent before the answer began)"
                       if stop_reason == "max_tokens" else "")
                )
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
    ) -> Any:
        """Ask for JSON and parse it, tolerating fenced or chatty output.

        A truncated answer is retried once with a larger budget rather than
        reported as unparseable: an object that was cut off before its closing
        brace is a budget problem, not a model that cannot follow instructions.
        """
        guarded = (
            prompt
            + "\n\nRespond with valid JSON only. No prose, no markdown fences, no commentary."
        )
        budget = min(max(max_tokens, JSON_MIN_TOKENS), MAX_OUTPUT_TOKENS)

        while True:
            raw, stop_reason = self.complete_with_reason(
                guarded, system=system, max_tokens=budget
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
                "JSON answer was truncated at %s tokens; retrying with %s", previous, budget
            )


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
