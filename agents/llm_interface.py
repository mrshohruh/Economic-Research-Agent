"""
Modular LLM interface. The rest of the codebase never talks to a provider
SDK directly -- it calls `LLM.complete(system, user, ...)`. This keeps the
system provider-agnostic (per architecture requirement #28) and lets every
agent degrade gracefully to deterministic template text when no API key is
configured, so V1 runs end-to-end without any external key.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)


class LLMInterface:
    def __init__(self) -> None:
        self.enabled = settings.LLM_ENABLED
        self._client = None
        if self.enabled:
            try:
                import anthropic  # local import: optional dependency at runtime

                self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to initialize Anthropic client: %s", exc)
                self.enabled = False

    def complete(self, system: str, user: str, max_tokens: int = 1200,
                 temperature: float = 0.3) -> Optional[str]:
        """Return LLM text, or None if the LLM is unavailable/fails.

        Callers MUST handle a None return with a deterministic fallback --
        the system must never block on an LLM call being available.
        """
        if not self.enabled or self._client is None:
            return None
        try:
            resp = self._client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "\n".join(parts).strip() or None
        except Exception as exc:  # pragma: no cover - network/runtime errors
            logger.warning("LLM call failed, falling back to template output: %s", exc)
            return None

    def complete_json(self, system: str, user: str, max_tokens: int = 1200) -> Optional[Any]:
        text = self.complete(system, user, max_tokens=max_tokens)
        if text is None:
            return None
        # Be tolerant of ```json fences.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        try:
            return json.loads(cleaned)
        except Exception:
            logger.warning("LLM did not return valid JSON; ignoring.")
            return None


LLM = LLMInterface()
