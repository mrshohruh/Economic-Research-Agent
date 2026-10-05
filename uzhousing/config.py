"""Sozlamalar - .env fayldan yuklanadi."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .llm import (
    ENV_KEY_FOR_PROVIDER,
    FALLBACK_ORDER,
    PROVIDER_DEFAULTS,
    parse_model_id,
    provider_for,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_PROVIDERS: tuple[str, ...] = tuple(ENV_KEY_FOR_PROVIDER.keys())


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


@dataclass
class Settings:
    """Barcha sozlamalar."""

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""

    model: str = "claude-opus-5"
    llm_provider_setting: str = ""
    llm_model_setting: str = ""

    groq_model: str = ""
    gemini_model: str = ""
    openrouter_model: str = ""

    fallback_enabled: bool = True

    web_research: bool = True
    search_results_per_query: int = 6
    pages_to_read: int = 3
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    policy_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / "knowledge" / "policy_events.json"
    )
    request_timeout: int = 20
    uzs_per_usd: float = 12_650.0
    olx_archive: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        out = os.getenv("OUTPUT_DIR", "").strip() or "outputs"
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        provider_setting = (os.getenv("LLM_PROVIDER", "") or "").strip().lower()
        if provider_setting and provider_setting not in SUPPORTED_PROVIDERS:
            provider_setting = ""
        return cls(
            anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY", "") or "").strip(),
            openai_api_key=(os.getenv("OPENAI_API_KEY", "") or "").strip(),
            groq_api_key=(os.getenv("GROQ_API_KEY", "") or "").strip(),
            gemini_api_key=(os.getenv("GEMINI_API_KEY", "") or "").strip(),
            openrouter_api_key=(os.getenv("OPENROUTER_API_KEY", "") or "").strip(),
            model=(
                os.getenv("MODEL", "")
                or os.getenv("ANTHROPIC_MODEL", "")
                or "claude-opus-5"
            ).strip(),
            llm_provider_setting=provider_setting,
            llm_model_setting=(os.getenv("LLM_MODEL", "") or "").strip(),
            groq_model=(os.getenv("GROQ_MODEL", "") or "").strip(),
            gemini_model=(os.getenv("GEMINI_MODEL", "") or "").strip(),
            openrouter_model=(os.getenv("OPENROUTER_MODEL", "") or "").strip(),
            fallback_enabled=_bool_env("LLM_FALLBACK_ENABLED", True),
            web_research=_bool_env("WEB_RESEARCH", True),
            search_results_per_query=_int_env("SEARCH_RESULTS_PER_QUERY", 6),
            pages_to_read=_int_env("PAGES_TO_READ", 3),
            output_dir=out_path,
            uzs_per_usd=_float_env("UZS_PER_USD", 12_650.0),
            olx_archive=(os.getenv("OLX_ARCHIVE", "") or "").strip(),
        )

    def api_key_for(self, provider: str) -> str:
        return {
            "anthropic":  self.anthropic_api_key,
            "openai":     self.openai_api_key,
            "groq":       self.groq_api_key,
            "gemini":     self.gemini_api_key,
            "openrouter": self.openrouter_api_key,
        }.get(provider, "")

    def model_for(self, provider: str) -> str:
        if provider == self._primary_candidate_provider() and self.llm_model_setting:
            _, bare = parse_model_id(self.llm_model_setting)
            if bare:
                return bare
        if provider in ("anthropic", "openai"):
            legacy_provider = provider_for(self.model)
            if legacy_provider == provider:
                return self.model
            return PROVIDER_DEFAULTS.get(provider, {}).get("model", "")
        override = {
            "groq":       self.groq_model,
            "gemini":     self.gemini_model,
            "openrouter": self.openrouter_model,
        }.get(provider, "")
        if override:
            return override
        return PROVIDER_DEFAULTS.get(provider, {}).get("model", "")

    def base_url_for(self, provider: str) -> str | None:
        return PROVIDER_DEFAULTS.get(provider, {}).get("base_url")

    def _primary_explicit_provider(self) -> str:
        if self.llm_provider_setting:
            return self.llm_provider_setting
        if self.llm_model_setting:
            prefix, _ = parse_model_id(self.llm_model_setting)
            if prefix:
                return prefix
        return ""

    def _primary_candidate_provider(self) -> str:
        explicit = self._primary_explicit_provider()
        if explicit:
            return explicit
        if self.model:
            low = self.model.strip().lower()
            if low.startswith(("claude", "anthropic.")):
                return "anthropic"
            if low.startswith(("gpt", "chatgpt", "o1", "o3", "o4", "o5")):
                return "openai"
        return ""

    @property
    def provider(self) -> str:
        explicit = self._primary_explicit_provider()
        if explicit:
            return explicit
        legacy = self._primary_candidate_provider()
        if legacy and (self.api_key_for(legacy) or not any(
            self.api_key_for(name) for name in FALLBACK_ORDER
        )):
            return legacy
        for name in FALLBACK_ORDER:
            if self.api_key_for(name):
                return name
        return legacy or "anthropic"

    @property
    def llm_model(self) -> str:
        return self.model_for(self.provider)

    @property
    def llm_api_key(self) -> str:
        return self.api_key_for(self.provider)

    @property
    def llm_key_name(self) -> str:
        return ENV_KEY_FOR_PROVIDER.get(self.provider, "API_KEY")

    @property
    def llm_enabled(self) -> bool:
        if self.llm_api_key:
            return True
        if self.fallback_enabled:
            return any(self.api_key_for(name) for name in FALLBACK_ORDER)
        return False

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def runs_dir(self) -> Path:
        return self.output_dir / "runs"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def snapshots_dir(self) -> Path:
        return self.output_dir / "olx_snapshots"

    def ensure_dirs(self) -> None:
        for path in (
            self.output_dir,
            self.reports_dir,
            self.runs_dir,
            self.tables_dir,
            self.snapshots_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
