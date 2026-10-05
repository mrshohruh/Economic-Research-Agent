"""O'zbekiston uy-joy bozori sharhi."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__version__ = "2.0.0"


@dataclass
class RunResult:
    report_path: Path | None = None
    pdf_path: Path | None = None
    run_log: Path | None = None
    duration_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.report_path is not None and self.report_path.exists()


__all__ = ["RunResult", "__version__"]
