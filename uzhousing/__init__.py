"""Uzbekistan Housing Market Research Agent.

Give it a JSON file, a SQL script or a database, and it profiles the data,
analyses trends and drivers, researches the policy and macro backdrop, and
writes a fully formatted Word report with figures and tables.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["Settings", "run", "__version__"]


def __getattr__(name: str):  # lazy so `import uzhousing` stays cheap
    if name == "Settings":
        from .config import Settings

        return Settings
    if name == "run":
        from .pipeline import run

        return run
    raise AttributeError(name)
