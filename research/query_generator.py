"""Thin re-export: query generation logic lives alongside the research plan
(agents/research_planner.py) since queries are derived directly from
detected findings. Exposed here too for architectural clarity."""
from agents.research_planner import build_search_queries  # noqa: F401
