"""
Structured data schemas used across the Economic Research Agent pipeline.

Every module communicates through these Pydantic models instead of free-form
text/dicts, so that data understanding, analysis, research, visualization and
report writing remain traceable and testable independently of any LLM.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]
Importance = Literal["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Stage 1-2: Data understanding & validation
# ---------------------------------------------------------------------------

class VariableMeta(BaseModel):
    name: str
    column: str
    unit: Optional[str] = None
    dtype: str
    kind: Literal["numeric", "categorical", "date", "text", "unknown"] = "unknown"
    definition: Optional[str] = None
    confidence: Confidence = "medium"
    missing_count: int = 0
    missing_pct: float = 0.0
    n_obs: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


class SheetMeta(BaseModel):
    sheet_name: str
    n_rows: int
    n_cols: int
    date_column: Optional[str] = None
    frequency: Optional[str] = None  # "daily","monthly","quarterly","annual","irregular"
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    variables: list[VariableMeta] = Field(default_factory=list)


class DatasetMetadata(BaseModel):
    dataset_name: str
    file_path: str
    sheets: list[SheetMeta] = Field(default_factory=list)
    primary_sheet: Optional[str] = None
    ambiguous_variables: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ValidationIssue(BaseModel):
    level: Literal["ok", "warning", "error"]
    message: str
    details: Optional[str] = None


class ValidationReport(BaseModel):
    n_observations: int
    frequency_detected: Optional[str] = None
    date_column: Optional[str] = None
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.level == "warning")

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.level == "error")


# ---------------------------------------------------------------------------
# Stage 3-4: Quantitative analysis & important-change detection
# ---------------------------------------------------------------------------

class StatResult(BaseModel):
    label: str
    variable: str
    stat_type: str  # e.g. "mean", "yoy_change", "mom_change", "correlation"
    value: Optional[float] = None
    unit: Optional[str] = None
    period: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    id: str
    finding: str
    importance: Importance
    variables: list[str] = Field(default_factory=list)
    period: Optional[str] = None
    magnitude: Optional[str] = None
    direction: Optional[str] = None
    unusual: bool = False
    co_movements: list[str] = Field(default_factory=list)
    possible_explanations: list[str] = Field(default_factory=list)
    supporting_stats: list[StatResult] = Field(default_factory=list)
    source: str = "computed from dataset"


# ---------------------------------------------------------------------------
# Stage 5-6: Research plan & hypotheses
# ---------------------------------------------------------------------------

class ResearchQuestion(BaseModel):
    id: str
    question: str
    related_finding_ids: list[str] = Field(default_factory=list)


class Hypothesis(BaseModel):
    id: str
    finding_id: str
    label: str
    description: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = "low"


class ResearchPlan(BaseModel):
    topic: str
    steps: list[str] = Field(default_factory=list)
    questions: list[ResearchQuestion] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 7-8: Web research & evidence
# ---------------------------------------------------------------------------

class ResearchEvidence(BaseModel):
    id: str
    claim: str
    source_title: str
    institution: Optional[str] = None
    url: str
    publication_date: Optional[str] = None
    snippet: str
    relevance: float = 0.5
    source_quality: float = 0.5
    confidence: Confidence = "medium"
    query_used: Optional[str] = None


class PolicyEvent(BaseModel):
    date: Optional[str] = None
    institution: str
    policy: str
    economic_channel: str
    evidence_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage 9-11: Visualization & tables
# ---------------------------------------------------------------------------

class FigurePlan(BaseModel):
    id: str
    title: str
    fig_type: Literal["line", "bar", "stacked_bar", "indexed_line", "growth_bar",
                       "contribution_bar", "moving_average", "scatter", "dual_axis"]
    variables: list[str]
    period: Optional[str] = None
    purpose: str
    reason: str
    related_finding_ids: list[str] = Field(default_factory=list)
    file_path: Optional[str] = None
    analysis_text: Optional[str] = None


class TablePlan(BaseModel):
    id: str
    title: str
    table_type: str
    variables: list[str]
    period: Optional[str] = None
    purpose: str
    file_path: Optional[str] = None
    dataframe_json: Optional[str] = None
    analysis_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage 12-13: Report & fact-check
# ---------------------------------------------------------------------------

class ReportSection(BaseModel):
    heading: str
    body: str
    table_ids: list[str] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)


class FactCheckItem(BaseModel):
    claim: str
    status: Literal["verified", "revised", "removed", "unverifiable"]
    detail: str


class FactCheckResult(BaseModel):
    items: list[FactCheckItem] = Field(default_factory=list)

    @property
    def n_flagged(self) -> int:
        return sum(1 for i in self.items if i.status != "verified")


class ClaimTrace(BaseModel):
    claim: str
    data_source: str
    calculation: Optional[str] = None
    interpretation: Optional[str] = None
    confidence: Confidence = "medium"


class RunLog(BaseModel):
    topic: str
    file_name: str
    started_at: str
    dataset_metadata: Optional[dict[str, Any]] = None
    validation: Optional[dict[str, Any]] = None
    findings: list[dict[str, Any]] = Field(default_factory=list)
    research_plan: Optional[dict[str, Any]] = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    figures: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    fact_check: Optional[dict[str, Any]] = None
    report_path: Optional[str] = None
