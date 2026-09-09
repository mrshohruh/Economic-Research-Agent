"""
Top-level orchestrator wiring together every stage of the AI Economic
Research Agent (Section 40 dev instructions). Used by both the Streamlit
app and the CLI test/demo script, so the pipeline logic lives in one place.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from agents import fact_checker, report_writer, research_agent, research_planner
from agents.economist_agent import write_finding_narrative
from agents.quantitative_agent import run_quantitative_analysis
from agents.visualization_agent import build_figures, build_tables
from data import ingestion, metadata, validation
from data.ingestion import SheetData
from models.schemas import (DatasetMetadata, Finding, FigurePlan, Hypothesis, PolicyEvent, ResearchEvidence,
                             ResearchPlan, RunLog, TablePlan, ValidationReport)
from storage.database import save_run

logger = logging.getLogger(__name__)

STAGE_LABELS = [
    "Understanding data",
    "Validating data",
    "Quantitative analysis",
    "Building research plan",
    "Researching policies/news",
    "Evaluating evidence",
    "Designing visualizations",
    "Generating tables/figures",
    "Writing report",
    "Fact-checking",
]

ProgressCB = Optional[Callable[[int, str], None]]


@dataclass
class PipelineResult:
    topic: str
    file_name: str
    meta: DatasetMetadata
    validation_report: ValidationReport
    primary_sheet: SheetData
    findings: list[Finding]
    plan: ResearchPlan
    hyp_map: dict[str, list[Hypothesis]]
    evidence: list[ResearchEvidence]
    policy_events: list[PolicyEvent]
    tables: dict = field(default_factory=dict)
    figures: dict = field(default_factory=dict)
    report_path: Optional[Path] = None
    fact_check_items: list = field(default_factory=list)
    correlation_matrix: Optional[pd.DataFrame] = None


def _tick(cb: ProgressCB, idx: int):
    if cb:
        cb(idx, STAGE_LABELS[idx - 1])


def run_full_pipeline(
    file_path: str,
    topic: str,
    options: dict,
    progress_cb: ProgressCB = None,
    excluded_finding_ids: Optional[list[str]] = None,
) -> PipelineResult:
    options = options or {}
    excluded_finding_ids = excluded_finding_ids or []

    # 1. Understanding data
    sheets = ingestion.load_and_analyze(file_path)
    if not sheets:
        raise ValueError("The workbook contains no readable sheets.")
    primary_name = ingestion.choose_primary_sheet(sheets)
    sd = sheets[primary_name]
    meta = metadata.build_dataset_metadata(file_path, sheets, primary_name)
    meta = metadata.refine_definitions_with_llm(meta, topic)
    _tick(progress_cb, 1)

    # 2. Validating data
    val = validation.validate_sheet(sd)
    _tick(progress_cb, 2)

    # 3. Quantitative analysis (+ important-change detection, stage 4)
    quant = run_quantitative_analysis(sd, topic)
    findings = [f for f in quant.findings if f.id not in excluded_finding_ids]
    _tick(progress_cb, 3)

    # 4. Research plan (stage 5) + hypotheses (stage 6)
    plan = research_planner.build_research_plan(topic, findings)
    hyp_map: dict[str, list[Hypothesis]] = {}
    for f in findings:
        if f.importance == "low":
            continue
        hyp_map[f.id] = research_planner.generate_hypotheses(f, topic)
    _tick(progress_cb, 4)

    # 5. Web research (stage 7-8)
    do_news = options.get("research_news", True)
    do_policy = options.get("research_policy", True)
    if do_news or do_policy:
        evidence, policy_events = research_agent.run_research(plan, do_news, do_policy)
    else:
        evidence, policy_events = [], []
    _tick(progress_cb, 5)

    # 6. Evaluating evidence: link to hypotheses, write narratives
    for fid, hyps in hyp_map.items():
        research_agent.link_evidence_to_hypotheses(hyps, evidence)
    finding_narratives: dict[str, str] = {}
    for f in findings:
        if f.importance == "low":
            continue
        finding_narratives[f.id] = write_finding_narrative(f, hyp_map.get(f.id, []), evidence, topic)
    _tick(progress_cb, 6)

    # 7. Designing visualizations (plan only happens inside build_tables/build_figures)
    _tick(progress_cb, 7)

    # 8. Generating tables/figures (+ per-item economic analysis text, stage 12)
    tables = build_tables(topic, sd, findings) if options.get("generate_tables", True) else {}
    figures = build_figures(topic, sd, findings) if options.get("generate_figures", True) else {}
    _tick(progress_cb, 8)

    # 9. Writing report
    # Fact-check narratives before embedding them (stage 23), softening
    # causal language where hypothesis confidence isn't high.
    narrative_confidence = {}
    for f in findings:
        if f.id not in finding_narratives:
            continue
        hyps = hyp_map.get(f.id, [])
        best_conf = "low"
        if any(h.confidence == "high" for h in hyps):
            best_conf = "high"
        elif any(h.confidence == "medium" for h in hyps):
            best_conf = "medium"
        narrative_confidence[f.id] = (finding_narratives[f.id], best_conf)

    revised_narratives, fc_result = fact_checker.run_fact_check(findings, evidence, narrative_confidence)
    for fid, text in revised_narratives.items():
        finding_narratives[fid] = text

    report_path = None
    if options.get("write_report", True):
        report_path = report_writer.build_report(
            topic=topic, meta=meta, validation=val, findings=findings, plan=plan, hyp_map=hyp_map,
            finding_narratives=finding_narratives, evidence=evidence, policy_events=policy_events,
            tables=tables, figures=figures,
            fact_check_notes=[f"{i.status.upper()}: {i.detail}" for i in fc_result.items if i.status != "verified"]
            or ["All checked claims were verified as directly traceable to the dataset or a cited source."],
            output_name=f"{Path(file_path).stem}_report.docx",
        )
    _tick(progress_cb, 9)

    # 10. Fact-checking (final pass summary already computed above)
    _tick(progress_cb, 10)

    result = PipelineResult(
        topic=topic, file_name=Path(file_path).name, meta=meta, validation_report=val, primary_sheet=sd,
        findings=findings, plan=plan, hyp_map=hyp_map, evidence=evidence, policy_events=policy_events,
        tables=tables, figures=figures, report_path=report_path, fact_check_items=fc_result.items,
        correlation_matrix=quant.correlation_matrix,
    )

    run_log = RunLog(
        topic=topic, file_name=result.file_name, started_at=datetime.now().isoformat(timespec="seconds"),
        dataset_metadata=meta.model_dump(), validation=val.model_dump(),
        findings=[f.model_dump() for f in findings], research_plan=plan.model_dump(),
        evidence=[e.model_dump() for e in evidence],
        figures=[f.model_dump() for f in figures.values()],
        tables=[t.model_dump() for t, _ in tables.values()],
        fact_check=fc_result.model_dump(), report_path=str(report_path) if report_path else None,
    )
    save_run(run_log)

    return result
