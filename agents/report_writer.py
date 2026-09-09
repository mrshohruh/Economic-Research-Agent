"""
Stage 21-22: Report Writer agent. Assembles all upstream artifacts into the
final structured report and writes the DOCX via reporting/document_generator.
Writing style follows Section 22 (precise, evidence-based, cautious causal
language) with a deterministic fallback when no LLM is configured.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.economist_agent import evaluate_hypotheses, select_framework
from agents.llm_interface import LLM
from config.settings import REPORTS_DIR
from models.schemas import (DatasetMetadata, Finding, FigurePlan, Hypothesis, PolicyEvent, ReportSection,
                             ResearchEvidence, ResearchPlan, TablePlan, ValidationReport)
from reporting import citations
from reporting.document_generator import generate_docx


def _template_executive_summary(topic: str, findings: list[Finding], n_evidence: int) -> str:
    high = [f for f in findings if f.importance == "high"][:3]
    bullet = " ".join(f"{f.finding}" for f in high) or "No high-importance changes were flagged in this dataset."
    return (f"This report investigates: \"{topic}\". Quantitative analysis of the uploaded dataset identified "
            f"{len(findings)} economically notable finding(s), of which {len(high)} were classified as high "
            f"importance. {bullet} The report cross-references these patterns against "
            f"{n_evidence} piece(s) of external evidence on relevant policy and news developments, evaluates "
            f"competing explanations, and presents the results with the tables and figures needed to support "
            f"each conclusion. Where evidence is limited, this report says so explicitly rather than overstating "
            f"confidence.")


def build_executive_summary(topic: str, findings: list[Finding], n_evidence: int) -> str:
    if LLM.enabled:
        system = ("You are a senior economist writing the executive summary of a research report (120-180 words). "
                   "Be precise, quantitative, and avoid unsupported causal claims.")
        user = f"Topic: {topic}\nKey findings: {[f.finding for f in findings[:6]]}\nEvidence collected: {n_evidence} sources."
        text = LLM.complete(system, user, max_tokens=400)
        if text:
            return text
    return _template_executive_summary(topic, findings, n_evidence)


def build_policy_section_text(policy_events: list[PolicyEvent]) -> str:
    if not policy_events:
        return ("No specific policy actions were identified from the available search results for this period. "
                "This may reflect limited web-search coverage rather than an absence of policy activity, and "
                "should be verified against official sources for a definitive record.")
    lines = ["The following policy developments were identified from official and news sources and may be "
             "relevant to the observed data patterns:"]
    for pe in policy_events[:10]:
        date = pe.date or "date unspecified"
        lines.append(f"- {date} — {pe.institution}: {pe.policy} (channel: {pe.economic_channel}).")
    return "\n".join(lines)


def build_news_section_text(evidence: list[ResearchEvidence]) -> str:
    if not evidence:
        return "No current news items were retrieved for this research run."
    recent = sorted(evidence, key=lambda e: e.source_quality, reverse=True)[:8]
    lines = ["Selected relevant reporting and official releases identified during this research run:"]
    for e in recent:
        lines.append(f"- {e.source_title} ({e.institution or 'source'}, {e.publication_date or 'n.d.'}): "
                      f"{e.snippet[:200]}")
    return "\n".join(lines)


def build_competing_explanations_text(findings: list[Finding], hyp_map: dict[str, list[Hypothesis]],
                                       evidence: list[ResearchEvidence]) -> str:
    parts = []
    for f in findings:
        if f.importance == "low":
            continue
        hyps = hyp_map.get(f.id, [])
        if not hyps:
            continue
        parts.append(f"For the finding \"{f.finding}\":\n" + evaluate_hypotheses(f, hyps, evidence))
    return "\n\n".join(parts) if parts else "No competing-explanation analysis was generated for this dataset."


def build_data_section_text(meta: DatasetMetadata, validation, findings: list[Finding]) -> str:
    primary = next((s for s in meta.sheets if s.sheet_name == meta.primary_sheet), meta.sheets[0] if meta.sheets else None)
    if primary is None:
        return "No usable data sheet was found."
    period = f"{primary.period_start} to {primary.period_end}" if primary.period_start else "period undetermined"
    lines = [
        f"The dataset '{meta.dataset_name}' contains {len(meta.sheets)} sheet(s); the primary analysis sheet "
        f"'{primary.sheet_name}' covers {period} at {primary.frequency or 'undetermined'} frequency "
        f"({primary.n_rows} observations across {primary.n_cols} columns).",
    ]
    if meta.ambiguous_variables:
        lines.append(f"The following variables had unit/definition ambiguity and should be confirmed against "
                      f"the original data source: {', '.join(meta.ambiguous_variables)}.")
    lines.append(f"Quantitative screening of this dataset produced {len(findings)} candidate finding(s) for "
                  f"further investigation, described below.")
    return " ".join(lines)


def build_report(
    topic: str,
    meta: DatasetMetadata,
    validation: ValidationReport,
    findings: list[Finding],
    plan: ResearchPlan,
    hyp_map: dict[str, list[Hypothesis]],
    finding_narratives: dict[str, str],
    evidence: list[ResearchEvidence],
    policy_events: list[PolicyEvent],
    tables: dict[str, tuple[TablePlan, pd.DataFrame]],
    figures: dict[str, FigurePlan],
    fact_check_notes: list[str],
    output_name: str,
) -> Path:
    exec_summary = build_executive_summary(topic, findings, len(evidence))

    intro = (f"This report addresses the research question: \"{topic}\". It follows a structured pipeline: "
             f"data understanding and validation, quantitative analysis, detection of economically important "
             f"changes, a research plan, evidence collection from historical and current policy/news sources, "
             f"economic interpretation, and fact-checking. Framework applied: {select_framework(topic)}-oriented "
             f"analysis.")

    data_section = build_data_section_text(meta, validation, findings)

    trends_body_parts = [finding_narratives.get(f.id, f.finding) for f in findings if f.importance != "low"][:8]
    trends_body = "\n\n".join(trends_body_parts) or "No significant trends were detected."

    drivers_body = ("The following factors were evaluated as potential drivers of the patterns identified above, "
                     "distinguishing demand-side, supply-side, external, and structural channels where the "
                     "evidence allows:\n\n" + trends_body)

    policy_text = build_policy_section_text(policy_events)
    news_text = build_news_section_text(evidence)
    competing_text = build_competing_explanations_text(findings, hyp_map, evidence)

    outlook = ("Based on the evidence gathered, near-term dynamics are likely to continue reflecting the drivers "
               "identified above, absent a material policy or external shock. This assessment carries meaningful "
               "uncertainty given the limitations noted in the fact-check appendix.")

    policy_implications = ("Any policy implications below follow directly from the evidence assessed in this "
                            "report and should be read as illustrative rather than prescriptive:\n" +
                            ("\n".join(f"- Monitor developments related to: {f.finding}" for f in findings
                                       if f.importance == "high") or "- No high-confidence policy implications "
                                                                      "could be drawn from the available evidence."))

    conclusion = (f"In summary, the data show {len([f for f in findings if f.unusual])} economically notable "
                  f"movement(s) relevant to the research question. The most likely contributing factors, based "
                  f"on available evidence, are discussed above; where evidence was insufficient, this report "
                  f"has said so explicitly rather than overstating confidence in a single explanation.")

    all_table_ids = list(tables.keys())
    all_figure_ids = list(figures.keys())

    sections = [
        ReportSection(heading="1. Introduction", body=intro),
        ReportSection(heading="2. Data and Recent Developments", body=data_section,
                      table_ids=[tid for tid in all_table_ids if tid in ("T1", "T2")]),
        ReportSection(heading="3. Key Trends and Patterns", body=trends_body,
                      table_ids=[tid for tid in all_table_ids if tid == "T3"],
                      figure_ids=all_figure_ids),
        ReportSection(heading="4. Main Economic Drivers", body=drivers_body),
        ReportSection(heading="5. Policy Developments", body=policy_text),
        ReportSection(heading="6. Current News and Events", body=news_text),
        ReportSection(heading="7. Assessment of Competing Explanations", body=competing_text),
        ReportSection(heading="8. Economic Outlook and Risks", body=outlook),
        ReportSection(heading="9. Policy Implications", body=policy_implications),
        ReportSection(heading="10. Conclusion", body=conclusion),
    ]

    validation_notes = [i.message for i in validation.issues]
    references = citations.build_reference_list(evidence)

    out_path = REPORTS_DIR / output_name
    return generate_docx(
        title=f"Economic Research Report: {topic}", topic=topic, executive_summary=exec_summary,
        sections=sections, tables=tables, figures=figures, references=references,
        output_path=out_path, validation_notes=validation_notes, fact_check_notes=fact_check_notes,
    )
