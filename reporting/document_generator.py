"""
Generates the final professionally formatted DOCX report using python-docx:
title, executive summary, numbered headings, embedded tables, embedded
figures with captions, source notes, references, and page numbers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from models.schemas import FigurePlan, ReportSection, TablePlan

NAVY = RGBColor(0x1B, 0x2A, 0x4A)


def _add_page_numbers(doc: Document) -> None:
    section = doc.sections[0]
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar"); fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar"); fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin); run._r.append(instr); run._r.append(fld_end)


def _style_heading(doc: Document, text: str, level: int = 1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = NAVY
    return h


def _insert_table(doc: Document, plan: TablePlan, df: pd.DataFrame):
    doc.add_paragraph(plan.title, style="Intense Quote")
    if df.empty:
        doc.add_paragraph("(No data available for this table.)")
        return
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr[i].text = str(col)
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, col in enumerate(df.columns):
            val = row[col]
            if isinstance(val, float):
                cells[i].text = f"{val:,.2f}"
            else:
                cells[i].text = "" if pd.isna(val) else str(val)
    note = doc.add_paragraph()
    note.add_run(f"Period: {plan.period or 'n/a'}. Source: computed from the uploaded dataset.").italic = True
    if plan.analysis_text:
        doc.add_paragraph(plan.analysis_text)
    doc.add_paragraph()


def _insert_figure(doc: Document, plan: FigurePlan):
    if plan.file_path and Path(plan.file_path).exists():
        doc.add_picture(plan.file_path, width=Inches(6.0))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.add_run(f"{plan.title}").bold = True
    if plan.analysis_text:
        doc.add_paragraph(plan.analysis_text)
    doc.add_paragraph()


def generate_docx(
    title: str,
    topic: str,
    executive_summary: str,
    sections: list[ReportSection],
    tables: dict[str, tuple[TablePlan, pd.DataFrame]],
    figures: dict[str, FigurePlan],
    references: list[str],
    output_path: str | Path,
    validation_notes: list[str] | None = None,
    fact_check_notes: list[str] | None = None,
) -> Path:
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    title_p = doc.add_heading(title, level=0)
    for r in title_p.runs:
        r.font.color.rgb = NAVY
    doc.add_paragraph(f"Research question: {topic}").italic = True
    doc.add_paragraph()

    _style_heading(doc, "Executive Summary", level=1)
    doc.add_paragraph(executive_summary)

    for section in sections:
        _style_heading(doc, section.heading, level=1)
        if section.body:
            doc.add_paragraph(section.body)
        for tid in section.table_ids:
            if tid in tables:
                plan, df = tables[tid]
                _insert_table(doc, plan, df)
        for fid in section.figure_ids:
            if fid in figures:
                _insert_figure(doc, figures[fid])

    if validation_notes:
        _style_heading(doc, "Appendix A: Data Validation Notes", level=1)
        for n in validation_notes:
            doc.add_paragraph(n, style="List Bullet")

    if fact_check_notes:
        _style_heading(doc, "Appendix B: Fact-Check Notes", level=1)
        for n in fact_check_notes:
            doc.add_paragraph(n, style="List Bullet")

    _style_heading(doc, "References", level=1)
    if references:
        for r in references:
            doc.add_paragraph(r, style="List Number")
    else:
        doc.add_paragraph("No external sources were retrieved during this research run "
                           "(web research disabled or unavailable). All findings above are based solely on the "
                           "uploaded dataset.")

    _add_page_numbers(doc)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    return output_path
