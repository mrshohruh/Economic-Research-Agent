"""
Streamlit UI for the AI Economic Research Agent (Section 3).
Run with: streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.llm_interface import LLM
from config.settings import SEARCH_PROVIDER
from pipeline import STAGE_LABELS, run_full_pipeline

st.set_page_config(page_title="AI Economic Research Agent", layout="wide")

st.title("AI Economic Research Agent")
st.caption("Upload an XLSX dataset and a research question. The agent will analyze the data, investigate "
           "possible explanations, gather evidence, and produce a professional research report.")

with st.sidebar:
    st.subheader("System status")
    st.write(f"LLM narrative writing: {'**enabled**' if LLM.enabled else '**disabled** (template fallback active)'}")
    st.write(f"Web search provider: **{SEARCH_PROVIDER}**")
    st.caption("Set ANTHROPIC_API_KEY in a .env file to enable LLM-backed writing and reasoning. "
               "The pipeline runs fully without it, using deterministic templated text.")

if "result" not in st.session_state:
    st.session_state.result = None
if "excluded_findings" not in st.session_state:
    st.session_state.excluded_findings = set()

col1, col2 = st.columns([2, 1])
with col1:
    topic = st.text_area("Research topic / question", height=80,
                          placeholder="e.g. Analyze the recent inflation dynamics and identify the main factors "
                                      "behind the decline in inflation.")
with col2:
    uploaded = st.file_uploader("Upload XLSX", type=["xlsx"])
    st.selectbox("Analysis period", ["Automatic"], disabled=True)

st.markdown("**Research options**")
o1, o2, o3, o4, o5 = st.columns(5)
research_news = o1.checkbox("Research current news", value=True)
research_policy = o2.checkbox("Research historical policies", value=True)
generate_tables = o3.checkbox("Generate tables", value=True)
generate_figures = o4.checkbox("Generate figures", value=True)
fact_check = o5.checkbox("Fact-check report", value=True)

output_language = st.selectbox("Output language", ["English"], index=0)

start = st.button("START RESEARCH", type="primary", use_container_width=True)

progress_area = st.container()

if start:
    if not uploaded:
        st.error("Please upload an XLSX file.")
    elif not topic.strip():
        st.error("Please enter a research topic/question.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        progress_bar = progress_area.progress(0)
        status_lines = progress_area.empty()
        completed: list[str] = []

        def on_progress(idx: int, label: str):
            completed.append(f"{idx}. {label} ✓")
            progress_bar.progress(idx / len(STAGE_LABELS))
            status_lines.markdown("\n".join(completed))

        options = {
            "research_news": research_news, "research_policy": research_policy,
            "generate_tables": generate_tables, "generate_figures": generate_figures,
            "write_report": True,
        }
        try:
            result = run_full_pipeline(tmp_path, topic.strip(), options, progress_cb=on_progress)
            st.session_state.result = result
            st.session_state.excluded_findings = set()
            st.success("Research complete.")
        except Exception as exc:
            st.exception(exc)

result = st.session_state.result

if result:
    st.divider()
    st.header("Human Review")

    st.subheader("Research Plan")
    with st.expander("View research plan", expanded=False):
        for s in result.plan.steps:
            st.markdown(f"- {s}")
        st.markdown("**Search queries used:**")
        st.write(", ".join(result.plan.search_queries))

    st.subheader("Key Data Findings")
    for f in result.findings:
        cols = st.columns([6, 1])
        with cols[0]:
            tag = {"high": "🔴", "medium": "🟡", "low": "⚪"}[f.importance]
            st.markdown(f"{tag} **[{f.importance.upper()}]** {f.finding}")
        with cols[1]:
            excluded = f.id in st.session_state.excluded_findings
            if st.checkbox("Exclude", value=excluded, key=f"excl_{f.id}"):
                st.session_state.excluded_findings.add(f.id)
            else:
                st.session_state.excluded_findings.discard(f.id)

    st.subheader("Research Findings / Evidence")
    if result.evidence:
        ev_df = pd.DataFrame([{
            "Source": e.source_title, "Institution": e.institution, "Date": e.publication_date,
            "URL": e.url, "Confidence": e.confidence,
        } for e in result.evidence])
        st.dataframe(ev_df, use_container_width=True, height=220)
    else:
        st.info("No external evidence was collected (web research disabled or unavailable). The report relies "
                "solely on the dataset.")

    if result.policy_events:
        st.subheader("Policy Timeline")
        pol_df = pd.DataFrame([{
            "Date": p.date, "Institution": p.institution, "Policy": p.policy, "Channel": p.economic_channel,
        } for p in result.policy_events])
        st.dataframe(pol_df, use_container_width=True)

    st.subheader("Proposed Tables")
    for tid, (plan, df) in result.tables.items():
        with st.expander(f"{plan.title}"):
            st.dataframe(df, use_container_width=True)
            if plan.analysis_text:
                st.caption(plan.analysis_text)

    st.subheader("Proposed Figures")
    fig_cols = st.columns(2)
    for i, (fid, plan) in enumerate(result.figures.items()):
        with fig_cols[i % 2]:
            if plan.file_path and Path(plan.file_path).exists():
                st.image(plan.file_path, caption=plan.title)
            if plan.analysis_text:
                st.caption(plan.analysis_text)

    st.subheader("Data Validation Report")
    for issue in result.validation_report.issues:
        icon = {"ok": "✅", "warning": "⚠️", "error": "❌"}[issue.level]
        st.markdown(f"{icon} {issue.message}")

    if fact_check:
        st.subheader("Fact-Check Summary")
        flagged = [i for i in result.fact_check_items if i.status != "verified"]
        st.write(f"{len(result.fact_check_items) - len(flagged)} claim(s) verified, "
                 f"{len(flagged)} claim(s) revised/flagged.")
        with st.expander("View fact-check details"):
            for item in result.fact_check_items:
                st.markdown(f"**{item.status.upper()}** — {item.claim}: {item.detail}")

    st.divider()
    regen_col, dl_col = st.columns(2)
    with regen_col:
        if st.button("Regenerate report with current exclusions"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                # Re-run against the same uploaded file path is not retained across reruns in this simple
                # V1 flow; ask the user to re-upload and click START RESEARCH again if the file was cleared.
                pass
            st.warning("Please click START RESEARCH again -- V1 regenerates from the original inputs plus your "
                       "current exclusions/instructions.")

    with dl_col:
        if result.report_path and Path(result.report_path).exists():
            with open(result.report_path, "rb") as f:
                st.download_button("Download DOCX report", f, file_name=Path(result.report_path).name,
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    use_container_width=True)
