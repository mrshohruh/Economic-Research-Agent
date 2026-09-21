"""Streamlit front end for the Uzbekistan Housing Market Research Agent.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from uzhousing.config import LANGUAGES, Settings  # noqa: E402
from uzhousing.pipeline import run  # noqa: E402

ACCENT = "#2a78d6"
ACCENT_DARK = "#184f95"


# Streamlit executes this file top to bottom, so helpers must be defined before use.
def _fmt(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= threshold:
            return f"{number / threshold:,.2f}{suffix}"
    return f"{number:,.1f}"


def _short(text: str, limit: int = 22) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"

st.set_page_config(
    page_title="Uzbekistan Housing Market Research Agent",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
      h1, h2, h3 {{ letter-spacing: -0.01em; }}
      .hero {{
        border-left: 4px solid {ACCENT};
        padding: 0.1rem 0 0.1rem 1rem;
        margin-bottom: 1.4rem;
      }}
      .hero h1 {{ margin: 0 0 0.3rem 0; font-size: 1.9rem; }}
      .hero p {{ margin: 0; opacity: 0.75; font-size: 0.98rem; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
      .stDownloadButton button {{
        background: {ACCENT}; color: #fff; border: 0; font-weight: 600;
        padding: 0.6rem 1.4rem;
      }}
      .stDownloadButton button:hover {{ background: {ACCENT_DARK}; color: #fff; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Uzbekistan Housing Market Research Agent</h1>
      <p>Upload housing data as JSON, SQL, CSV or Excel. The agent profiles it, analyses trends
      and drivers, researches the policy backdrop, and writes a Word report with figures and tables.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
settings = Settings.from_env()

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Anthropic API key",
        value=settings.anthropic_api_key,
        type="password",
        help="Optional. With a key, the analysis, policy synthesis and report prose are written "
             "by Claude. Without one, the agent falls back to deterministic statistics and "
             "template narrative — the report is still complete.",
    )
    model = st.selectbox(
        "Model",
        ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"],
        index=0,
        help="Opus is the default and gives the best writing; Sonnet is faster and cheaper.",
    )

    st.divider()
    web = st.toggle(
        "Live web research", value=settings.web_research,
        help="Search the web for Uzbek housing policy, mortgage programmes and macro context, "
             "with citations. Adds a few minutes to the run.",
    )
    depth = st.select_slider(
        "Research depth", options=["quick", "standard", "deep"], value="standard",
        disabled=not web,
    )
    language = st.selectbox(
        "Report language",
        options=list(LANGUAGES),
        format_func=lambda code: LANGUAGES[code],
        index=list(LANGUAGES).index(settings.language),
    )

    st.divider()
    title = st.text_input("Report title", value="Uzbekistan Housing Market")

    st.caption(
        "Status: " + ("🟢 Claude enabled" if api_key else "🟡 no API key — template narrative")
    )

DEPTH = {"quick": (4, 1), "standard": (6, 3), "deep": (10, 5)}

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
st.subheader("1 · Data")
tab_upload, tab_sample, tab_db, tab_olx = st.tabs(["Upload a file", "Use sample data", "Connect a database", "OLX.uz avtomatik"])

data_path: Path | None = None
connection_url: str | None = None
query: str | None = None

with tab_upload:
    uploaded = st.file_uploader(
        "JSON, SQL, SQLite, CSV or Excel",
        type=["json", "jsonl", "sql", "db", "sqlite", "sqlite3", "csv", "tsv", "xlsx", "xls"],
        help="Any shape of table works: the agent works out which columns are dates, regions, "
             "prices, volumes, rates and so on.",
    )
    if uploaded is not None:
        temp_dir = Path(tempfile.gettempdir()) / "uzhousing_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        data_path = temp_dir / uploaded.name
        data_path.write_bytes(uploaded.getbuffer())
        st.success(f"Loaded **{uploaded.name}** ({len(uploaded.getbuffer()):,} bytes)")

with tab_sample:
    st.write(
        "A synthetic monthly dataset for 8 Uzbek regions, 2018–2026: prices per m², transaction "
        "counts, completions, plus a macro table with the policy rate, mortgage rates, inflation, "
        "wages and the exchange rate."
    )
    st.caption("Synthetic data for demonstration — not official statistics.")
    sample_json = ROOT / "sample_data" / "uz_housing_sample.json"
    if st.button("Generate / refresh sample data"):
        from sample_data.generate import write_sample, write_sql_sample

        write_sample()
        write_sql_sample()
        st.success("Sample data written to sample_data/")
    if sample_json.exists() and st.checkbox("Use the sample dataset", value=False):
        data_path = sample_json

with tab_db:
    connection_input = st.text_input(
        "SQLAlchemy connection URL",
        placeholder="postgresql://user:password@host:5432/database",
        help="Also accepts mysql+pymysql://, sqlite:///path.db and others. "
             "The relevant driver package must be installed.",
    )
    query_input = st.text_area(
        "SQL query (optional)",
        placeholder="SELECT date, region, avg_price_per_sqm, transactions FROM housing_prices",
        height=90,
    )
    if connection_input.strip():
        connection_url = connection_input.strip()
        query = query_input.strip() or None

# ---------------------------------------------------------------------------
st.subheader("2 · Generate the report")

with tab_olx:
    use_olx = st.checkbox("OLX.uz dan avtomatik yig'ish", value=False)
    olx_pages = st.number_input("Har bir toifa uchun sahifalar", min_value=1, max_value=100, value=5)
    olx_browser = st.checkbox("Brauzer orqali yig'ish", value=True,
                              help="OLX oddiy HTTP so'rovlarini HTTP 403 bilan rad etadi, shuning uchun sahifalar haqiqiy brauzerda ochiladi.")
    olx_archive = st.text_input("Tarixiy arxiv papkasi (ixtiyoriy)", value="",
                                help="Arxivlangan OLX .db fayllari bo'lgan papka. Ko'rsatilsa, hisobotga kvartira sotuvi bo'yicha tarixiy narx qatori qo'shiladi.")
    st.caption("Sotuv va uzoq muddatli ijara: kvartira va hovlilar. Hisobot o'zbek (lotin) tilida, PDF va Word shaklida tayyorlanadi. OLX kirishni cheklasa, sabab ko'rsatiladi.")

ready = data_path is not None or connection_url is not None or use_olx
if not ready:
    st.info("Choose a data source above to enable the run.")

if st.button("Run the analysis", type="primary", disabled=not ready, width="content"):
    settings.anthropic_api_key = api_key.strip()
    settings.model = model
    settings.web_research = web
    settings.language = language
    settings.search_results_per_query, settings.pages_to_read = DEPTH[depth]

    st.session_state.pop("result", None)
    log_box = st.status("Starting…", expanded=True)
    lines: list[str] = []

    def progress(message: str) -> None:
        lines.append(message)
        log_box.update(label=message)
        log_box.write(message)

    try:
        with st.spinner("Working…"):
            if use_olx:
                from uzhousing.report.olx_bulletin import run_olx
                result = run_olx(settings, pages=int(olx_pages), progress=progress,
                                 title="" if title == "Uzbekistan Housing Market" else title,
                                 browser=olx_browser,
                                 archive=olx_archive.strip() or None)
            else:
                result = run(
                    settings=settings, data_path=data_path,
                    connection_url=connection_url, query=query,
                    title=title, progress=progress,
                )
        log_box.update(label="Done", state="complete", expanded=False)
        st.session_state["result"] = result
    except Exception as exc:
        log_box.update(label="Failed", state="error", expanded=True)
        st.error(f"The run failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

# ---------------------------------------------------------------------------
result = st.session_state.get("result")
if result is not None and result.ok:
    st.divider()
    st.subheader("3 · Results")

    if result.pdf_path:
        st.success("O'zbek tilidagi OLX hisoboti tayyor.")
        st.download_button("PDF hisobotni yuklab olish", data=result.pdf_path.read_bytes(),
                           file_name=result.pdf_path.name, mime="application/pdf")
        st.download_button("Word hisobotni yuklab olish", data=result.report_path.read_bytes(),
                           file_name=result.report_path.name,
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        st.caption(f"Jadvallar, grafiklar va metodologiya hisobot ichida. Yig'ish qaydi: {result.run_log}")
        st.stop()

    analysis = result.analysis
    headline = (analysis.brief.get("headline") or {}) if analysis else {}
    coverage = (analysis.brief.get("coverage") or {}) if analysis else {}

    columns = st.columns(5)
    columns[0].metric("Headline indicator", _short(headline.get("label", "—")))
    columns[1].metric(
        "Latest", _fmt(headline.get("latest")),
        delta=f"{headline['yoy_pct']:+.1f}% YoY" if headline.get("yoy_pct") is not None else None,
    )
    columns[2].metric("CAGR", f"{headline['cagr_pct']:+.1f}%" if headline.get("cagr_pct") is not None else "—")
    columns[3].metric("Periods", coverage.get("periods", "—"))
    columns[4].metric("Figures", len(result.figures))

    with open(result.report_path, "rb") as handle:
        st.download_button(
            "⬇  Download the Word report",
            data=handle.read(),
            file_name=Path(result.report_path).name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    st.caption(f"Saved to `{result.report_path}` · generated in {result.duration_seconds:.0f}s")

    tabs = st.tabs(["Figures", "Summary", "Drivers", "Policy & sources", "Data profile", "Notes"])

    with tabs[0]:
        for figure in result.figures:
            st.markdown(f"**{figure.id}. {figure.title}**")
            st.image(str(figure.path), width="stretch")
            st.caption(figure.caption)
            if figure.table is not None and len(figure.table):
                with st.expander("Underlying data"):
                    st.dataframe(figure.table, width="stretch")
            st.divider()

    with tabs[1]:
        narrative = result.narrative
        if narrative:
            st.markdown("#### Executive summary")
            for paragraph in narrative.executive_summary:
                st.write(paragraph)
            if narrative.key_findings:
                st.markdown("#### Key findings")
                for finding in narrative.key_findings:
                    st.markdown(f"- {finding}")
            if narrative.recommendations:
                st.markdown("#### Recommendations")
                import pandas as pd

                st.dataframe(
                    pd.DataFrame([
                        {
                            "Priority": r.priority.capitalize(),
                            "Who should act": r.audience,
                            "Action": r.action,
                            "Why": r.rationale,
                        }
                        for r in narrative.recommendations
                    ]),
                    width="stretch", hide_index=True,
                )
            st.caption(f"Narrative written by: {narrative.generated_by}")

    with tabs[2]:
        import pandas as pd

        for key in ("drivers", "activity_drivers"):
            block = (analysis.brief.get(key) or {}) if analysis else {}
            links = block.get("links") or []
            if not links:
                continue
            st.markdown(f"#### Drivers of {block.get('target_label', block.get('target'))}")
            st.caption(f"Computed on {block.get('basis', 'growth rates')}.")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Driver": l.get("driver_label") or l.get("driver"),
                        "r": l.get("correlation"),
                        "p": l.get("p_value"),
                        "Best lag": l.get("best_lag"),
                        "Lagged r": l.get("best_lag_correlation"),
                        "Strength": l.get("strength"),
                        "Significant": "yes" if l.get("significant") else "no",
                    }
                    for l in links
                ]),
                width="stretch", hide_index=True,
            )

    with tabs[3]:
        research = result.research
        if research and research.policy_events:
            import pandas as pd

            st.markdown("#### Policy and macro events")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Date": e.date, "Measure": e.title, "Type": e.category,
                        "Effect": e.direction, "Confidence": e.confidence, "Origin": e.origin,
                    }
                    for e in research.policy_events
                ]),
                width="stretch", hide_index=True,
            )
        if research:
            sources = research.source_index()
            if sources:
                st.markdown(f"#### Sources ({len(sources)})")
                for i, source in enumerate(sources, 1):
                    st.markdown(f"{i}. [{source['title']}]({source['url']}) — *{source['domain']}*")
            else:
                st.info("No web sources were retrieved for this run.")

    with tabs[4]:
        if analysis:
            import pandas as pd

            understanding = analysis.understanding
            profile = understanding.primary_profile
            st.markdown(f"#### Primary table: `{understanding.primary}`")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Column": c.name, "Role": c.role, "Unit": c.unit or "—",
                        "Missing %": c.missing_pct, "Distinct": c.unique,
                        "Identified by": c.reason,
                    }
                    for c in profile.columns
                ]),
                width="stretch", hide_index=True,
            )
            st.markdown("#### Sample of the tidied data")
            st.dataframe(understanding.tidy.head(200), width="stretch")

    with tabs[5]:
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)
        else:
            st.success("No warnings were recorded for this run.")
        st.caption(f"Run log: `{result.run_log}`")
