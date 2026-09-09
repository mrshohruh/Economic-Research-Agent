# AI Economic Research Agent (V1)

An AI-powered research assistant that takes an XLSX dataset and a research
question, and produces a professional economic research report (DOCX) with
tables, figures, and evidence-based interpretation.

It is built around this logic:

```
XLSX + Research Topic
  -> Understand the data
  -> Validate the data
  -> Analyze quantitatively (Python does the arithmetic, not the LLM)
  -> Detect economically important changes
  -> Build a research plan
  -> Generate hypotheses
  -> Research historical/current policy & news
  -> Collect traceable evidence (no fabricated sources)
  -> Evaluate competing explanations
  -> Plan and generate tables/figures
  -> Write grounded economic analysis for every table/figure
  -> Assemble the report
  -> Fact-check (hedge unsupported causal language)
  -> DOCX report
```

## What was built

A modular Python application with:

- **`data/`** — XLSX ingestion (any sheet layout), variable/date/frequency
  detection, unit inference, and a validation engine (missing values,
  duplicates, structural breaks, irregular frequency).
- **`analysis/`** — a pure pandas/numpy/scipy quantitative engine: descriptive
  stats, MoM/QoQ/YoY changes, trend/turning-point detection, anomaly (z-score)
  detection, historical comparisons, correlation/lead-lag/regression.
- **`agents/`** — the logical multi-agent pipeline: `quantitative_agent`
  (finds economically important changes and ranks them), `research_planner`
  (builds the research plan + hypotheses *before* any web search),
  `research_agent` (executes searches, links evidence to hypotheses),
  `economist_agent` (applies the right economic framework — inflation,
  labor, external, fiscal — and writes cautious, evidence-graded narrative),
  `visualization_agent`, `report_writer`, `fact_checker`.
- **`research/`** — pluggable web search (DuckDuckGo by default, no API key
  needed), source-quality ranking (Central Bank / stats agency / IMF /
  World Bank / Reuters etc. ranked above generic sources), evidence
  structuring, and policy-timeline extraction.
- **`visualization/`** — a visualization *planner* (decides what's actually
  needed for the topic, not a fixed chart set), a matplotlib figure
  generator, and a pandas table generator. Every table/figure gets a
  grounded, LLM-or-template analysis paragraph referencing the real numbers.
- **`reporting/`** — a `python-docx` report generator producing a formatted
  DOCX with headings, embedded tables/figures, captions, source notes,
  a references section, and page numbers.
- **`models/schemas.py`** — Pydantic schemas used everywhere instead of
  free-form text, so every claim is traceable (Data → Calculation →
  Interpretation → Confidence).
- **`storage/database.py`** — SQLite run log + a JSON snapshot of every run
  under `outputs/runs/`.
- **`app.py`** — the Streamlit UI (topic box, XLSX upload, research option
  checkboxes, live progress through the 10 pipeline stages, human review of
  findings/tables/figures with per-finding exclude checkboxes, DOCX download).
- **`pipeline.py`** — the single orchestrator used by both the UI and tests.
- **`sample_data/generate_synthetic_data.py`** — generates a clearly-labeled
  **synthetic** monthly macro dataset (inflation components, exchange rate,
  policy rate, wages, industrial production, trade) with a deliberate food-
  inflation "bump" so the anomaly-detection logic has something to find.
- **`tests/test_pipeline.py`** — unit tests covering ingestion, date/frequency
  detection, validation, YoY/trend/anomaly analysis, research-plan and
  hypothesis generation, table/figure generation, and fact-checker behavior.

## Central design principle

The system never does `XLSX -> LLM -> summary`. All arithmetic runs in
Python (`analysis/`, `agents/quantitative_agent.py`). The LLM (when
configured) is only used for: refining ambiguous variable definitions,
improving research-plan wording, generating/ranking hypotheses, and writing
the grounded narrative prose around numbers Python already computed. Every
narrative passes through `agents/fact_checker.py`, which softens causal
language ("caused" -> "may have contributed to") whenever the supporting
hypothesis confidence isn't `high`.

**The system runs completely without any API key.** With no
`ANTHROPIC_API_KEY` set, every agent falls back to deterministic,
template-based text generation, so the full V1 pipeline (ingestion through
DOCX) still works end-to-end. Setting the key upgrades narrative quality and
reasoning without changing the pipeline's structure.

## How to run

```powershell
# from the project directory
python -m venv .venv        # optional but recommended
.venv\Scripts\activate
pip install -r requirements.txt

# generate the synthetic test dataset (writes sample_data/synthetic_inflation_data.xlsx)
python sample_data/generate_synthetic_data.py

# run the test suite
python -m unittest discover tests -v

# optional: run the full pipeline once from the CLI (no browser needed) and
# produce outputs/reports/synthetic_inflation_data_report.docx directly
python run_demo.py

# launch the app
streamlit run app.py
```

Then in the browser UI:
1. Paste a research topic, e.g. *"Analyze the recent inflation dynamics and
   identify the main factors behind the decline in headline inflation."*
2. Upload `sample_data/synthetic_inflation_data.xlsx` (or your own XLSX).
3. Click **START RESEARCH** and watch the 10-stage progress list.
4. Review findings/tables/figures, optionally exclude a finding, then
   download the generated DOCX report.

## API keys / configuration

Copy `.env.example` to `.env` and fill in what you have:

```
ANTHROPIC_API_KEY=       # optional — enables LLM-backed writing/reasoning
ANTHROPIC_MODEL=claude-sonnet-5
SEARCH_PROVIDER=duckduckgo   # "duckduckgo" (no key needed) or "none"
```

Nothing is hard-coded; `.env` is git-ignored.

## What remains to be improved (post-V1)

- PDF export (DOCX is implemented; PDF via `reportlab` is a natural follow-on).
- A richer "Regenerate" flow in the UI that re-runs from cached artifacts
  instead of requiring the file to be re-uploaded.
- Swapping DuckDuckGo for a paid, higher-reliability search/news API for
  production use, plus real publication-date extraction (many DuckDuckGo
  results don't return a structured date, which is passed through honestly
  as `None`/"n.d." rather than guessed).
- A persistent RAG/policy-database layer instead of live search per run.
- Multi-language report output (Uzbek/Russian), multi-country support.
- LLM-based, contradiction-aware hypothesis confidence scoring (current
  evidence-to-hypothesis linking is keyword-overlap based when no LLM key
  is set — functional, but cruder than a real semantic match).

## Where things live

| Concern | File |
|---|---|
| Orchestration | `pipeline.py` |
| UI | `app.py` |
| Schemas | `models/schemas.py` |
| Config / API keys | `config/settings.py`, `.env` |
| Quantitative engine | `analysis/*.py`, `agents/quantitative_agent.py` |
| Research plan & hypotheses | `agents/research_planner.py` |
| Web research & evidence | `research/*.py`, `agents/research_agent.py` |
| Economic interpretation | `agents/economist_agent.py` |
| Tables & figures | `visualization/*.py`, `agents/visualization_agent.py` |
| Report assembly | `agents/report_writer.py`, `reporting/*.py` |
| Fact-checking | `agents/fact_checker.py` |
| Run logs | `storage/database.py`, `outputs/runs/*.json` |
| Outputs | `outputs/reports/`, `outputs/figures/`, `outputs/tables/` |
