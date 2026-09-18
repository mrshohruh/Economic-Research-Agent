# Uzbekistan Housing Market Research Agent

Give it housing or rental data as a **`.db` SQLite file, JSON or SQL** (CSV, Excel, Parquet, a
whole folder and live databases work too). It reads the data, works out what every variable means,
sets aside the ones that have nothing to do with the housing market, computes the statistics,
researches the policy and macroeconomic backdrop on the web, and writes a **formatted Word report**
with figures, tables, explanations of the trends, and recommendations.

What it is and the rules it works under are specified in [AGENT.md](AGENT.md).

```
python run.py --data sample_data/uz_housing_sample.json
```

---

## What it actually does

| Stage | What happens |
|---|---|
| **1. Ingest** | Parses JSON (nested, column-oriented, multi-table), `.sql` dumps (replayed into SQLite, with MySQL/Postgres syntax cleaned up), SQLite files, CSV, Excel, or a live SQLAlchemy database. Normalises column names and coerces text like `"1 234,5"` and `"12%"` into numbers. Recognises **property-marketplace listing feeds** (OLX-style pages of adverts with their attributes in a `params` array) and lifts them into one row per listing — price, currency, region, city, floor area, rooms — converting som and dollar-linked "у.е." onto a single currency and detecting whether the feed is rentals or sales. |
| **2. Understand** | Assigns a semantic role to every column — date, region, segment, price, price per m², transaction volume, supply, mortgage, rate, income, inflation, FX — using name patterns in **English, Russian and Uzbek** plus value checks. With an API key, Claude reviews and corrects the mapping. Picks the most informative table and reshapes it into a tidy panel, merging metrics from supporting tables. Then writes a **glossary**: what each column actually measures, and — where two columns share a name stem — how their values relate to each other in this dataset. |
| **3. Screen** | Decides which of those variables belong in a housing analysis at all, and **drops the ones that do not** before a single statistic is computed. A scraped marketplace feed carries advert view counts, seller ratings, photo counts, promotion flags, map coordinates and record keys alongside the price; left in, they become candidate drivers and eventually a sentence about how user ratings rose alongside prices. Prices, rents, areas, transaction counts, completions, credit, rates, incomes, inflation, FX and population stay. Empty, constant and per-row-identifier columns go too. The screen is conservative — a recognised housing indicator survives anything short of an unambiguous match, and if everything would be dropped the screen is abandoned instead. Decisions are logged to the console and the run log, never to the report. |
| **4. Analyse** | **Cross-section** (property microdata): median and mean price by region and city, price per m², price by dwelling size and state of repair, percentiles and skew, which region is most expensive and by what ratio, with thin samples flagged rather than silently ranked. **Time series**: levels, period and year-on-year growth, YTD, CAGR, volatility, drawdown from peak; linear trend test, STL seasonal decomposition, structural-break detection, turning points, Holt-Winters projection; regional ranking, dispersion, σ-convergence and concentration. Every number in the report is computed here by pandas — the language model interprets these tables, it never produces the figures in them. |
| **5. Explain** | Tests every other indicator against the headline series **and** against the main activity series, on year-on-year growth rates, with lead/lag scanning, then fits a multivariate OLS regression. Excludes same-family and collinear regressors so the coefficients mean something. |
| **6. Research** | Searches the web (DuckDuckGo, no key needed) across nine themes — market state, housing policy, mortgage programmes, monetary policy, macro drivers, rental demand and its seasonality, construction costs, risks, regional dynamics — in English, Russian and Uzbek. Official government and central bank announcements outrank news outlets, which outrank brokerage blogs. Downloads the best pages and, with an API key, synthesises them into cited findings and a dated policy timeline. |
| **7. Report** | Renders up to 15 charts and writes a Word document: cover page, table of contents, 13 numbered sections, numbered figures and tables, a policy table, a recommendations table, sources with working hyperlinks, and page numbers. |

---

## Quick start

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 2. (Optional) add your API key
copy .env.example .env            # then paste your key into ANTHROPIC_API_KEY

# 3. Try it on the sample data
python run.py --make-sample
python run.py --data sample_data/uz_housing_sample.json
```

The report lands in `outputs/reports/`, the charts in `outputs/figures/`, and a full JSON
record of the run in `outputs/runs/`.

### Web interface

```bash
streamlit run app.py
```

Upload a file, pick your settings, watch the analysis run, browse the figures, and download the
`.docx`.

---

## Using your own data

**Nothing needs to be in a particular shape.** The agent inspects whatever you give it. These all
work:

<details>
<summary>A flat list of records</summary>

```json
[
  {"date": "2024-01", "region": "Tashkent", "avg_price_per_sqm_usd": 1180, "transactions": 640},
  {"date": "2024-02", "region": "Tashkent", "avg_price_per_sqm_usd": 1195, "transactions": 705}
]
```
</details>

<details>
<summary>Several named tables in one file</summary>

```json
{
  "regional_housing": [ {...}, {...} ],
  "macro_indicators": [ {...}, {...} ]
}
```
Metrics from the second table are merged in automatically, so mortgage rates and wages become
available as candidate drivers of prices.
</details>

<details>
<summary>Column-oriented, or wrapped in a payload</summary>

```json
{"data": {"date": ["2024-01", "2024-02"], "price": [1180, 1195]}}
```
</details>

<details>
<summary>A SQL dump</summary>

```sql
CREATE TABLE prices (date TEXT, region TEXT, price_per_sqm REAL);
INSERT INTO prices VALUES ('2024-01-01', 'Samarkand', 640);
```
Run with `--data dump.sql`. MySQL and PostgreSQL dumps are cleaned up before replay.
</details>

<details>
<summary>A whole folder of files</summary>

```
Tashkent/
  olx_house_price_2024_1.db
  olx_house_price_2024_2.db
  olx_house_price_2024_3.db
```
Point `--data` at the folder. Every supported file in it (subfolders included) is read, and
tables that share a shape are stacked into a single series covering the whole period. Files that
cannot be read are listed in the report rather than stopping the run.

Millions of rows will not fit in memory at once, so above `--max-rows` (400,000 by default) the
loader reads every *n*th row instead, spread evenly across the data, and the report states the
sample size. Averages, shares and trends are unaffected; counts and totals scale down with the
sample. Use `--max-rows 0` to read everything.
</details>

Russian and Uzbek column names are recognised directly — `narx`, `viloyat`, `sana`, `цена`,
`область`, `дата`, `ипотека`, `ставка` and many others.

### Command line

```bash
python run.py --data FILE            # JSON, SQL, SQLite, CSV, Excel
python run.py --data FOLDER          # every supported file in it, stacked into one dataset
python run.py --data FOLDER --max-rows 0   # read every row (default budget: 400,000)
python run.py --db URL --query SQL   # any SQLAlchemy database
python run.py --data FILE --no-web   # skip web research (offline, fully reproducible)
python run.py --data FILE --lang ru  # report in Russian (en | ru | uz)
python run.py --data FILE --title "Tashkent Primary Market Review"
python run.py --data FILE --model claude-opus-5
python run.py --make-sample          # write a demo dataset and exit
python run.py --help                 # everything else
```

---

## Does it need an API key?

**No.** The pipeline runs end to end without one and still produces a complete report with every
chart, every table and every computed statistic. What you lose is the written analysis: the
narrative comes from a deterministic template instead, and web sources are presented as an
organised, ranked evidence digest rather than a synthesis.

With `ANTHROPIC_API_KEY` set, Claude additionally:

- reviews and corrects the inferred column mapping,
- explains in plain language what each column measures,
- reviews which variables are relevant to the housing market and which should be set aside,
- reads the retrieved pages and writes a cited policy and macro synthesis,
- builds a dated policy timeline with expected transmission channels,
- writes all thirteen report sections, including the reasoning behind each trend and the
  recommendations.

The report always states which mode produced it, in section 2.

---

## Keeping the policy knowledge current

`knowledge/policy_events.json` is a plain, editable record of measures affecting the market. The
agent merges it with what it finds on the web and labels each entry by confidence. Add your own
entries — the file documents its own schema, and every field is explained inline.

---

## Layout

```
AGENT.md                     what the agent is, and the rules it works under
run.py                       CLI entry point
app.py                       Streamlit interface
knowledge/policy_events.json editable policy record
sample_data/generate.py      synthetic demo dataset (JSON + SQL)
uzhousing/
  config.py                  settings from .env
  llm.py                     Anthropic wrapper, degrades cleanly without a key
  pipeline.py                orchestration
  cli.py                     argument parsing
  ingest/
    loader.py                JSON / SQL / SQLite / CSV / Excel / database
    profiler.py              semantic column roles, date parsing, tidying
    glossary.py              what each column actually measures
    relevance.py             drops variables unrelated to the housing market
    listings.py              marketplace advert feeds -> one row per listing
    translate.py             readable labels from ru / uz / en column names
  analysis/
    metrics.py               levels, growth, CAGR, volatility
    timeseries.py            trend, seasonality, breaks, turning points, forecast
    regional.py              ranking, dispersion, convergence, concentration
    drivers.py               correlation, lead/lag, regression attribution
  research/
    websearch.py             DuckDuckGo search + page reading, cached
    knowledge.py             research agenda and local policy record
    context.py               synthesis into cited findings
  viz/
    theme.py                 palette, matplotlib defaults, drawing helpers
    charts.py                the chart factory
  report/
    narrative.py             LLM writer + deterministic fallback
    docx_builder.py          Word primitives (tables, figures, TOC, hyperlinks)
    composer.py              assembles the 13-section report
tests/test_pipeline.py       end-to-end and unit tests
```

---

## Report contents

1. Executive summary and key findings
2. Data and methodology — including the inferred column roles, so the reader can check the agent's interpretation
3. Current state of the market
4. Historical trends and turning points
5. Regional and segment analysis
6. What is driving the market
7. Policy environment and its transmission
8. Macroeconomic and external context
9. Outlook
10. Recommendations — each naming who should act, what to do, and the evidence behind it
11. Risks
12. Limitations and data quality
13. Sources

---

## Notes on method

- **Variables are understood before they are analysed, and screened before they are computed.**
  Relevance is judged from established meaning, never from a column name alone — which is why the
  glossary runs first and the screen second.
- **Every policy and news claim carries its source URL inline.** A claim that cannot be attributed
  to a retrieved source is not made.
- **Explanations are graded, not uniformly hedged.** A dated policy measure plus a visible response
  in the data is stated as well supported; an explanation that merely fits the timing is labelled a
  hypothesis in those words.
- **Correlations are computed on year-on-year growth rates**, not levels. Two independently
  trending series correlate strongly in levels for no meaningful reason. Overlapping year-on-year
  windows are serially correlated, so p-values are used as a ranking device and the report says so.
- **No dual-axis charts.** Two series on different scales get two stacked panels sharing a time
  axis, or are indexed to a common base.
- **Every figure ships with its numbers**, so nothing depends on reading a colour correctly.
- **The report never claims causation** from a correlation, and labels every attribution as
  statistical evidence, documented policy, or judgement.
- **Charts use a colour-vision-deficiency-validated palette** in fixed slot order.

## Testing

```bash
python -m pytest tests/ -v
```

## Licence

MIT


### Offline property-variable selection

Without an API key, listing analysis inspects the available columns and tests
recognised property attributes against asking price. Numeric attributes use
Spearman rank correlation; absolute correlations of at least 0.15 are exploratory
candidates. Categorical attributes use price medians for groups with at least 15
observations. Both require at least 30 paired rows and 20% price coverage.
Unknown meanings are flagged for review rather than guessed. Price-derived columns
are excluded as predictors. Weak marginal associations do not prove irrelevance,
and none of these comparisons establish causation or predictive value.

Selected comparisons appear in the Word report even without an API key. Every
column decision, including exclusions and reasons, is saved in the run JSON at
brief.cross_section.variable_assessment. This assessment concerns the normalised
listing table; nested feed attributes not extracted by the loader are not assessed.
Known housing measures are no longer rejected merely because their integer values
are all distinct. Property flags are not treated as promotion flags solely because
their names begin with is_ or has_.
