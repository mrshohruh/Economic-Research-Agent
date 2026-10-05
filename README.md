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

---

## How to run

Every command below assumes the project's virtualenv. On Windows that is
`.venv\Scripts\python.exe`; after `.venv\Scripts\activate` plain `python` works too.

### Everything at once

The full run: both marketplaces, plus the historical trend.

```powershell
.venv\Scripts\python.exe run.py --olx --olx-browser --lang uz --olx-archive "C:\Users\<you>\Desktop\olx\price_database"
```

This collects **every listing both marketplaces publish**. Measured on
2026-09-22 that is 117,225 on OLX (22,648 apartment rentals, 60,017 apartment
sales, 2,261 house rentals, 32,299 house sales) and 11,540 on Uybor, so allow
**three to four hours**, almost all of it OLX. Requests stay paced two seconds
apart throughout; the time is the pacing, not the parsing.

It writes a PDF and a Word report to `outputs/reports/`. **This mode needs
`ANTHROPIC_API_KEY`**: the bulletin's commentary is written by Claude from the
computed tables, and a missing key stops the run before any collection starts
rather than quietly falling back to a template. The rest of this section
explains each part, and every flag can be dropped independently: without
`--olx-archive` there is no trend section, and `--no-uybor` collects from OLX
alone.

Neither marketplace needs a page count. For a quick partial run instead of the
full sweep, `--olx-pages N` and `--uybor-pages N` cap the collection; a capped
OLX run takes minutes but reaches at most about 1,040 listings per category, for
the reason explained next.

### Collect from OLX.uz and write the Uzbek bulletin

```powershell
.venv\Scripts\python.exe run.py --olx --olx-browser --lang uz
```

`--olx-browser` is required. Plain HTTP is refused by the site's CDN firewall
with HTTP 403 before any page is served, so collection runs through a real
browser. Install it once with `pip install playwright`; the system Chrome is
used when present, otherwise run `python -m playwright install chromium`.

#### Why a category needs more than one query

OLX refuses a paging offset beyond 1000, so **any single query hands over at
most about 1,040 listings**, however many pages are asked for. Apartments for
sale alone had 60,017 listings when this was written, so that category simply
cannot be read as one query — which is why capped runs looked as though the site
only had a thousand ads per category.

The limit is on the query, not on the category: a narrower query gets its own
full window. So an uncapped run asks whether a query reaches past its own window
— it requests the last page the window serves, and a page that comes back full
means listings remain out beyond it — and splits any query too large to page
through into smaller ones until each part fits:

```
category  ->  region  ->  city  ->  district  ->  price band (halved as needed)
```

Those splits are the marketplace's own structure, read from its geo lists rather
than hard-coded, so every listing falls in exactly one part and none is skipped.
Listings are de-duplicated, because one sitting exactly on a price boundary
appears in both neighbouring bands. The snapshot records, per category, how many
listings were collected and how many queries it took (`coverage[].queries`).
There is no site-wide total beside them: the only endpoint that publishes an
unclamped count is one robots.txt asks crawlers to leave alone, so the agent
reports what it read rather than a share of a number it may not request.

`--olx-pages N` cuts a run short at N pages of a single unsplit query per
category. It is the fast path for a smoke test, not a smaller version of the
full sweep: values below about 5 leave most regional medians blank, because a
median is suppressed under 15 observations.

### Add historical price trends

```powershell
.venv\Scripts\python.exe run.py --olx --olx-browser --lang uz --olx-archive "C:\Users\<you>\Desktop\olx\price_database"
```

Turns the bulletin from a single-date snapshot into a quarterly review. From a
folder of archived OLX `.db` files it adds, for apartment sales:

- **quarter-on-quarter comparison tables** by region and by Tashkent district,
  for the primary and secondary markets separately — two dated levels and the
  change into each, in millions of so'm per m²;
- an **asking-price index** with the first observed quarter set to 100;
- the **monthly trend chart**, with unobserved months left as visible breaks.

The archive is read once and cached as three small tables, so later runs start
immediately. Pass `--olx-rebuild-history` after adding new files to the archive.
See [Historical price trends from an archive](#historical-price-trends-from-an-archive)
for what the series does and does not cover.

### Add a second marketplace

```powershell
.venv\Scripts\python.exe run.py --olx --olx-browser --lang uz
```

Uybor.uz listings are collected by default and pooled into the same medians;
`--no-uybor` leaves them out. Each page is up to 100 listings and every page is
read. Uybor publishes no deep-paging limit — 11,540 listings over 116 pages when
this was written, all of them reachable — so no splitting is needed there and
the run collects the whole catalogue. Uybor needs no browser. See
[Second source: Uybor.uz](#second-source-uyboruz) for what is collected and how
pooling is disclosed.

### Rebuild a report without touching the network

```powershell
.venv\Scripts\python.exe run.py --olx-snapshot outputs\olx_snapshots\SNAPSHOT.json --lang uz
```

Every successful collection is archived, so a report can be regenerated from a
saved snapshot at any time. This is the command to use while adjusting the
report itself, and it works offline.

### Analyse your own dataset

```powershell
.venv\Scripts\python.exe run.py --data data.json                 # JSON, SQL, SQLite, CSV, Excel
.venv\Scripts\python.exe run.py --data C:\data\Tashkent          # a whole folder, stacked
.venv\Scripts\python.exe run.py --data C:\data\Tashkent --max-rows 0   # read every row
.venv\Scripts\python.exe run.py --db "postgresql://user:pw@host/db" --query "SELECT * FROM housing"
```

This is the full eight-stage pipeline: profiling, variable screening,
statistics, driver regressions, web research and a Word report. The default row
budget is 400,000 and thins larger sources evenly, which is fine for averages
and trends but makes counts read low; `--max-rows 0` reads everything.

### The web interface

```powershell
.venv\Scripts\streamlit.exe run app.py
```

Upload a file, pick your settings, watch the analysis run, browse the figures
and download the `.docx`. The **OLX.uz avtomatik** tab runs collection instead.
There, **Barcha sahifalarni yig'ish** is ticked by default and collects every
listing, exactly as the uncapped CLI run does; unticking it reveals a page cap
for a quick partial run. Browser collection is on by default, and the archive
folder field is prefilled from `OLX_ARCHIVE` in `.env` so it need not be pasted
each time.

### When something fails

```powershell
.venv\Scripts\python.exe run.py --olx --olx-browser --olx-show-browser --olx-pages 1 --lang uz
.venv\Scripts\python.exe run.py --data data.json --verbose
```

`--olx-show-browser` opens a visible Chrome window, which shows whether OLX is
serving a challenge page. `--verbose` prints the full traceback instead of a
one-line message.

A run that stops with `OLX hisoboti uchun ANTHROPIC_API_KEY talab qilinadi` has
no key in the environment; a full sweep fails on this at the start rather than
after hours of collection, so it is worth a `--olx-pages 1` smoke test before a
long run.

### Unattended runs

Scheduling is external: the flags above perform one collection and one report.
For a recurring snapshot, point Windows Task Scheduler at
`<project>\.venv\Scripts\python.exe` with arguments
`run.py --olx --olx-browser --lang uz`, and set **Start in** to
the project directory. Run it manually once first. No scheduled task is
installed automatically.

### Where the output goes

| Path | Contents |
|---|---|
| `outputs/reports/` | the PDF and Word reports |
| `outputs/olx_snapshots/` | dated raw JSON from each collection |
| `outputs/olx_history.sqlite` | accumulated listing observations |
| `outputs/price_history_monthly.csv` | cached monthly series from the archive |
| `outputs/tables/` | chart images and their CSV data |
| `outputs/runs/` | a JSON record of every run |

Run `python run.py --help` for the complete flag list.

### Settings from `.env`

`copy .env.example .env` and fill in what applies. Every setting is optional
except the key, which the `--olx` bulletin requires; the `--data` pipeline runs
without one.

| Setting | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` | *(auto-detect)* | which vendor to call first (`groq`, `gemini`, `openrouter`, `openai`, `anthropic`). `--provider` overrides per run. See **[Free LLM providers](#free-llm-providers)** below |
| `LLM_MODEL` | *(provider default)* | the model id for the chosen provider; may carry a prefix such as `groq:openai/gpt-oss-120b`, in which case the prefix selects the vendor |
| `LLM_FALLBACK_ENABLED` | `on` | if `on`, the free-provider fallback chain is tried when the primary provider is rate-limited or unavailable. `--no-llm-fallback` turns it off per run |
| `GROQ_API_KEY` | empty | key for Groq (free, preferred default). Combined with `GROQ_MODEL` (default `openai/gpt-oss-120b`) |
| `GEMINI_API_KEY` | empty | key for Google Gemini (free tier). Combined with `GEMINI_MODEL` (default `gemini-2.5-flash`) |
| `OPENROUTER_API_KEY` | empty | key for OpenRouter. Pin a specific model via `OPENROUTER_MODEL` |
| `ANTHROPIC_API_KEY` | empty | enables Claude's written analysis when the chosen provider is `anthropic`; **required for `--olx` / `--olx-snapshot`** when Claude is the primary |
| `OPENAI_API_KEY` | empty | the same, for the `openai` provider |
| `MODEL` | `claude-opus-5` | legacy model id. If `LLM_PROVIDER` / `LLM_MODEL` are unset, this picks the vendor (`claude-*` → Anthropic, `gpt-*` → OpenAI). `--model` overrides per run. `ANTHROPIC_MODEL` is still read for older `.env` files |
| `OLX_ARCHIVE` | empty | folder of archived OLX `.db` files, used to prefill the archive field in the web interface. The CLI takes the folder as `--olx-archive`; the folder is read one level deep, so it must be the one holding the `.db` files |
| `UZS_PER_USD` | `12650` | the rate used to put som and dollar-linked "у.е." listings on one currency. Set it to the rate that applied when the data was collected |
| `MAX_ROWS` | `400000` | row budget for `--data`; `0` reads everything |
| `OUTPUT_DIR` | `outputs` | where reports, figures, snapshots and run logs are written |
| `REPORT_LANGUAGE` | `en` | default report language (`en` \| `ru` \| `uz`); `--lang` overrides it |
| `WEB_RESEARCH` | `on` | `off` skips the web research stage, as `--no-web` does per run |
| `SEARCH_RESULTS_PER_QUERY` | `6` | search results considered per research question |
| `PAGES_TO_READ` | `3` | best pages actually downloaded per research theme |

The `--olx` bulletin is always written in Uzbek Latin; `REPORT_LANGUAGE` and
`--lang` apply to the `--data` pipeline's report.

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
python run.py --data FILE --model claude-sonnet-5   # cheaper than the claude-opus-5 default
python run.py --make-sample          # write a demo dataset and exit
python run.py --help                 # everything else
```

For OLX collection and the historical trend, see [How to run](#how-to-run).

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

**The `--olx` bulletin is the exception.** Its commentary is Claude-written from
the computed tables, with no template fallback, so `--olx` and `--olx-snapshot`
stop with an error when no key is set. See
[Claude analysis in the OLX/Uybor bulletin](#claude-analysis-in-the-olxuybor-bulletin).

---

## Free LLM providers

The agent supports five LLM providers out of the box, three of which are
free to use with a key:

| Provider | Default model | Key |
|---|---|---|
| **Groq** (preferred default) | `openai/gpt-oss-120b` | https://console.groq.com/keys |
| **Google Gemini** | `gemini-2.5-flash` | https://aistudio.google.com/app/apikey |
| **OpenRouter** | *(pin via `OPENROUTER_MODEL`)* | https://openrouter.ai/keys |
| OpenAI (paid) | legacy `MODEL` id | https://platform.openai.com/api-keys |
| Anthropic (paid) | legacy `MODEL` id | https://console.anthropic.com/ |

All three free providers talk the OpenAI chat-completions protocol under
the hood, so no extra SDK is needed. Point the agent at one with
`LLM_PROVIDER=…` (or `--provider …`), set the matching key, and run.

### Switch to Groq (recommended)

PowerShell (Windows):

```powershell
$env:GROQ_API_KEY  = "YOUR_GROQ_KEY"
$env:LLM_PROVIDER  = "groq"
$env:LLM_MODEL     = "openai/gpt-oss-120b"
python run.py --data sample_data\uz_housing_sample.json
```

Command Prompt (Windows):

```bat
set GROQ_API_KEY=YOUR_GROQ_KEY
set LLM_PROVIDER=groq
set LLM_MODEL=openai/gpt-oss-120b
python run.py --data sample_data\uz_housing_sample.json
```

Or on one command line:

```powershell
python run.py --data sample_data\uz_housing_sample.json ^
              --provider groq --model openai/gpt-oss-120b --api-key YOUR_GROQ_KEY
```

### Switch to Gemini

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_KEY"
$env:LLM_PROVIDER   = "gemini"
$env:LLM_MODEL      = "gemini-2.5-flash"
python run.py --data sample_data\uz_housing_sample.json
```

### Switch to OpenRouter

```powershell
$env:OPENROUTER_API_KEY = "YOUR_OPENROUTER_KEY"
$env:LLM_PROVIDER       = "openrouter"
$env:OPENROUTER_MODEL   = "openai/gpt-4.1-mini"   # any model OpenRouter exposes
python run.py --data sample_data\uz_housing_sample.json
```

### Fallback behaviour

With `LLM_FALLBACK_ENABLED=on` (the default), the agent tries the next free
provider whose key is set when the primary one hits a rate limit, times
out, or is otherwise unavailable. The preferred order is
**Groq → Gemini → OpenRouter → OpenAI → Anthropic**, skipping any provider
without a configured key. A deterministic failure (bad model id, invalid
key, programmer error) is surfaced immediately rather than disguised
behind a fallback. The run log in `outputs/runs/run_*.json` records which
provider and model actually served the run and whether the fallback chain
was used.

Pass `--no-llm-fallback` to pin a provider for one run:

```powershell
python run.py --data sample_data\uz_housing_sample.json ^
              --provider groq --no-llm-fallback
```

### Existing OpenAI / Anthropic setups

Nothing changes for existing `.env` files that pin `MODEL=claude-opus-5`
with `ANTHROPIC_API_KEY=…` (or the equivalent OpenAI pair). The legacy
`MODEL` id continues to pick the vendor when `LLM_PROVIDER` is unset.

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
    olx.py                   bounded OLX collection, dated snapshots
    olx_browser.py           browser transport (plain HTTP is refused)
    olx_client.py            paced HTTP access and schema checks
    olx_partition.py         splits an oversized category into servable queries
    olx_geo.py               the site's own region, city and district lists
    olx_store.py             observation history (SQLite)
    uybor.py                 the second marketplace
    price_history.py         the archive -> monthly, quarterly and district tables
    fx_history.py            quarter-end Central Bank rates, cached
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
    layout.py                the bulletin's visual system (cover, tables, palette)
    olx_bulletin.py          the Uzbek bulletin: what it says, and from what
    olx_narrative.py         Claude's bulletin commentary, with figures masked
tests/                       end-to-end and unit tests, one file per area
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

## Bulletin contents (the `--olx` report)

A different report from the one above: Uzbek Latin, Claude-written analysis, and built from
listings rather than an uploaded dataset. It follows the structure of the
supplied quarterly review, as far as listing data can support it.

| Section | Source | Notes |
|---|---|---|
| Qisqacha xulosa | both | the two or three findings the report stands on, nominal and real |
| Ma'lumotlar sifati tekshiruvi | snapshot | what each screen found and what it did with it |
| Birlamchi uy-joy bozori | archive + snapshot | findings and a change chart; the tables go to the appendix |
| Ikkilamchi uy-joy bozori | archive + snapshot | the same, for the secondary market |
| Bozor turi ko'rsatilmagan | snapshot | adverts with no market field, kept apart rather than assigned |
| Uy-joy narxlari indeksi | archive + official | fixed-weight index, drawn against its inflation-adjusted path |
| Tarixiy taqqoslash | archive | the monthly series, with unobserved months left broken |
| Ijara bozori | snapshot | region and Tashkent-district rents; the archive holds no rentals |
| Toshkent shahri tumanlarida sotuv | snapshot | current district cross-section |
| Ijara rentabelligi (yillik) | snapshot | gross proxy on matched region / property / room strata |
| Makroiqtisodiy sharoit, qurilish va ipoteka | official statistics | inflation, incomes, lending rates and population, dated and cited; what is still missing is named with its publisher |
| Manbalar tarkibi | snapshot | how many listings each source contributed, so a change of mix is visible |
| Metodologiya va manbalar | both | one caveat box, screens, dated rates and coverage |
| Ilova: batafsil jadvallar | both | every detail table, numbered A1, A2, … |

Sections that need an archive say so and fall back to the snapshot alone when
`--olx-archive` is not given; nothing is left silently empty. Each analytical
section carries one chart that makes its point; the full table it was drawn
from is moved to the appendix and pointed at from the section's note.

### Data-quality audit — [`analysis/quality.py`](uzhousing/analysis/quality.py)

Every run screens the pooled cross-section before a single table is built, and
prints what it found as a section of its own:

| Screen | What it looks for | What it does |
|---|---|---|
| Place names | a region no canonical list knows | keeps the advert, drops it from regional medians |
| Cross-source duplicates | the same price, area, rooms and location on both sites | counts the dwelling once |
| Repeats within one site | identical adverts under different ids | reports them, keeps them: reposting is normal |
| Room counts | fewer than one or more than twelve, or fractional | clears the field, keeps the advert |
| Sale/rent classification | a price per m² outside its own category's plausible band | drops the per-m² figure, keeps the advert |
| Distribution tails | more than four median absolute deviations into the log tail of its own group | drops the per-m² figure |
| Yields | gross yields outside 1–25 % | holds the stratum back as a mismatched comparison |

Nothing is removed silently: each screen reports its count and its decision,
and the run log records all of them under `quality`.

### Official statistics — [`research/official.py`](uzhousing/research/official.py)

Mortgage lending, construction, household income, population and interest
rates are not on a classifieds site, and "not available" tells the reader
nothing. Inflation, lending rates, credit depth, investment, GNI per capita,
population and urbanisation are read from the World Bank's open API, dated and
cited; the annual inflation series also deflates the price index, so the report
can say whether housing outran consumer prices or merely kept up with them.

Indicators only the Uzbek institutions publish are listed in
[`knowledge/official_indicators.json`](knowledge/official_indicators.json) with
the publication that carries them — Central Bank bank statistics for mortgage
stock and rates, the Statistics Agency for dwellings commissioned, construction
volumes, incomes and registered transactions. Fill in a `value`, `unit` and
`period` there and the figure joins the table with its own citation; leave it
`null` and the report names the source to consult. Values are cached for a week,
and an offline run reports what it has rather than failing.

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
- **The headline index holds its strata weights fixed.** Advert volumes move between regions and
  segments far faster than prices do; weighting a quarter by its own volumes would report that
  movement as a price change.
- **A caveat is stated once.** Everything that governs the whole bulletin stands in one box in the
  methodology section; section notes carry only what is specific to their own table.
- **The prose is read by an editorial pass before the file is written**, and what it still objects
  to is recorded in the run log rather than quietly shipped.

## Testing

```bash
python -m pytest tests/ -v                      # everything, offline
python -m pytest tests/test_olx_partition.py -v # the query-splitting walk alone
```

The suite touches no network: OLX, Uybor and the Central Bank are stubbed, and
the archive tests build their own `.db` files.

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


## OLX: automatic collection and Uzbek bulletin

The commands are in [How to run](#how-to-run); this section describes what the
mode produces and the limits it works under.

In the app, open **OLX.uz avtomatik**, select automatic collection and run the
analysis. This mode writes Uzbek Latin PDF and Word reports and requires an
Anthropic key for Claude-written findings. The language selector applies to the original dataset
analysis mode. Reports are in `outputs/reports`, source snapshots in
`outputs/olx_snapshots`, and chart data in `outputs/tables`.

The supplied Q3 review is the **structure and design reference**: the bulletin
follows its cover, contents page, centred section headings, bulleted commentary,
banded tables with direction-coloured change columns, tinted source and note
callouts, and circled page numbers. Not one figure is taken from it. Sections it
contains that listing data cannot support — housing completions, mortgage
volumes, rates and bank product terms — are named as unavailable rather than
filled in.

Primary/secondary classification requires explicit listing fields. Missing data
and thin samples are disclosed. Region and district names are folded onto one
spelling per place so the three sources pool into one row; the original spelling
stays in the snapshot.

OLX collection can stop if the site changes or returns 403/429. No API key,
login bypass or paid service is used. The current CBU USD rate is fetched and
archived with each successful snapshot.

For unattended runs see [Unattended runs](#unattended-runs). A daily trigger
archives snapshots and generates reports while the machine is on. No scheduled
task is installed automatically.


### Browser-based collection (required)

Plain HTTP requests to www.olx.uz are refused by the site's CDN firewall with
HTTP 403 before any page is served, including `/robots.txt`. Collection therefore
runs through a real browser, which is served the same public pages a visitor sees:

```powershell
.venv\Scripts\python.exe -m pip install playwright
.venv\Scripts\python.exe -m playwright install chromium   # skip if Chrome is installed
.venv\Scripts\python.exe run.py --olx --olx-browser --lang uz
```

`--olx-browser` uses the system Chrome when present and falls back to Playwright's
own Chromium. Add `--olx-show-browser` to watch the window, which is the first
thing to try if a run is refused. In the app, **Brauzer orqali yig'ish** is on by
default in the **OLX.uz avtomatik** tab.

No login is performed, no CAPTCHA is solved and no access control is bypassed.
A challenge page that does not clear on its own stops the run, exactly as a 403
does. Everything else is unchanged by the switch: the same robots.txt rules,
request pacing, schema validation and snapshot format apply.

Medians below 15 observations are suppressed, so capping a run with a small
`--olx-pages` leaves many regional cells blank; the default, which reads every
page, fills them.


### Historical price trends from an archive

The bulletin is a current snapshot on its own. Given a folder of archived OLX
`.db` files it also reports a monthly asking-price trend:

```powershell
.venv\Scripts\python.exe run.py --olx --olx-browser --lang uz --olx-archive "C:\Users\<you>\Desktop\olx\price_database"
```

One pass over the archive produces three cached tables next to each other:

| File | What it holds |
|---|---|
| `outputs/price_history_monthly.csv` | monthly median by region and market |
| `outputs/price_history_monthly_quarterly.csv` | quarterly median by region and market |
| `outputs/price_history_monthly_quarterly_district.csv` | quarterly median by Tashkent district |

Later runs reuse them in a moment. `--olx-rebuild-history` re-reads the archive,
which is what to pass after adding new files to it. A cache written before one
of the tables existed is detected and rebuilt rather than half-trusted.

**Regions and districts are aggregated separately, from the adverts.** A median
cannot be recovered from a set of medians, so the region figure is not derived
from its districts, nor the quarter from its months.

**A quarter needs a sample to be a quarter.** A region with fewer than 100
adverts in a quarter keeps its row and shows a dash; the section text names it
with its count. Quarters the archive observed for fewer than three months are
listed as partial, because they are not comparable with full ones.

**Scope.** The series covers **apartment sales only**, because the archive holds
no rental adverts: its prices run from about 24,000 to 299,000 y.e., and its
`Tip zhilya` field is a primary/secondary split that applies to sales. Rent
therefore stays a current-snapshot section with no historical comparison.

**Currency.** y.e. is treated as the USD, matching the live collector.
Som-quoted adverts are excluded rather than converted, because a 2022 som price
needs a 2022 rate and the archive carries no dated rate. Files whose `currency`
column is entirely empty are still used: their price scale is unambiguously
dollar-linked, and the 1,000-5,000,000 USD screen excludes anything else.

To report those dollar prices in so'm, the quarterly tables use the Central
Bank's **rate at the end of each quarter**, fetched once from
`cbu.uz/uz/arkhiv-kursov-valyut/json/USD/<YYYY-MM-DD>/` and cached in
`outputs/cbu_quarter_end_rates.csv`. Two traps are handled rather than left to
bite: the same endpoint accepts `dd.mm.yyyy` but silently answers *today's*
rate for it, so a response is accepted only when the date it carries is the
date that was asked for; and several quarters end on a weekend or an Uzbek
public holiday, when no rate is published, so the search walks back to the last
rate in force and the report prints which date that was. If the rates cannot be
fetched, the historical tables are reported in USD/m² with the reason stated —
never at another quarter's rate.

The index is built on dollar prices while the table levels are in so'm, so the
section states both movements and attributes the difference to the exchange
rate rather than leaving it to read as a second price movement.

**Gaps are never filled.** Months with no observation break the line in the
chart, are shaded and labelled, and are listed in the text. A comparison across
a break is reported as the change between two dated points, not as a trend, and
the report says which months were not observed. When the current month rests on
far fewer listings than the historical month it is compared with, the report
says so, because part of such a difference can be sample composition rather than
price.

### Second source: Uybor.uz

Residential listings from Uybor.uz are collected alongside OLX by default,
through the public JSON API the site's own pages call. Every page of up to 100
listings is read until the API runs out; `--uybor-pages N` stops after N of
them, and `--no-uybor` skips the source entirely:

```powershell
.venv\Scripts\python.exe run.py --olx --olx-browser --lang uz
```

Unlike OLX it is reachable over plain HTTP, so no browser is needed for it.
`uybor.uz/robots.txt` disallows only `/admin/`, `/cgi-bin/` and `/tmp/`, and is
checked on every run; the API host publishes no robots.txt, which declares no
restriction. Requests are paced as elsewhere, and 401/403/429 stops the run.

Only residential categories are kept (apartment, house, private house). Land,
offices, warehouses and whole businesses are advertised there too and are not
housing. Two traps are handled explicitly: `priceType` is `sqm` on some
listings, where `price` is per square metre rather than a total, and `sot`
on others, where it is per sotka; and rent is only kept where
`pricePeriodUnit` is `month`. Seller identity, contacts and photos are dropped
before anything is written to a snapshot.

**Pooled figures.** Every median in the report pools both sources. Because the
two sites carry different populations of advertisers, a **MANBALAR TARKIBI**
section reports how many listings each source contributed to each segment, so a
change in the mix is visible rather than read as a price movement. The same
listing advertised on both sites can be counted twice: there is no identifier
shared between them, and none is invented.

**Region names are canonicalised.** OLX names regions in Russian and Uybor in
Latin Uzbek, so pooling needs one spelling per region or the same place becomes
two rows of every table. Names are mapped to the Latin Uzbek spelling the
archive uses, and Tashkent city is separated from the surrounding region by the
city name. The original spelling from each source stays in the snapshot; only
the grouping key is unified.

### OLX collection client and observation history

Collection uses `uzhousing/ingest/olx_client.py` for paced HTTP access. The
collector validates response IDs and schema; 404/410 detail responses are
counted as unavailable, never classified as sold, and 401/403/429 stop
collection without retries or any access-control workaround.

Category pages are client-rendered, so the collector first establishes the
category's own numeric ID by reading it from the listings the page actually
shows. The ID is never guessed: it is accepted only when independent listings
agree on it. With an ID established, the bounded, deduplicated
`/api/v1/offers/` pagination returns about forty full records per request
instead of one request per listing, and the snapshot records which strategy
each category used under `coverage[].strategy`. Where no ID can be verified,
collection falls back to reading listing pages directly.

OLX validates the paging offset and refuses anything beyond 1000 with HTTP 400,
so one query yields at most about 1,040 listings however many pages are asked
for. That is a limit on the query, not on the category. An uncapped run
therefore reads a category as many narrower queries — region, then city, then
district, then halved price bands. A query is judged too large by asking
`/api/v1/offers/` for the last page its window reaches: a full page means there
is more behind it. The site's own search metadata would give an exact total, but
robots.txt disallows `*/api/v1/offers/metadata/` and the agent obeys the file as
published on the day of the run, so that endpoint is never requested.
`uzhousing/ingest/olx_partition.py` holds that walk and `uzhousing/ingest/
olx_geo.py` reads the region, city and district lists from the site's
geo-encoder.

A part that is still too large after every available split is read as far as the
window allows and recorded under `coverage[].unreachable`, so a shortfall is
always stated rather than hidden — as a fact, not a count, since the number that
stayed out of reach cannot be asked for. A
capped `--olx-pages N` run stops at the offset limit instead, recorded as
`depth_limit` in `coverage[].stop`; that is a normal stop, not a failure.
A mid-run 401/403/429 still stops the run, and is never mistaken for the cap.

Successful collections write a JSON snapshot for the report and append
observations to `outputs/olx_history.sqlite` (or the selected output folder).
`collection_runs` stores coverage and FX metadata; `listing_snapshots` stores
housing fields and price/currency for each collection timestamp;
`listing_observation_history` exposes first_seen, last_seen and observation
count. These are observed dates, not confirmed time on market or transaction
dates. Missing listings in bounded crawls are not marked sold or removed.
Reimporting one snapshot does not duplicate its observations. Contacts and
photos are excluded. If the SQLite archive cannot be written, the run stops
and the failure is surfaced to the caller; the JSON snapshot remains.


### Claude analysis in the OLX/Uybor bulletin

The `--olx` / `--olx-snapshot` workflow requires `ANTHROPIC_API_KEY` and uses
the configured `ANTHROPIC_MODEL` (or `--model`) to interpret computed tables
and write Uzbek summaries and findings. Source notes and methodology are
deterministic; the authorship statement identifies Claude.

Section prose is split by purpose. The bullet blocks carry findings only: the
direction and breadth of a move, the places behind it, the levels and the
comparisons that can be read off the table above them. Coverage, sample
thresholds, breaks in the series, partly observed periods and source differences
sit in the `Manba` / `Eslatma` / `Qamrov` callouts, which Claude is not asked to
rewrite, so a caveat is stated once where it belongs rather than repeated as
commentary. The system prompt forbids describing the dataset or the production of
the report in the findings, and requires every masked figure to be cited one cell
at a time, with its unit and its own region, period and segment.

The model receives computed tables and limitations, not raw seller data. Numeric
references resolve to exact supplied values; unknown references, literal new
numbers, incomplete blocks and invalid output fail the run before report writing.
Period phrases are masked whole, so a quarter's Roman numeral travels with its
year and a numeral the model wrote for itself is refused. Masking is one pass
over the text: a second pass would mask the digits inside a reference already
inserted, and a model copying the inner reference then prints a literal `[[F46]]`
on the page. A paragraph that still carries any fragment of a reference after
decoding fails the run rather than being printed.
These checks prevent new numeric literals but do not prove every interpretation;
causal explanations must remain hypotheses and require review.

No silent template fallback is used in the CLI/app OLX workflow. Missing keys
or API failures produce an error. Low-level `write(..., llm=None)` remains
available for offline template tests. The run log records
`narrative.generated_by`, the model, last successful response token usage and
the generated text blocks. Token usage is for that response, not total
billing including retries.


Prose follows one set of Uzbek conventions, held in
[`report/style.py`](uzhousing/report/style.py): a comma for the decimal
separator and a space for thousands (`20,37 mln so'm`), the word `foiz` written
outside the figure so a sentence can inflect it (`3,4 foizga oshdi`), and
quarters named in Roman numerals (`2026-yil II chorak`). Table cells keep the
compact machine forms — `20.37`, `+3.4%`, `Δ Ch2/Ch1` — because a cell is read
as a value, not as a sentence.

Sentences are written for a reader who runs a household or a business, not for
a statistician: units are spelled out in words (`har bir kvadrat metr uchun
20,37 million so'm`, not `20,37 mln so'm/m²`), no `%`, `Δ` or formula appears
inside a sentence, and a term that cannot be avoided is explained where it is
used — a median is the price at which half the adverts are cheaper and half
dearer, and is never called an average. The editorial pass reports notation left
in prose as a fault.

The commentary reads its tables rather than narrating them: two or three
findings per section, each answering what the movement means — how broadly it
is shared, whether it continued or reversed the previous quarter, how far apart
the ends of the market sit, what it implies for affordability. Descriptive data
cannot identify a cause, so an explanation is either marked as an untested
hypothesis or the text says outright that the data do not settle it.

Every caveat that governs the whole report — asking prices are not transaction
prices, the archive and the dated snapshot are not the same coverage, rent and
sale are not comparable side by side, thin groups carry no median, the report is
descriptive — stands once, in a single box in the methodology section. Section
notes carry only what is specific to their own table.

### The editorial pass — [`report/editorial.py`](uzhousing/report/editorial.py)

After the prose is assembled, it is read the way a desk editor would read it:
worn phrases (`darajalar bo'yicha`, `bevosita taqqoslanmaydi`, `joriy kesim` and
the rest), a clause repeated in two sections, two paragraphs opening the same
way, a paragraph that restates a figure without saying what it means, an
over-long paragraph, and sections or blocks with nothing in them. The flagged
paragraphs — and only those — go back to Claude for one revision round with the
objection attached; a failed revision leaves the accepted first draft in place.
Whatever the second read still objects to is recorded in the run log under
`narrative.editorial`, together with how many blocks were rewritten. Sections
with nothing to show are dropped before rendering rather than printing as a
blank page, and a caption is kept on the page with the figure or table it
belongs to.

Regenerate saved data without recollecting:
`python run.py --olx-snapshot outputs/olx_snapshots/SNAPSHOT.json --lang uz`
Add the same `--olx-archive` path as the original run to retain historical tables.
