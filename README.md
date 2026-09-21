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
.venv\Scripts\python.exe run.py --olx --olx-browser --olx-pages 26 --uybor-pages 20 --lang uz --olx-archive "C:\Users\<you>\Desktop\olx\price_database"
```

Roughly 15-25 minutes, most of it OLX. It writes a PDF and a Word report to
`outputs/reports/`. No API key is needed. The rest of this section explains each
part, and every flag can be dropped independently: without `--olx-archive` there
is no trend section, without `--uybor-pages` only OLX is collected.

### Collect from OLX.uz and write the Uzbek bulletin

```powershell
.venv\Scripts\python.exe run.py --olx --olx-browser --olx-pages 26 --lang uz
```

`--olx-browser` is required. Plain HTTP is refused by the site's CDN firewall
with HTTP 403 before any page is served, so collection runs through a real
browser. Install it once with `pip install playwright`; the system Chrome is
used when present, otherwise run `python -m playwright install chromium`.

`--olx-pages` sets how many pages per category. **26 is the practical maximum**:
OLX refuses a paging offset beyond 1000, so a category yields at most about
1,040 listings however many pages are asked for. Reaching that ceiling is a
normal stop, not an error. Values below about 5 leave most regional medians
blank, because a median is suppressed under 15 observations.

### Add historical price trends

```powershell
.venv\Scripts\python.exe run.py --olx --olx-browser --olx-pages 26 --lang uz --olx-archive "C:\Users\<you>\Desktop\olx\price_database"
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
.venv\Scripts\python.exe run.py --olx --olx-browser --olx-pages 26 --uybor-pages 20 --lang uz
```

Adds Uybor.uz listings, pooled into the same medians. Each page is up to 100
listings, so `--uybor-pages 20` is about 2,000. Uybor needs no browser. See
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
and download the `.docx`. The **OLX.uz avtomatik** tab runs collection instead,
with browser collection on by default and the archive folder as an optional
field.

### When something fails

```powershell
.venv\Scripts\python.exe run.py --olx --olx-browser --olx-show-browser --olx-pages 1 --lang uz
.venv\Scripts\python.exe run.py --data data.json --verbose
```

`--olx-show-browser` opens a visible Chrome window, which shows whether OLX is
serving a challenge page. `--verbose` prints the full traceback instead of a
one-line message.

### Unattended runs

Scheduling is external: the flags above perform one collection and one report.
For a recurring snapshot, point Windows Task Scheduler at
`<project>\.venv\Scripts\python.exe` with arguments
`run.py --olx --olx-browser --olx-pages 26 --lang uz`, and set **Start in** to
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

## Bulletin contents (the `--olx` report)

A different report from the one above: Uzbek Latin, Claude-written analysis, and built from
listings rather than an uploaded dataset. It follows the structure of the
supplied quarterly review, as far as listing data can support it.

| Section | Source | Notes |
|---|---|---|
| Qisqacha xulosa | both | what was collected, in what units |
| Birlamchi uy-joy bozori | archive + snapshot | quarterly region and Tashkent-district tables, then today's cross-section |
| Ikkilamchi uy-joy bozori | archive + snapshot | the same, for the secondary market |
| Bozor turi ko'rsatilmagan | snapshot | adverts with no market field, kept apart rather than assigned |
| Uy-joy narxlari indeksi | archive | first observed quarter = 100, with the quarterly levels |
| Tarixiy taqqoslash | archive | the monthly series, with unobserved months left broken |
| Ijara bozori | snapshot | region and Tashkent-district rents; the archive holds no rentals |
| Toshkent shahri tumanlarida sotuv | snapshot | current district cross-section |
| Ijara rentabelligi (yillik) | snapshot | gross proxy on matched region / property / room strata |
| Uy-joy qurilishi va ipoteka bozori | — | **stated as unavailable**: not in listing data |
| Manbalar tarkibi | snapshot | how many listings each source contributed, so a change of mix is visible |
| Metodologiya va manbalar | both | screens, dated rates and their effective dates, coverage |

Sections that need an archive say so and fall back to the snapshot alone when
`--olx-archive` is not given; nothing is left silently empty.

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
.venv\Scripts\python.exe run.py --olx --olx-browser --olx-pages 26 --lang uz
```

`--olx-browser` uses the system Chrome when present and falls back to Playwright's
own Chromium. Add `--olx-show-browser` to watch the window, which is the first
thing to try if a run is refused. In the app, **Brauzer orqali yig'ish** is on by
default in the **OLX.uz avtomatik** tab.

No login is performed, no CAPTCHA is solved and no access control is bypassed.
A challenge page that does not clear on its own stops the run, exactly as a 403
does. Everything else is unchanged by the switch: the same robots.txt rules,
request pacing, schema validation and snapshot format apply.

Verified live on 2026-09-21: robots.txt and all four category pages returned
HTTP 200, and a `--olx-pages 2` run collected 412 listings across the four
categories with the dated CBU rate. Medians below 15 observations are suppressed,
so a small `--olx-pages` value leaves many regional cells blank; raise it to fill them.


### Historical price trends from an archive

The bulletin is a current snapshot on its own. Given a folder of archived OLX
`.db` files it also reports a monthly asking-price trend:

```powershell
.venv\Scripts\python.exe run.py --olx --olx-browser --olx-pages 26 --lang uz --olx-archive "C:\Users\<you>\Desktop\olx\price_database"
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

`--uybor-pages N` also collects up to N pages of 100 residential listings from
Uybor.uz, whose public JSON API the site's own pages call:

```powershell
.venv\Scripts\python.exe run.py --olx --olx-browser --olx-pages 26 --uybor-pages 20 --lang uz
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

Collection now uses `uzhousing/ingest/olx_client.py` for paced HTTP access.
Category pages provide embedded offers or explicit numeric IDs on listing cards;
when an ID has no embedded offer, the collector requests the undocumented
`/api/v1/offers/{id}/` endpoint. It validates the response ID and schema.
404/410 detail responses are counted as unavailable, never classified as sold.
401/403/429 stop collection without retries or an access-control workaround.

Category pages are now client-rendered and no longer embed their listings, so the
collector first establishes the category's own numeric ID by reading it from the
listings that page actually shows. The ID is never guessed: it is accepted only
when independent listings agree on it. With an ID established, the bounded,
deduplicated `/api/v1/offers/` pagination returns about forty full records per
request instead of one request per listing, and the snapshot records which
strategy each category used under `coverage[].strategy`. Where no ID can be
verified, collection falls back to reading listing pages as before.

OLX validates the paging offset and refuses anything beyond 1000 with HTTP 400,
so a category yields at most about 1,040 listings however many pages are asked
for. Reaching that ceiling is a normal stop, recorded as `depth_limit` in
`coverage[].stop`, not a failure: the pages already collected are kept and the
report is still written. `--olx-pages` above roughly 26 therefore adds nothing.
A mid-run 401/403/429 still stops the run, and is never mistaken for the cap.

Successful collections retain the same JSON snapshot format for the report and
also append observations to `outputs/olx_history.sqlite` (or the selected output
folder). `collection_runs` stores coverage and FX metadata; `listing_snapshots`
stores housing fields and price/currency for each collection timestamp.
`listing_observation_history` exposes first_seen, last_seen and observation count.
These are observed dates, not confirmed time on market or transaction dates.
Missing listings in bounded crawls are not marked sold or removed. Reimporting
one snapshot does not duplicate its observations. Contacts/photos remain excluded.
JSON snapshots remain available if the separate SQLite archive cannot be written;
such a storage failure stops the run and is surfaced to the caller.

This update changes data collection/storage only. Report templates, calculations,
language, layout and the existing synthetic sample are unchanged.


### Claude analysis in the OLX/Uybor bulletin

The normal `--olx` / `--olx-snapshot` workflow now requires
`ANTHROPIC_API_KEY` and uses the configured `ANTHROPIC_MODEL` (or `--model`)
to interpret computed tables and write Uzbek summaries and findings. Existing
section titles/order, text-block and bullet counts, tables, charts and renderers
are preserved. Prose length can change pagination. Source notes and methodology
remain deterministic, with the authorship statement updated to identify Claude.

The model receives computed tables and limitations, not raw seller data. Numeric
references resolve to exact supplied values; unknown references, literal new
numbers, incomplete blocks and invalid output fail the run before report writing.
These checks prevent new numeric literals but do not prove every interpretation;
causal explanations must remain hypotheses and require review.

No silent template fallback is used in the CLI/app OLX workflow. Missing keys or
API failures produce an error. Low-level `write(..., llm=None)` remains available
for offline template tests. The run log records `narrative.generated_by`, model,
last successful response token usage and the actual generated text blocks. Token
usage is for that response, not total billing including any retries.

Regenerate saved data without recollecting:
`python run.py --olx-snapshot outputs/olx_snapshots/SNAPSHOT.json --lang uz`
Add the same `--olx-archive` path as the original run to retain historical tables.
