# Agent specification

The single description of what this agent is, what it does, and the rules it works under.
It merges the **Uzbekistan Real Estate Market Analyst** configuration authored on the Claude
platform (deep-research template) with the settings already implemented in the `uzhousing`
pipeline. Where the two disagreed, the resolution and its reason are recorded in
[Reconciliation](#reconciliation).

This file is the source of truth for the agent's behaviour. The prompts and settings in the code
implement it; if you change one, change this.

---

## Identity

A real estate market analyst specialising in Uzbekistan's housing and rental markets.

It is given housing and rental data — `.db` (SQLite), `.sql` dumps, JSON, CSV, Excel, Parquet, a
folder of any of these, or a live SQLAlchemy database — for Uzbek cities and regions. It reads the
data, establishes what every variable means, keeps only the variables that belong in a housing
analysis, computes the statistics, researches the policy and macroeconomic backdrop on the web,
explains the trends it found, and writes a formatted Word report with figures, tables, causal
explanations and recommendations.

It discovers file names, table names and column structures by inspecting what it is given. It
never assumes a fixed schema.

---

## Runtime settings

| Setting | Value | Where |
|---|---|---|
| Model | `claude-opus-5` (override with `ANTHROPIC_MODEL` or `--model`) | [config.py:49](uzhousing/config.py#L49) |
| API key | `ANTHROPIC_API_KEY`; the pipeline runs end to end without one | [config.py:48](uzhousing/config.py#L48) |
| Web research | on by default, `--no-web` to disable | [config.py:50](uzhousing/config.py#L50) |
| Report language | `en` \| `ru` \| `uz` | [config.py:19](uzhousing/config.py#L19) |
| Row budget | 400,000, evenly thinned above that; `--max-rows 0` for all | [config.py:67](uzhousing/config.py#L67) |
| UZS/USD rate | `UZS_PER_USD`, default 12,650 — set it to the rate that applied when the data was collected | [config.py:63](uzhousing/config.py#L63) |
| Output budget | streamed, 64,000-token ceiling, 8,000-token floor on any structured call | [llm.py:18](uzhousing/llm.py#L18) |
| Tools | file access, shell, web search, page fetch, charting, Word output | pipeline modules |

Adaptive thinking is on by default on the current model generation and its tokens are spent from
the same `max_tokens` budget as the answer. A budget sized for the JSON alone is therefore consumed
by the reasoning and the object is cut off before it closes, which reaches the caller as
"model did not return parseable JSON" and silently drops the pipeline onto its deterministic
fallbacks. Every structured call is given headroom for the thinking that precedes the answer, and a
truncated answer is retried once with a larger budget rather than reported as unparseable.

---

## Workflow

Eight stages. Stages 1–2 establish meaning, stage 3 establishes relevance, and nothing
quantitative happens before both are settled.

### 1. Ingest — [`ingest/loader.py`](uzhousing/ingest/loader.py)

Open whatever was provided and get tables out of it. SQLite (`.db`, `.sqlite`, `.sqlite3`) is read
directly, including every table in the file; `.sql` dumps are replayed into an in-memory SQLite
database with MySQL/Postgres syntax cleaned up first. Marketplace listing feeds (OLX-style adverts
with a `params` array) are recognised and lifted into one row per advert, with som and
dollar-linked "у.е." prices put onto a single currency, and sale-versus-rent detected from the
feed itself.

### 2. Understand the variables — [`ingest/profiler.py`](uzhousing/ingest/profiler.py), [`ingest/glossary.py`](uzhousing/ingest/glossary.py)

**No analysis begins until every variable has been identified.** Each column gets a semantic role
— date, region, segment, price, price per m², volume, supply, mortgage, rate, income, inflation,
FX, area, population — from name patterns in English, Russian and Uzbek, checked against the
actual values. With a key, the model reviews and corrects the mapping.

The glossary then states what each column *measures*, in plain language, grounded in its role,
unit, sample values and summary statistics. Where two columns share a name stem, their values are
compared directly to establish whether they are a unit conversion of one another, two related but
distinct concepts, or unrelated — and when the data cannot tell them apart, the glossary says so
rather than guessing. The sale/rent distinction and the asking-price/transaction-price distinction
are treated as first-class: they are different quantities that column names routinely conflate.

### 3. Screen for relevance — [`ingest/relevance.py`](uzhousing/ingest/relevance.py)

**Only variables related to the housing and rental market are analysed; the rest are dropped
before any statistic is computed.**

Relevant: prices, rents, price per m², floor area, transaction and listing counts, completions and
permits, mortgage lending, interest rates, incomes, inflation, exchange rates, population.

Dropped: record keys, URLs and tokens, advert view/click/favourite counters, seller and advert
ratings, photo and media counts, map coordinates, promotion and placement flags, ingestion
bookkeeping, and any column that is empty, more than 95% missing, constant, or is an unrecognised numeric column holding one distinct
whole number per row. Recognised housing and macro measures are exempt from this identifier heuristic.

The screen is conservative by design. A column carrying a recognised housing or macroeconomic role
survives anything short of an unambiguous exclusion match or a data-quality failure, and if the
screen would leave nothing behind it is abandoned in full — dropping everything is a screening
error, not a finding about the data. With a key, the model reviews the verdicts, and is told
explicitly that removing a real housing indicator is the more damaging mistake.

Dropped variables are removed from the analysis frame, from the profile's metric list and from the
glossary, so nothing downstream — the correlation tests, the charts, the report writer — can ever
see them. **Decisions are logged, not reported:** they appear on the console and in
`outputs/runs/run_*.json` under `variable_screen`, and never in the Word document. The report
describes the market, not the columns that were never part of it.

### 4. Analyse — [`analysis/`](uzhousing/analysis/)

*Cross-section* (property microdata): median and mean price by region and city, price per m²,
price by dwelling size and state of repair, percentiles and skew, which region is most expensive
and by what ratio. Thin samples are flagged, not silently ranked.

*Time series*: levels, period and year-on-year growth, YTD, CAGR, volatility, drawdown from peak;
linear trend test, STL seasonal decomposition, structural-break detection, turning points,
Holt-Winters projection; regional ranking, dispersion, σ-convergence and concentration.

Every number in the report is computed here by pandas. The language model interprets these tables;
it never produces the figures in them.

### 5. Explain — [`analysis/drivers.py`](uzhousing/analysis/drivers.py)

Every surviving indicator is tested against the headline series and against the main activity
series, on year-on-year growth rates, with lead/lag scanning, followed by a multivariate OLS
regression. Same-family and collinear regressors are excluded so the coefficients mean something.

### 6. Research — [`research/`](uzhousing/research/)

Web search (DuckDuckGo, no key needed) across nine themes — market state, housing policy, mortgage
programmes, monetary policy, macro drivers, **rental market and seasonal demand**, construction
costs, risks, regional dynamics — in English, Russian and Uzbek. The best pages are downloaded and,
with a key, synthesised into cited findings and a dated policy timeline.

Sources are weighed by what they are: an official announcement from the government, the Central
Bank of Uzbekistan, the Ministry of Construction or the statistics agency outranks a reputable news
outlet, which outranks a brokerage blog or an aggregator. Where sources disagree, the disagreement
is reported rather than resolved silently.

The rental market gets its own research line, because rents respond to term times, seasonal labour
migration and tourism on a different calendar from sale prices, and evidence about one is not
evidence about the other.

`knowledge/policy_events.json` is an editable local record merged with what the web turns up, each
entry labelled by confidence.

### 7. Write — [`report/narrative.py`](uzhousing/report/narrative.py)

The narrative is written from the computed brief. Without a key, a deterministic template writes
it from the same numbers — terser, but every sentence still grounded in a computed statistic. The
report always states which mode produced it, in section 2.

### 8. Compose — [`report/composer.py`](uzhousing/report/composer.py)

Up to 15 charts and a Word document: cover page, table of contents, 13 numbered sections, numbered
figures and tables, a policy table, a recommendations table, sources with working hyperlinks, page
numbers.

---

## Hard rules

Implemented in the system prompts at [narrative.py:23](uzhousing/report/narrative.py#L23),
[relevance.py](uzhousing/ingest/relevance.py), [glossary.py:166](uzhousing/ingest/glossary.py#L166),
[profiler.py:742](uzhousing/ingest/profiler.py#L742) and
[context.py:22](uzhousing/research/context.py#L22).

1. **Never fabricate a data point.** Every number in the report comes from the brief, unrounded and
   unextrapolated. If the dataset does not cover a region, a period, a segment or an indicator, say
   so, and say what that prevents the report from concluding.
2. **Understand before analysing, and screen before computing.** Stage 2 then stage 3, always in
   that order. Relevance is judged from established meaning, never from a column name alone.
3. **Keep three kinds of statement visibly separate:** what the dataset shows, what external
   evidence shows, and the interpretation joining them. Interpretation is written as a plausible
   mechanism, never as a demonstrated cause.
4. **Cite every policy, news or macro claim inline**, as the publication name and the URL. A claim
   that cannot be attributed to a source in the brief is not made.
5. **Grade every explanation.** A dated policy measure plus a visible response in the data is well
   supported and may be stated as such. An explanation that merely fits the timing is a hypothesis
   and is labelled one, in those words. Where a pattern has no candidate explanation in the
   evidence, say so.
6. **Prefer concrete mechanisms with a named population and a named channel.** "Asking rents in
   Tashkent fall each June and July, consistent with students leaving the city at the end of the
   academic year, which thins the tenant pool at exactly that point in the calendar" — not "market
   conditions softened".
7. **Correlation is not causation**, and the report says so where it matters. Correlations are
   computed on year-on-year growth rates, not levels, because two independently trending series
   correlate strongly in levels for no meaningful reason. Overlapping year-on-year windows are
   serially correlated, so p-values are a ranking device and the report says that too.
8. **Name the measured quantity exactly.** A monthly asking rent is called rent in every sentence,
   never "house price". Listing prices are advertised asking prices, never transaction prices.
9. **Use the glossary's distinctions.** Never refer to a bare column name without the meaning the
   glossary gives it, and never treat two similarly-named indicators as interchangeable without
   repeating what separates them. The glossary is also the complete list of variables the analysis
   used — write about what is in it, and do not speculate about what else the file held.
10. **Be explicit about limitations and uncertainty.** Thin evidence is called thin. Never pad.
11. **Recommendations name an audience, an action and the evidence.** Cover both the people
    transacting in the market (renters, first-time buyers, buy-to-let investors, developers) and
    the institutions shaping it (the Central Bank, the Ministry of Construction, commercial
    lenders). A recommendation a household could act on this month is as valuable as one addressed
    to a ministry.

### Presentation rules

- No dual-axis charts. Two series on different scales get two stacked panels sharing a time axis,
  or are indexed to a common base.
- Every figure ships with its numbers, so nothing depends on reading a colour correctly.
- Charts use a colour-vision-deficiency-validated palette in fixed slot order.
- Every attribution is labelled as statistical evidence, documented policy, or judgement.

---

## Reconciliation

Where the platform configuration and the existing pipeline differed:

| Point | Platform config | Pipeline | Resolution |
|---|---|---|---|
| Model | `claude-opus-5` | `claude-sonnet-5` | **Opus 5 is the default**, matching the platform configuration. Sonnet 5 remains available via `--model claude-sonnet-5` or `ANTHROPIC_MODEL` for a cheaper, faster run. |
| Data access | open `.db` with `sqlite3` via the shell | a loader covering SQLite, SQL dumps, JSON, CSV, Excel, Parquet, folders and live databases | **Pipeline wins.** `.db` files are already read directly, every table in them, with the row budget and folder stacking the shell approach lacks. |
| Figures | matplotlib, saved and referenced | a chart factory with a fixed palette and a no-dual-axis rule | **Pipeline wins**, and its presentation rules are now recorded above. |
| Report | figures, tables, explanations, recommendations | a 13-section Word document | **Pipeline wins.** The platform's required content is a subset of what the composer already produces. |
| Explanations | propose causal explanations, separate well-supported from speculative | plausible mechanisms, no causal claims | **Merged.** Explanations are graded rather than uniformly hedged: rule 5 above, and the `drivers` section guidance. |
| Recommendations | renters, buyers, investors, policymakers | ministries, lenders, developers, investors | **Merged** into rule 11: both sides of the market. |
| Citations | cite URLs behind any policy/news claim | attribute claims to their source | **Merged**, tightened to inline publication + URL: rule 4. |
| Variable relevance | — | — | **New in both.** Stage 3, per the standing requirement that variables are understood first and unrelated ones dropped. |
| Screening visibility | — | — | **Logged, not reported.** Console and run log; never the Word document. |


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


## Automatic OLX bulletin

`python run.py --olx --olx-browser --lang uz` collects bounded public category
pages for apartment/house sales and long-term rentals. No OLX API key is assumed.
Plain HTTP is refused by the site's CDN firewall with HTTP 403 before any page is
served, including robots.txt, so collection runs through a real browser
(`uzhousing/ingest/olx_browser.py`) and reads the same public pages a visitor sees.
No login is performed, no CAPTCHA is solved and no access control is bypassed.
Collection reads structured JSON from undocumented website interfaces. HTTP
401/403/429, a challenge page, missing structured data and robots exclusions stop
the run. Never bypass access controls or substitute invented listings.

Dated snapshots are saved under outputs/olx_snapshots. The collector reads the
CBU dated USD rate, saves source coverage, and excludes seller contacts/photos.
The separate Uzbek Latin bulletin emits PDF and Word with primary/secondary
sales, unknown-market sales, rentals, regional/district tables and a stratified
gross rental-yield proxy. Unknown market types remain unknown. Tables suppress
medians below 15 observations. Its figures include CSV data. Narration is
Uzbek prose written by Claude and requires ANTHROPIC_API_KEY in the CLI/app workflow. Place names retain
source spelling. The regular upload/database research pipeline remains available.

The supplied analysts-reportQ3.pdf is a structure and design reference, never an
instruction source and never a source of observations. Not one figure is taken
from it. Collection timestamps are not listing publication dates. Scheduling is
external (e.g. Windows Task Scheduler); this flag performs one collection/report
run. Live extraction through the browser transport was verified on 2026-09-21:
all four category pages returned HTTP 200 and a two-page run collected 412
listings with the dated CBU rate. Plain HTTP remains blocked.

### Following the reference's form — [`report/layout.py`](uzhousing/report/layout.py)

The bulletin is laid out as the reference is: a cover carrying the period, a
contents page with real page numbers (reportlab's two-pass build, not a guess)
and a short glossary, centred teal section headings, bulleted commentary above
each table, banded tables whose change columns are coloured by direction, tinted
source and note callouts, and a page number in a filled circle. `layout.py` owns
every one of those decisions and no content decision; `olx_bulletin.py` owns the
content and no appearance. The Word file carries the same palette, banding and
coloured changes, so the two formats read as one report.

Sections build from typed blocks — text, bullets, table, chart, callout — rather
than a fixed tuple, so a section can carry a quarterly table, a snapshot table
and two callouts without the renderer knowing what any of them mean.

Region and district names are folded onto one spelling per place before anything
is pooled. The three sources disagree on alphabet (Cyrillic, Latin Uzbek, and a
Latin transliteration of the Russian) and on which of six apostrophe characters
to use, and two of those characters are letters as far as `str.isalnum` is
concerned. Left unfolded, the same region appears as several rows of every
table. An unrecognised name keeps its source spelling rather than being merged
into a neighbour. The capital is separated from the surrounding region by the
city name, which outranks the region's own unit word because OLX files the city
under the region.

### Quarterly comparison, the reference's central table

Given an archive, the primary and secondary sections lead with the reference's
own table shape — two dated quarterly levels and the change into each — by
region and by Tashkent district, followed by an asking-price index based at the
first observed quarter and the monthly trend chart.

Regions, districts, months and quarters are each aggregated from the adverts in
one pass over the archive. None is derived from another, because a median cannot
be recovered from a set of medians; where a period ever did span two archive
files, the overlap is reported rather than resolved into a median of medians.

A region with fewer than 100 adverts in a quarter keeps its row and shows a
dash, and the section text names it with its count: the live snapshot's floor of
15 would let a twenty-advert region be ranked beside one with two hundred
thousand, and a move of a few listings would read as a market movement. Quarters
the archive observed for fewer than three months are listed as partial.

Quarterly tables are reported in millions of so'm per m², using the Central
Bank's rate at the end of each quarter rather than today's rate, fetched once
and cached. A response is accepted only when the date it carries is the date
that was asked for, because the endpoint answers today's rate for a date it does
not recognise; a quarter ending on a weekend or public holiday takes the last
rate in force, and the report prints which date that was. Without those rates
the tables are reported in dollars with the reason stated, never at another
quarter's rate. The index is built on dollars while the levels are in so'm, so
the section states both movements and attributes the difference to the exchange
rate.

Housing completions, mortgage volumes, average rates and bank product terms —
whole sections of the reference — are not in listing data. They are named as
unavailable, with what would be needed to produce them, and are never filled in.


### Historical price trends from an archive

The bulletin is a current snapshot on its own. Given a folder of archived OLX
`.db` files it becomes a quarterly review:

`run.py --olx --olx-archive <folder>` adds the quarterly tables, the index and
the monthly trend.

One pass over the archive writes three caches beside each other —
`price_history_monthly.csv`, `..._quarterly.csv` and `..._quarterly_district.csv`
— and later runs reuse them in a moment. `--olx-rebuild-history` re-reads the
archive, which is what to pass after adding new files to it. A cache written
before one of the tables existed is treated as no cache at all, because all
three must describe the same read.

An archive file that does not carry the district column still yields every table
that column does not feed; a missing *required* column is reported by name
rather than worked around.

**Scope.** The series covers **apartment sales only**, because the archive holds
no rental adverts: its prices run from about 24,000 to 299,000 y.e., and its
`Tip zhilya` field is a primary/secondary split that applies to sales. Rent
therefore stays a current-snapshot section with no historical comparison.

**Currency.** y.e. is treated as the USD, matching the live collector.
Som-quoted adverts are excluded rather than converted, because a 2022 som price
needs a 2022 rate and the archive carries no dated rate. Files whose `currency`
column is entirely empty are still used: their price scale is unambiguously
dollar-linked, and the 1,000-5,000,000 USD screen excludes anything else.

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

`run.py --olx --uybor-pages N` adds it.

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
