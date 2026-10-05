"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import LANGUAGES, SUPPORTED_PROVIDERS, Settings
from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uzhousing",
        description=(
            "Analyse a housing-market dataset and write a Word research report on the "
            "Uzbek housing market."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python run.py --data sample_data/uz_housing_sample.json
  python run.py --data C:\\data\\Tashkent      # every supported file in the folder
  python run.py --data dump.sql --title "Tashkent Primary Market Review"
  python run.py --data market.sqlite --query "SELECT * FROM prices WHERE year >= 2019"
  python run.py --db "postgresql://user:pw@host/db" --query "SELECT * FROM housing"
  python run.py --data data.json --no-web --lang ru
  python run.py --make-sample            # write a demo dataset and exit
""",
    )
    source = parser.add_argument_group("data source")
    source.add_argument(
        "--data", "-d",
        help="path to a .json, .sql, .sqlite, .csv or .xlsx file, or a folder of them",
    )
    source.add_argument("--olx", action="store_true", help="collect OLX listings and write an Uzbek PDF/Word bulletin")
    source.add_argument("--olx-pages", type=int, default=None, metavar="N",
                        help="stop after N pages per OLX category "
                             "(default: every page the site serves)")
    source.add_argument("--olx-snapshot", help="regenerate the Uzbek bulletin from a saved OLX snapshot")
    source.add_argument("--olx-browser", action="store_true",
                        help="collect through a real browser (needed: plain HTTP is refused with 403)")
    source.add_argument("--olx-show-browser", action="store_true",
                        help="with --olx-browser, show the browser window instead of running it hidden")
    source.add_argument("--uybor-pages", type=int, default=None, metavar="N",
                        help="stop after N pages of up to 100 Uybor.uz listings "
                             "(default: every page; 0 = skip Uybor)")
    source.add_argument("--no-uybor", action="store_true",
                        help="collect OLX only, without Uybor.uz")
    source.add_argument("--olx-archive", metavar="DIR",
                        help="folder of archived OLX .db files, to add a historical price trend")
    source.add_argument("--olx-rebuild-history", action="store_true",
                        help="re-read the archive instead of using the cached monthly series")
    source.add_argument("--db", help="SQLAlchemy database URL, e.g. postgresql://user:pw@host/db")
    source.add_argument("--query", "-q", help="SQL query to run against --db or a SQLite --data file")

    output = parser.add_argument_group("output")
    output.add_argument("--out", "-o", help="output directory (default: outputs/)")
    output.add_argument("--title", "-t", default="", help="report title")
    output.add_argument("--lang", choices=sorted(LANGUAGES), help="report language (default: en)")

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument("--no-web", action="store_true", help="skip live web research")
    behaviour.add_argument("--provider",
                           choices=list(SUPPORTED_PROVIDERS),
                           help="LLM vendor to call first. Defaults to the vendor named by "
                                "LLM_PROVIDER or by the model id, else to whichever free "
                                "provider has a key set.")
    behaviour.add_argument("--model",
                           help="Model id, e.g. claude-opus-5, gpt-5 or openai/gpt-oss-120b. "
                                "May carry a provider prefix such as groq:openai/gpt-oss-120b, "
                                "in which case the prefix selects the vendor.")
    behaviour.add_argument("--api-key",
                           help="API key for the chosen provider (overrides .env)")
    behaviour.add_argument("--no-llm-fallback", dest="llm_fallback",
                           action="store_false", default=None,
                           help="Disable the free-provider fallback chain for this run.")
    behaviour.add_argument("--results", type=int, help="search results per query (default: 6)")
    behaviour.add_argument("--pages", type=int, help="pages to download per theme (default: 3)")
    behaviour.add_argument("--max-rows", type=int, metavar="N",
                           help="row budget held in memory (default: 400000; 0 for no limit)")

    misc = parser.add_argument_group("other")
    misc.add_argument("--make-sample", action="store_true",
                      help="generate a realistic demo dataset and exit")
    misc.add_argument("--verbose", "-v", action="store_true", help="show debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.make_sample:
        from sample_data.generate import write_sample

        path = write_sample()
        print(f"Sample dataset written to {path}")
        print(f"Now run:  python run.py --data {path}")
        return 0

    if sum(bool(x) for x in (args.data, args.db, args.olx, args.olx_snapshot)) > 1:
        print("Choose only one data source", file=sys.stderr)
        return 2

    if not args.data and not args.db and not args.olx and not args.olx_snapshot:
        build_parser().print_help()
        print("\nerror: pass --data <file> or --db <url> (or --make-sample to try it out)", file=sys.stderr)
        return 2

    settings = Settings.from_env()
    if args.out:
        settings.output_dir = Path(args.out).resolve()
    if args.lang:
        settings.language = args.lang
    if args.no_web:
        settings.web_research = False
    if args.provider:
        settings.llm_provider_setting = args.provider
    if args.model:
        # ``--model`` can carry a provider prefix such as ``groq:openai/...``
        # — the resolver in Settings honours it.
        settings.llm_model_setting = args.model
        # Keep the legacy ``.model`` attribute in sync too so any code that
        # reads it directly still sees what the user asked for.
        from .llm import parse_model_id

        _, bare_model = parse_model_id(args.model)
        if bare_model:
            settings.model = bare_model
    if args.llm_fallback is False:
        settings.fallback_enabled = False
    if args.api_key:
        # Assigned to whichever key the chosen provider actually reads, so
        # one flag works for every supported vendor.
        provider = settings.provider
        if provider == "openai":
            settings.openai_api_key = args.api_key
        elif provider == "groq":
            settings.groq_api_key = args.api_key
        elif provider == "gemini":
            settings.gemini_api_key = args.api_key
        elif provider == "openrouter":
            settings.openrouter_api_key = args.api_key
        else:
            settings.anthropic_api_key = args.api_key
    if args.results:
        settings.search_results_per_query = args.results
    if args.pages:
        settings.pages_to_read = args.pages
    if args.max_rows is not None:
        settings.max_rows = max(0, args.max_rows)

    def progress(message: str) -> None:
        print(f"  {message}", flush=True)

    print("\n" + "=" * 72)
    print("  Uzbekistan Housing Market Research Agent")
    print("=" * 72)

    if args.olx or args.olx_snapshot:
        if args.lang and args.lang != "uz":
            print("The OLX bulletin currently supports --lang uz only.", file=sys.stderr)
            return 2
        from .report.olx_bulletin import run_olx
        try:
            result = run_olx(settings, pages=args.olx_pages, progress=progress,
                             title=args.title, snapshot_path=args.olx_snapshot,
                             browser=args.olx_browser, headless=not args.olx_show_browser,
                             archive=args.olx_archive, rebuild_history=args.olx_rebuild_history,
                             uybor_pages=0 if args.no_uybor else args.uybor_pages)
        except Exception as exc:
            print(f"OLX failed: {exc}", file=sys.stderr)
            if args.verbose:
                raise
            return 1
        print(f"PDF: {result.pdf_path}\nWord: {result.report_path}\nLog: {result.run_log}")
        return 0

    try:
        result = run(
            settings=settings,
            data_path=args.data,
            connection_url=args.db,
            query=args.query,
            title=args.title,
            progress=progress,
        )
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        print("Re-run with --verbose for the full traceback.", file=sys.stderr)
        return 1

    print("-" * 72)
    if result.warnings:
        print(f"  {len(result.warnings)} note(s) recorded — see section 2 of the report.")
    print(f"  Figures:  {len(result.figures)}")
    print(f"  Run log:  {result.run_log}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(f"\n  REPORT:   {result.report_path}\n")
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
