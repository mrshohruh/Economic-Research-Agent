"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import LANGUAGES, Settings
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
    source.add_argument("--db", help="SQLAlchemy database URL, e.g. postgresql://user:pw@host/db")
    source.add_argument("--query", "-q", help="SQL query to run against --db or a SQLite --data file")

    output = parser.add_argument_group("output")
    output.add_argument("--out", "-o", help="output directory (default: outputs/)")
    output.add_argument("--title", "-t", default="", help="report title")
    output.add_argument("--lang", choices=sorted(LANGUAGES), help="report language (default: en)")

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument("--no-web", action="store_true", help="skip live web research")
    behaviour.add_argument("--model", help="Anthropic model id, e.g. claude-opus-5")
    behaviour.add_argument("--api-key", help="Anthropic API key (overrides .env)")
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

    if not args.data and not args.db:
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
    if args.model:
        settings.model = args.model
    if args.api_key:
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
