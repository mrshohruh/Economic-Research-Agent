"""One-off CLI demo runner to exercise the full pipeline end-to-end and
produce a sample DOCX report, without going through Streamlit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import run_full_pipeline

TOPIC = ("Analyze the recent inflation dynamics in the dataset and identify the main factors behind the "
         "decline in headline inflation, including the role of food prices, the exchange rate, and monetary "
         "policy.")
FILE = str(Path(__file__).parent / "sample_data" / "synthetic_inflation_data.xlsx")


def cb(idx, label):
    print(f"[{idx}/10] {label} ... done")


if __name__ == "__main__":
    result = run_full_pipeline(FILE, TOPIC, {
        "research_news": True, "research_policy": True,
        "generate_tables": True, "generate_figures": True, "write_report": True,
    }, progress_cb=cb)

    print("\n=== SUMMARY ===")
    print(f"Findings: {len(result.findings)}")
    for f in result.findings[:8]:
        print(f"  [{f.importance}] {f.finding}")
    print(f"Evidence collected: {len(result.evidence)}")
    print(f"Policy events: {len(result.policy_events)}")
    print(f"Tables: {list(result.tables.keys())}")
    print(f"Figures: {list(result.figures.keys())}")
    print(f"Fact-check items: {len(result.fact_check_items)}, "
          f"flagged/revised: {sum(1 for i in result.fact_check_items if i.status != 'verified')}")
    print(f"Report written to: {result.report_path}")
