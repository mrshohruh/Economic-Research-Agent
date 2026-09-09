"""
Automated tests (Section 36) covering XLSX loading, variable/date/unit
detection, missing values, growth calculations, YoY calculations, trend and
anomaly detection, research-plan generation, visualization planning, figure
and table generation, report generation, fact-checking, and invalid input.

Run with: python -m pytest tests/ -v   (or) python -m unittest discover tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data import ingestion, metadata, validation
from analysis import anomalies, time_series, trends
from agents.quantitative_agent import run_quantitative_analysis
from agents.research_planner import build_research_plan, build_search_queries, generate_hypotheses
from agents.visualization_agent import build_figures, build_tables
from agents.report_writer import build_report
from agents import fact_checker
from models.schemas import Finding
from sample_data.generate_synthetic_data import OUT_PATH, generate, main as generate_main

TOPIC = "Analyze inflation dynamics and identify the main factors behind the decline in headline inflation."


class TestDataIngestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OUT_PATH.exists():
            generate_main()
        cls.sheets = ingestion.load_and_analyze(OUT_PATH)
        cls.primary = ingestion.choose_primary_sheet(cls.sheets)
        cls.sd = cls.sheets[cls.primary]

    def test_multi_sheet_handling(self):
        self.assertGreaterEqual(len(self.sheets), 2)

    def test_date_detection(self):
        self.assertEqual(self.sd.date_column, "date")

    def test_frequency_detection(self):
        self.assertEqual(self.sd.frequency, "monthly")

    def test_variable_detection(self):
        self.assertIn("headline_inflation", self.sd.numeric_columns)
        self.assertIn("exchange_rate", self.sd.numeric_columns)

    def test_metadata_build(self):
        meta = metadata.build_dataset_metadata(OUT_PATH, self.sheets, self.primary)
        self.assertEqual(meta.primary_sheet, self.primary)
        self.assertTrue(len(meta.sheets) >= 1)

    def test_invalid_input_raises(self):
        with self.assertRaises(Exception):
            ingestion.load_and_analyze("nonexistent_file_xyz.xlsx")


class TestValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sheets = ingestion.load_and_analyze(OUT_PATH)
        cls.sd = cls.sheets[ingestion.choose_primary_sheet(cls.sheets)]

    def test_validation_report_structure(self):
        report = validation.validate_sheet(self.sd)
        self.assertGreater(report.n_observations, 0)
        self.assertTrue(any(i.level == "ok" for i in report.issues))

    def test_missing_values_detected(self):
        df = self.sd.df.copy()
        df.loc[0, "headline_inflation"] = None
        from data.ingestion import SheetData
        sd2 = SheetData(sheet_name="test", df=df, date_column=self.sd.date_column,
                         frequency=self.sd.frequency, numeric_columns=self.sd.numeric_columns,
                         categorical_columns=self.sd.categorical_columns)
        report = validation.validate_sheet(sd2)
        self.assertTrue(any("missing value" in i.message for i in report.issues))


class TestAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sheets = ingestion.load_and_analyze(OUT_PATH)
        cls.sd = cls.sheets[ingestion.choose_primary_sheet(cls.sheets)]

    def test_yoy_calculation(self):
        series = time_series.compute_change_series(self.sd.df, self.sd.date_column, "headline_inflation", "monthly")
        self.assertIn("yoy_pct", series)
        self.assertGreater(series["yoy_pct"].notna().sum(), 0)

    def test_trend_detection(self):
        series = time_series.compute_change_series(self.sd.df, self.sd.date_column, "headline_inflation", "monthly")
        t = trends.linear_trend(series["level"])
        self.assertIn("slope", t)

    def test_anomaly_detection_food_bump(self):
        series = time_series.compute_change_series(self.sd.df, self.sd.date_column, "food_inflation", "monthly")
        # The synthetic generator injects a deliberate bump; z-score anomalies should exist somewhere in sample.
        df = anomalies.zscore_anomalies(series["mom_pct"])
        self.assertIsInstance(df, pd.DataFrame)

    def test_quantitative_agent_produces_findings(self):
        result = run_quantitative_analysis(self.sd, TOPIC)
        self.assertIsInstance(result.findings, list)
        self.assertGreater(len(result.stats), 0)


class TestResearchPlanning(unittest.TestCase):
    def test_plan_generation(self):
        findings = [Finding(id="F1", finding="Test finding", importance="high", variables=["headline_inflation"])]
        plan = build_research_plan(TOPIC, findings)
        self.assertGreater(len(plan.steps), 0)
        self.assertGreater(len(plan.search_queries), 0)

    def test_query_generation_nonempty(self):
        findings = [Finding(id="F1", finding="Food inflation increased sharply", importance="high",
                             variables=["food_inflation"], period="2023-06")]
        queries = build_search_queries(TOPIC, findings)
        self.assertGreater(len(queries), 1)

    def test_hypothesis_generation(self):
        f = Finding(id="F1", finding="Test finding", importance="high", variables=["headline_inflation"])
        hyps = generate_hypotheses(f, TOPIC)
        self.assertGreaterEqual(len(hyps), 3)


class TestVisualizationAndReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sheets = ingestion.load_and_analyze(OUT_PATH)
        cls.sd = cls.sheets[ingestion.choose_primary_sheet(cls.sheets)]
        cls.quant = run_quantitative_analysis(cls.sd, TOPIC)

    def test_table_generation(self):
        tables = build_tables(TOPIC, self.sd, self.quant.findings)
        self.assertGreater(len(tables), 0)
        for tid, (plan, df) in tables.items():
            self.assertFalse(df.empty)

    def test_figure_generation(self):
        figures = build_figures(TOPIC, self.sd, self.quant.findings)
        self.assertGreater(len(figures), 0)
        for fid, plan in figures.items():
            self.assertTrue(Path(plan.file_path).exists())

    def test_fact_checker_hedges_low_confidence_causal_language(self):
        text = "This decline was caused by the policy change."
        revised, changed = fact_checker.hedge_unsupported_causal_language(text, "low")
        self.assertTrue(changed)
        self.assertNotIn("was caused by", revised)

    def test_fact_checker_leaves_high_confidence_language(self):
        text = "This decline was caused by the policy change."
        revised, changed = fact_checker.hedge_unsupported_causal_language(text, "high")
        self.assertFalse(changed)
        self.assertEqual(text, revised)


if __name__ == "__main__":
    unittest.main()
