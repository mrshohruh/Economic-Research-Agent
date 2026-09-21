"""Smoke tests for the Streamlit interface.

``AppTest`` actually executes app.py in a headless session, so an exception in
the layout code fails the test rather than only showing up in a browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def app():
    instance = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    instance.run()
    return instance


def test_app_runs_without_exceptions(app):
    assert not app.exception, [str(e) for e in app.exception]


def test_header_and_sections_render(app):
    markdown = " ".join(block.value for block in app.markdown)
    assert "Uzbekistan Housing Market Research Agent" in markdown
    headers = [block.value for block in app.subheader]
    assert any("Data" in h for h in headers)
    assert any("Generate the report" in h for h in headers)


def test_sidebar_controls_exist(app):
    assert app.text_input, "expected an API key input"
    labels = [widget.label for widget in app.selectbox]
    assert any("Model" in label for label in labels)
    assert any("language" in label.lower() for label in labels)


def test_run_button_is_disabled_without_data(app):
    run_buttons = [b for b in app.button if "Run the analysis" in b.label]
    assert run_buttons, "the run button should be present"
    assert app.info, "expected the 'choose a data source' hint"


def test_olx_result_shows_downloads_without_empty_analysis(tmp_path, monkeypatch):
    from uzhousing.pipeline import RunResult
    from uzhousing.report import olx_bulletin
    docx = tmp_path / 'report.docx'
    pdf = tmp_path / 'report.pdf'
    docx.write_bytes(b'test')
    pdf.write_bytes(b'%PDF-test')
    monkeypatch.setattr(olx_bulletin, 'run_olx', lambda *a, **kw: RunResult(report_path=docx, pdf_path=pdf))
    instance = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
    next(w for w in instance.checkbox if 'OLX.uz' in w.label).check().run()
    next(w for w in instance.button if 'Run the analysis' in w.label).click().run()
    assert not instance.exception
    assert any('OLX' in item.value for item in instance.success)
    assert not instance.metric
