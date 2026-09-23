import json
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import pytest
from uzhousing.ingest.olx import collect, extract_offers, CollectionError, CATEGORIES
from uzhousing.report.olx_bulletin import Tbl, normalise, regional_table, sections, write


def table_of(content, caption, index=0):
    """A table by a fragment of its caption, wherever the report put it.

    Detail tables are moved to the appendix and numbered there, so a test asks
    for the table by what it is rather than by the section it started in.
    """
    found = [frame for section in content for block in section.blocks
             if isinstance(block, Tbl) and caption.lower() in block.caption.lower()
             for frame in [block.frame]]
    return found[index] if found else pd.DataFrame()


def advert(i=1, category="rent_apartment", price=500, currency="USD", market="primary"):
    return {"id": i, "title": "Uy", "url": f"https://www.olx.uz/d/obyavlenie/{i}",
            "created_time": "2020-01-01", "collection_category": category,
            "location": {"region": {"name": "Toshkent"}, "city": {"name": "Toshkent"}, "district": {"name": "Chilonzor"}},
            "params": [{"key": "price", "value": {"value": price, "currency": currency}},
                       {"key": "total_area", "value": {"key": "50"}},
                       {"key": "number_of_rooms", "value": {"key": "2"}},
                       {"key": "market", "value": {"key": market}}]}


def snapshot():
    data = [advert(i) for i in range(20)] + [advert(i+100, "sale_apartment", 50000) for i in range(20)]
    return {"data": data, "collected_at": "2026-09-21T00:00:00+00:00", "fx": {"rate": 12500, "date": "21.09.2026", "source": "https://cbu.uz/"}, "coverage": []}


def response(text="", status=200, payload=None):
    r = Mock(status_code=status, text=text)
    r.json.return_value = payload
    if status >= 400:
        r.raise_for_status.side_effect = RuntimeError("HTTP error")
    return r


def test_extract_nested_and_json_parse():
    payload = {"listing": {"ads": [advert(), advert()]}}
    assert len(extract_offers('<script type="application/json">' + json.dumps(payload) + '</script>')) == 1
    assert len(extract_offers('<script>window.state = JSON.parse(' + json.dumps(json.dumps(payload)) + ')</script>')) == 1
    assert extract_offers('<h1>Access denied</h1>') == []


@pytest.mark.parametrize("code", [401,403,429])
def test_blocked_never_saves_snapshot(tmp_path, code):
    client = Mock()
    client.get.return_value = response(status=code)
    with pytest.raises(CollectionError):
        collect(tmp_path, session=client, pause=lambda _: None)
    assert not list(tmp_path.rglob("*.json"))


def test_success_saves_provenance_and_no_contacts(tmp_path):
    client = Mock()
    ad = advert()
    ad["user"] = {"phone": "private"}
    html = '<script type="application/json">' + json.dumps({"data": [ad]}) + '</script>'
    client.get.side_effect = [response('User-agent: *\nAllow: /')] + [response(html) for _ in CATEGORIES] + [response(payload=[{"Rate": "12500", "Date": "21.09.2026"}])]
    path = collect(tmp_path, pages=1, session=client, pause=lambda _: None, progress=lambda _: None)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["coverage"]) == 4
    assert payload["fx"]["rate"] == 12500
    assert all("user" not in item for item in payload["data"])


def test_robots_disallow(tmp_path):
    client = Mock()
    client.get.return_value = response('User-agent: *\nDisallow: /nedvizhimost/')
    with pytest.raises(CollectionError, match="robots"):
        collect(tmp_path, session=client, pause=lambda _: None)
    assert client.get.call_count == 1


def test_separate_rents_and_sales_and_fx():
    payload = snapshot()
    payload['data'].append(advert(500, 'rent_house', 6250000, 'UZS', 'unknown'))
    frame = normalise(payload)
    assert frame.loc[frame.listing_id == 500, 'price_usd'].iloc[0] == 500
    assert set(frame.kind) == {'sale','rent'}
    assert frame.loc[frame.listing_id == 500, 'market'].iloc[0] == 'Aniqlanmagan'
    yield_table = table_of(sections(payload, frame), "yalpi ko'rsatkich")
    assert yield_table.iloc[0,-1] == 12.0
    assert regional_table(frame,'sale','Birlamchi').iloc[0,-1] == 12.5


def test_thin_groups_and_unknown_market():
    payload = snapshot()
    payload['data'] = [advert(market='unknown')]
    frame = normalise(payload)
    assert pd.isna(regional_table(frame,'rent').iloc[0,-1])
    assert regional_table(frame,'sale','Birlamchi').empty


def test_full_uzbek_pdf_docx(tmp_path):
    path = tmp_path / 'SYNTHETIC_DEMO.json'
    path.write_text(json.dumps(snapshot()), encoding='utf-8')
    docx, pdf, log = write(path, tmp_path, 'SINOV NAMUNASI - suniy malumotlar')
    assert docx.exists() and pdf.read_bytes().startswith(b'%PDF') and log.exists()
    from docx import Document
    text = '\n'.join(p.text for p in Document(docx).paragraphs)
    assert 'IJARA BOZORI' in text and 'METODOLOGIYA VA MANBALAR' in text
    assert 'Executive summary' not in text


def test_missing_rooms_and_area_do_not_crash():
    payload = snapshot()
    for ad in payload['data']:
        ad['params'] = [p for p in ad['params'] if p['key'] not in ('total_area','number_of_rooms')]
    frame = normalise(payload)
    assert frame.sqm_usd.isna().all()
    assert table_of(sections(payload, frame), "yalpi ko'rsatkich").empty


def test_late_failure_does_not_save_partial_run(tmp_path):
    client = Mock()
    html = '<script type="application/json">' + json.dumps({'data':[advert()]}) + '</script>'
    client.get.side_effect = [response('User-agent: *\nAllow: /'), response(html), response(status=403)]
    with pytest.raises(CollectionError):
        collect(tmp_path, pages=1, session=client, pause=lambda _: None)
    assert not list(tmp_path.rglob('*.json'))


def test_cli_rejects_conflicting_sources():
    from uzhousing.cli import main
    assert main(['--olx', '--data', 'file.json']) == 2


def test_encoded_hydration_state():
    html = '<script>window.__PRERENDERED_STATE__ = ' + json.dumps(json.dumps({'data':[advert()]})) + ';</script>'
    assert len(extract_offers(html)) == 1
