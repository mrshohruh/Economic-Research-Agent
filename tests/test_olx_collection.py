"""Collection-only tests; all network responses are fixtures."""
import json
import sqlite3
from unittest.mock import Mock
import pytest
from uzhousing.ingest.olx import collect, discover_ids
from uzhousing.ingest.olx_client import OLXClient, CollectionError, OfferUnavailable
from uzhousing.ingest.olx_store import save_observations


def offer(i=123, price=500):
    return {"id": i, "title": "Uy", "location": {}, "collection_category": "rent_apartment",
            "params": [{"key": "price", "value": {"value": price, "currency": "USD"}}]}


def reply(data=None, text="", status=200):
    response = Mock(status_code=status, text=text)
    response.json.return_value = data
    return response


def test_detail_unwrap_and_rate_limit():
    session, pause = Mock(), Mock()
    session.get.side_effect = [reply({"data": offer()}), reply(offer(124))]
    client = OLXClient(session, pause=pause)
    assert client.get_offer(123)["id"] == 123
    assert client.get_offer(124)["id"] == 124
    pause.assert_called_once_with(2)
    assert session.get.call_args.kwargs['timeout'] == 30


@pytest.mark.parametrize('status', [401,403,429])
def test_no_retries_on_block(status):
    session = Mock()
    session.get.return_value = reply(status=status)
    with pytest.raises(CollectionError):
        OLXClient(session).get_offer(123)
    assert session.get.call_count == 1


@pytest.mark.parametrize('status', [404,410])
def test_expired_listing(status):
    session = Mock()
    session.get.return_value = reply(status=status)
    with pytest.raises(OfferUnavailable):
        OLXClient(session).get_offer(123)


def test_bad_json_wrong_id_and_invalid_input():
    session = Mock()
    client = OLXClient(session, pause=lambda _: None)
    for invalid in ['../users/me', 'ID4tbNr', -1, True]:
        with pytest.raises(ValueError):
            client.get_offer(invalid)
    assert session.get.call_count == 0
    session.get.return_value = reply({'data': offer(999)})
    with pytest.raises(CollectionError):
        client.get_offer(123)
    session.get.return_value.json.side_effect = ValueError('html')
    with pytest.raises(CollectionError):
        client.get_offer(123)


def test_pagination_bounded_duplicate_stop():
    session = Mock()
    session.get.side_effect = [reply({'data':[offer(1),offer(2)]}), reply({'data':[offer(2),offer(3)]}), reply({'data':[offer(2),offer(3)]})]
    client = OLXClient(session, pause=lambda _: None)
    assert [o['id'] for o in client.iter_offers(category_id=1234, max_pages=5, limit=2)] == [1,2,3]
    assert [c.kwargs['params']['offset'] for c in session.get.call_args_list] == [0,2,4]
    with pytest.raises(ValueError):
        list(client.iter_offers(category_id=1234, max_pages=0))
    with pytest.raises(ValueError):
        client.get_offers(category_id=None)


def test_discover_only_explicit_listing_card_ids():
    html = '<div id="999"></div><div data-cy="l-card" id="123"></div><div data-cy="l-card" data-id="456"></div><div data-cy="l-card" id="ID4tbNr"></div><div data-cy="l-card" id="123"></div>'
    assert discover_ids(html) == ['123','456']


def test_collection_falls_back_to_details_and_stores_history(tmp_path):
    session = Mock()
    html = '<div data-cy="l-card" id="123"></div>'
    session.get.side_effect = [reply(text='User-agent: *\nAllow: /')] + [item for _ in range(4) for item in (reply(text=html), reply({'data':offer()}))] + [reply([{'Rate':'12500','Date':'21.09.2026'}])]
    path = collect(tmp_path, pages=1, session=session, pause=lambda _: None, progress=lambda _: None)
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert len(payload['data']) == 4
    assert all('collection_category' in item for item in payload['data'])
    with sqlite3.connect(tmp_path/'olx_history.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM listing_snapshots').fetchone()[0] == 4
        assert db.execute('SELECT observations FROM listing_observation_history').fetchone()[0] == 1


def test_history_keeps_repricing_and_is_idempotent(tmp_path):
    path = tmp_path/'history.sqlite'
    first = {'collected_at':'2026-09-21T00:00:00Z','data':[offer(price=500)]}
    second = {'collected_at':'2026-09-28T00:00:00Z','data':[offer(price=450)]}
    save_observations(path, first)
    save_observations(path, first)
    save_observations(path, second)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT price FROM listing_snapshots ORDER BY captured_at').fetchall() == [(500.0,),(450.0,)]
        assert db.execute('SELECT observations FROM listing_observation_history').fetchone()[0] == 2
        assert db.execute('SELECT DISTINCT status FROM listing_snapshots').fetchall() == [('observed',)]


def test_robots_applies_to_detail_fallback(tmp_path):
    session = Mock()
    session.get.side_effect = [reply(text='User-agent: *\nDisallow: /api/v1/offers/'), reply(text='<div data-cy="l-card" id="123"></div>')]
    with pytest.raises(CollectionError, match='robots'):
        collect(tmp_path, pages=1, session=session, pause=lambda _: None)
    assert session.get.call_count == 2
    assert not list(tmp_path.rglob('*.json'))


def categorised(i, cid=13):
    return offer(i) | {"category": {"id": cid, "type": "real_estate"}}


def test_verified_category_switches_to_the_bulk_endpoint(tmp_path):
    # Two listings agree on the category, so one paged request replaces one
    # request per listing.
    session = Mock()
    cards = '<div data-cy="l-card" id="11"></div><div data-cy="l-card" id="12"></div>'
    per_category = [reply(text=cards), reply({'data': categorised(11)}),
                    reply({'data': categorised(12)}),
                    reply({'data': [categorised(21), categorised(22)]})]
    session.get.side_effect = ([reply(text='User-agent: *\nAllow: /')]
                               + per_category * 4
                               + [reply([{'Rate': '12500', 'Date': '21.09.2026'}])])
    path = collect(tmp_path, pages=1, session=session, pause=lambda _: None, progress=lambda _: None)
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert [c['strategy'] for c in payload['coverage']] == ['category_api:13'] * 4
    assert len(payload['data']) == 8  # two bulk listings per category
    bulk = [c for c in session.get.call_args_list if (c.kwargs.get('params') or {}).get('category_id')]
    assert [c.kwargs['params']['category_id'] for c in bulk] == [13] * 4


def test_disagreeing_categories_fall_back_instead_of_guessing(tmp_path):
    session = Mock()
    cards = ('<div data-cy="l-card" id="11"></div><div data-cy="l-card" id="12"></div>'
             '<div data-cy="l-card" id="13"></div>')
    per_category = [reply(text=cards), reply({'data': categorised(11, 13)}),
                    reply({'data': categorised(12, 99)}), reply({'data': categorised(13, 13)})]
    session.get.side_effect = ([reply(text='User-agent: *\nAllow: /')]
                               + per_category * 4
                               + [reply([{'Rate': '12500', 'Date': '21.09.2026'}])])
    path = collect(tmp_path, pages=1, session=session, pause=lambda _: None, progress=lambda _: None)
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert [c['strategy'] for c in payload['coverage']] == ['listing_pages'] * 4
    # No request carried a guessed category_id, and the verification fetches
    # were reused rather than repeated.
    assert not [c for c in session.get.call_args_list if (c.kwargs.get('params') or {}).get('category_id')]
    assert len(payload['data']) == 12


def test_offset_cap_stops_paging_instead_of_failing():
    # OLX rejects offset > 1000 with HTTP 400; the pages already read stay valid.
    session = Mock()
    session.get.side_effect = [reply({'data': [offer(i) for i in range(p * 40, p * 40 + 40)]})
                               for p in range(26)]
    client = OLXClient(session, pause=lambda _: None)
    collected = list(client.iter_offers(category_id=1147, max_pages=100, limit=40))
    assert len(collected) == 26 * 40          # offsets 0..1000 inclusive
    assert client.last_stop == 'depth_limit'
    assert max(c.kwargs['params']['offset'] for c in session.get.call_args_list) == 1000


def test_late_http_400_is_survived_but_403_is_not():
    session = Mock()
    session.get.side_effect = [reply({'data': [offer(1), offer(2)]}), reply(status=400)]
    client = OLXClient(session, pause=lambda _: None)
    assert len(list(client.iter_offers(category_id=1147, max_pages=5, limit=2))) == 2
    assert client.last_stop == 'depth_limit'
    # A block mid-run must still stop the run, not be mistaken for a depth cap.
    session.get.side_effect = [reply({'data': [offer(1), offer(2)]}), reply(status=403)]
    client = OLXClient(session, pause=lambda _: None)
    with pytest.raises(CollectionError, match='chekladi'):
        list(client.iter_offers(category_id=1147, max_pages=5, limit=2))


def test_no_page_limit_reads_every_page_the_site_serves():
    # The default asks until the marketplace runs out, not for a preset count.
    session = Mock()
    session.get.side_effect = [reply({'data': [offer(1), offer(2)]}),
                               reply({'data': [offer(3), offer(4)]}),
                               reply({'data': [offer(5)]})]
    client = OLXClient(session, pause=lambda _: None)
    assert [o['id'] for o in client.iter_offers(category_id=1147, limit=2)] == [1, 2, 3, 4, 5]
    assert client.last_stop == 'exhausted' and client.last_pages == 3


def test_unlimited_run_stops_when_a_page_repeats():
    session = Mock()
    session.get.return_value = reply({'data': [offer(i) for i in range(40)]})
    client = OLXClient(session, pause=lambda _: None)
    collected = list(client.iter_offers(category_id=1147, limit=40))
    # Every page repeats the same ids, so the walk ends on the first repeat.
    assert len(collected) == 40 and client.last_stop == 'exhausted'
    assert session.get.call_count == 2


def test_listing_pages_walk_ends_past_the_last_page(tmp_path):
    session = Mock()
    cards = '<div data-cy="l-card" id="11"></div><div data-cy="l-card" id="12"></div>'
    # Category verification disagrees, so the listing-page walk is used; the
    # third request returns a page with no cards at all, which ends it.
    per_category = [reply(text=cards), reply({'data': categorised(11, 13)}),
                    reply({'data': categorised(12, 99)}),
                    reply(text='<div data-cy="l-card" id="21"></div>'),
                    reply({'data': categorised(21, 13)}),
                    reply(text='<html></html>')]
    session.get.side_effect = ([reply(text='User-agent: *\nAllow: /')]
                               + per_category * 4
                               + [reply([{'Rate': '12500', 'Date': '21.09.2026'}])])
    path = collect(tmp_path, session=session, pause=lambda _: None, progress=lambda _: None)
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert [c['stop'] for c in payload['coverage']] == ['exhausted'] * 4
    assert [c['pages_requested'] for c in payload['coverage']] == [3] * 4
    assert len(payload['data']) == 12  # three listings per category


def test_page_count_must_be_positive_or_absent(tmp_path):
    for bad in (0, -1, 1001, 'all'):
        with pytest.raises(ValueError):
            collect(tmp_path, pages=bad, session=Mock(), pause=lambda _: None,
                    progress=lambda _: None)
