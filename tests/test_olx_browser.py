"""Browser transport tests; no browser is launched and no network is used."""
import json
from unittest.mock import Mock

import pytest

from uzhousing.ingest.olx import collect
from uzhousing.ingest.olx_browser import BrowserResponse, BrowserSession
from uzhousing.ingest.olx_client import CollectionError


def session_with(page):
    """A session whose browser is already 'started', so _start() is a no-op."""
    session = BrowserSession()
    session._page = page
    return session


def page_serving(body, *, status=200, content_type="text/html", raw="RAW BODY"):
    page = Mock()
    navigation = Mock(status=status, headers={"content-type": content_type})
    navigation.text.return_value = raw
    page.goto.return_value = navigation
    page.content.return_value = body
    return page


def test_html_is_read_from_the_dom_not_the_raw_body():
    # The listing data is injected by the page's own scripts, so the rendered
    # DOM is what collection needs -- not the server's first response.
    page = page_serving("<html>rendered</html>", raw="<html>before scripts</html>")
    response = session_with(page).get("https://www.olx.uz/x/")
    assert response.text == "<html>rendered</html>"
    assert response.status_code == 200


@pytest.mark.parametrize("content_type", ["text/plain", "application/json"])
def test_non_html_is_read_raw(content_type):
    # Chrome wraps robots.txt and JSON in a viewer; parsing that would fail.
    page = page_serving("<html><pre>User-agent: *</pre></html>",
                        content_type=content_type, raw="User-agent: *\nAllow: /")
    assert session_with(page).get("https://www.olx.uz/robots.txt").text == "User-agent: *\nAllow: /"
    page.content.assert_not_called()


def test_query_parameters_are_appended():
    page = page_serving("<html></html>")
    session_with(page).get("https://www.olx.uz/x/", params={"page": 3})
    assert page.goto.call_args[0][0] == "https://www.olx.uz/x/?page=3"


def test_challenge_page_is_fatal_and_not_worked_around():
    page = page_serving("<html><h1>403 ERROR</h1>Request blocked.</html>")
    with pytest.raises(CollectionError, match="to'sdi"):
        session_with(page).get("https://www.olx.uz/x/")


def test_navigation_failure_is_reported_not_retried():
    page = Mock()
    page.goto.side_effect = RuntimeError("Timeout 30000ms exceeded")
    with pytest.raises(CollectionError):
        session_with(page).get("https://www.olx.uz/x/")
    assert page.goto.call_count == 1


def test_response_mimics_requests():
    assert BrowserResponse("u", 200, '{"a": 1}').json() == {"a": 1}
    BrowserResponse("u", 200, "").raise_for_status()
    with pytest.raises(CollectionError):
        BrowserResponse("u", 403, "").raise_for_status()


def test_browser_session_is_closed_after_collection(tmp_path, monkeypatch):
    offer = {"id": 1, "title": "Uy", "location": {}, "params": []}
    page = Mock()
    replies = [("User-agent: *\nAllow: /", "text/plain")] + [
        ("<script>" + json.dumps({"data": [offer]}) + "</script>", "text/html")] * 4 + [
        (json.dumps([{"Rate": "12500", "Date": "21.09.2026"}]), "application/json")]

    def goto(url, **kwargs):
        body, content_type = replies.pop(0)
        page.content.return_value = body
        navigation = Mock(status=200, headers={"content-type": content_type})
        navigation.text.return_value = body
        return navigation

    page.goto.side_effect = goto
    session = session_with(page)
    monkeypatch.setattr("uzhousing.ingest.olx_browser.BrowserSession",
                        lambda **kwargs: session)
    collect(tmp_path, pages=1, browser=True, pause=lambda _: None, progress=lambda _: None)
    assert session._page is None  # closed even though collection owned it
