"""Browser-backed transport for OLX, used in place of a plain HTTP session.

Plain HTTP requests to www.olx.uz are refused by the site's CDN firewall with
HTTP 403 before any page is served, including /robots.txt. A real browser
requests the same public pages a visitor would see, so collection is attempted
through one instead.

No login is performed, no CAPTCHA is solved and no access control is bypassed.
A challenge page that does not clear on its own is a fatal collection error,
exactly as an HTTP 403 is. The object exposed here mimics the small part of
``requests.Session`` that :mod:`uzhousing.ingest.olx_client` uses, so robots.txt
enforcement, request pacing, schema validation and snapshot storage are
unchanged by the switch.
"""
from __future__ import annotations

from urllib.parse import urlencode

from .olx_client import CollectionError

#: Set by the caller; the page is given this long to settle before it is read.
DEFAULT_TIMEOUT = 30

_BLOCK_MARKERS = ("Request blocked", "Access Denied", "Are you a robot")


class BrowserUnavailable(CollectionError):
    """Playwright or a usable browser is not installed on this machine."""


class BrowserResponse:
    """The subset of ``requests.Response`` the OLX client and FX lookup use."""

    def __init__(self, url: str, status_code: int, text: str):
        self.url = url
        self.status_code = status_code
        self.text = text

    def json(self):
        import json

        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise CollectionError(f"HTTP {self.status_code}: {self.url}")


class BrowserSession:
    """A lazily started browser that serves pages through a ``get`` method.

    ``headless=False`` shows the window, which is the setting to try first when
    a run is refused: it makes the refusal visible rather than guessed at.
    """

    def __init__(self, *, headless: bool = True, channel: str | None = "chrome",
                 locale: str = "ru-RU", timeout: int = DEFAULT_TIMEOUT):
        self.headless = headless
        self.channel = channel
        self.locale = locale
        self.timeout = timeout
        self._playwright = None
        self._browser = None
        self._page = None

    # -- lifecycle ---------------------------------------------------------
    def _start(self):
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserUnavailable(
                "Brauzer orqali yig'ish uchun Playwright kerak. O'rnatish: "
                "pip install playwright  (va kerak bo'lsa: python -m playwright install chromium)"
            ) from exc
        self._playwright = sync_playwright().start()
        launch = {"headless": self.headless}
        try:
            self._browser = self._playwright.chromium.launch(channel=self.channel, **launch)
        except Exception:
            # No system Chrome; fall back to Playwright's own Chromium build.
            try:
                self._browser = self._playwright.chromium.launch(**launch)
            except Exception as exc:
                self.close()
                raise BrowserUnavailable(
                    "Brauzer ishga tushmadi. O'rnatish: python -m playwright install chromium"
                ) from exc
        context = self._browser.new_context(
            locale=self.locale,
            extra_http_headers={"Accept-Language": "uz-UZ,uz;q=0.9,ru;q=0.8"},
        )
        context.set_default_timeout(self.timeout * 1000)
        self._page = context.new_page()

    def close(self):
        for attribute in ("_browser", "_playwright"):
            handle = getattr(self, attribute, None)
            if handle is not None:
                try:
                    handle.close() if attribute == "_browser" else handle.stop()
                except Exception:
                    pass
            setattr(self, attribute, None)
        self._page = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
        return False

    # -- transport ---------------------------------------------------------
    def get(self, url, *, params=None, timeout=None, headers=None, **_ignored):
        """Navigate to ``url`` and return its body.

        ``headers`` is accepted and ignored: the browser sends its own, and
        overriding them would defeat the point of using one.
        """
        self._start()
        if params:
            url = url + ("&" if "?" in url else "?") + urlencode(params)
        timeout_ms = (timeout or self.timeout) * 1000
        try:
            response = self._page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception as exc:
            raise CollectionError(f"Brauzer sahifani ocholmadi: {url} ({exc})") from exc
        if response is None:
            raise CollectionError(f"Brauzer javob olmadi: {url}")
        status = response.status
        content_type = (response.headers or {}).get("content-type", "")
        if "html" not in content_type.lower():
            # robots.txt and JSON endpoints: the raw body, not the DOM viewer
            # Chrome wraps them in, which would corrupt both parsers.
            try:
                body = response.text()
            except Exception as exc:
                raise CollectionError(f"Brauzer javob matnini o'qiy olmadi: {url}") from exc
        else:
            try:
                self._page.wait_for_load_state("networkidle", timeout=timeout_ms)
            except Exception:
                pass  # A page that never goes idle is still readable.
            body = self._page.content()
        if status == 200 and any(marker in body[:4000] for marker in _BLOCK_MARKERS):
            raise CollectionError(
                "OLX brauzer so'rovini ham to'sdi (himoya sahifasi qaytdi). Jonli ma'lumot "
                "olinmadi. Himoya tekshiruvini chetlab o'tishga urinilmaydi."
            )
        return BrowserResponse(url, status, body)
