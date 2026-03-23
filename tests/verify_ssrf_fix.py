
import asyncio
import ipaddress
from unittest.mock import MagicMock, AsyncMock

# Mocking httpx and other dependencies before importing our modules
import sys
from types import ModuleType

httpx_mock = ModuleType("httpx")
httpx_mock.AsyncClient = MagicMock()
httpx_mock.Request = MagicMock()
httpx_mock.RequestError = type("RequestError", (Exception,), {})
httpx_mock.HTTPError = type("HTTPError", (Exception,), {})
httpx_mock.HTTPStatusError = type("HTTPStatusError", (httpx_mock.HTTPError,), {})
httpx_mock.Response = MagicMock()
sys.modules["httpx"] = httpx_mock

bs4_mock = ModuleType("bs4")
bs4_mock.BeautifulSoup = MagicMock()
sys.modules["bs4"] = bs4_mock

readability_mock = ModuleType("readability")
sys.modules["readability"] = readability_mock

# Now we can import the module to test
from cuba_search.scraper import _is_ssrf_safe, _ssrf_redirect_hook

def test_is_ssrf_safe():
    print("Testing _is_ssrf_safe...")
    # Public URLs
    assert _is_ssrf_safe("https://google.com") is True
    assert _is_ssrf_safe("https://8.8.8.8") is True

    # Private URLs
    assert _is_ssrf_safe("http://localhost") is False
    assert _is_ssrf_safe("http://127.0.0.1") is False
    assert _is_ssrf_safe("http://192.168.1.1") is False
    assert _is_ssrf_safe("http://10.0.0.1") is False
    assert _is_ssrf_safe("http://169.254.169.254") is False
    assert _is_ssrf_safe("http://[::1]") is False
    assert _is_ssrf_safe("http://0.0.0.0") is False

    # New private ranges
    assert _is_ssrf_safe("http://0.1.2.3") is False
    assert _is_ssrf_safe("http://[::]") is False

    # Integer IP formats
    assert _is_ssrf_safe("http://2130706433") is False  # 127.0.0.1
    assert _is_ssrf_safe("http://0x7f000001") is False  # 127.0.0.1

    # Internal hostnames
    assert _is_ssrf_safe("http://service.local") is False
    assert _is_ssrf_safe("http://database.internal") is False

    # Schemes
    assert _is_ssrf_safe("file:///etc/passwd") is False
    assert _is_ssrf_safe("gopher://localhost") is False

    print("✓ _is_ssrf_safe tests passed")

def test_ssrf_redirect_hook():
    print("Testing _ssrf_redirect_hook...")

    # Safe request
    safe_req = MagicMock()
    safe_req.url = "https://example.com"
    try:
        _ssrf_redirect_hook(safe_req)
    except Exception as e:
        assert False, f"Should not have raised an exception for safe URL: {e}"

    # Unsafe request
    unsafe_req = MagicMock()
    unsafe_req.url = "http://127.0.0.1"
    try:
        _ssrf_redirect_hook(unsafe_req)
        assert False, "Should have raised httpx.RequestError for unsafe URL"
    except httpx_mock.RequestError as e:
        print(f"Caught expected error: {e}")

    print("✓ _ssrf_redirect_hook tests passed")

if __name__ == "__main__":
    test_is_ssrf_safe()
    test_ssrf_redirect_hook()
    print("\nAll tests passed successfully!")
