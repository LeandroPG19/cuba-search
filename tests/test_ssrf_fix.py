import pytest
from cuba_search.scraper import _is_ssrf_safe

@pytest.mark.parametrize("url, expected", [
    ("https://google.com", True),
    ("http://1.1.1.1", True),
    ("https://8.8.8.8/test", True),

    # Private IPv4
    ("http://127.0.0.1", False),
    ("http://10.0.0.1", False),
    ("http://192.168.1.1", False),
    ("http://172.16.0.1", False),
    ("http://169.254.169.254", False),
    ("http://0.0.0.0", False),

    # Private IPv6
    ("http://[::1]", False),
    ("http://[fc00::1]", False),
    ("http://[fe80::1]", False),
    ("http://[::]", False),

    # Decimal/Hex/Octal bypasses
    ("http://2130706433", False),  # 127.0.0.1
    ("http://0x7f000001", False),  # 127.0.0.1
    ("http://017700000001", False), # 127.0.0.1 (octal)
    ("http://3232235777", False),  # 192.168.1.1

    # Local hostname bypasses
    ("http://localhost", False),
    ("http://test.local", False),
    ("http://service.internal", False),
    ("http://router.lan", False),
    ("http://home.home.arpa", False),

    # Schemes
    ("file:///etc/passwd", False),
    ("gopher://localhost", False),
    ("ftp://1.2.3.4", False),
])
def test_is_ssrf_safe(url, expected):
    assert _is_ssrf_safe(url) == expected

import httpx
import asyncio
from cuba_search.scraper import _ssrf_redirect_hook

@pytest.mark.asyncio
async def test_ssrf_redirect_hook():
    # Test that the hook raises for unsafe URL
    request = httpx.Request("GET", "http://127.0.0.1")
    with pytest.raises(httpx.ConnectError):
        _ssrf_redirect_hook(request)

    # Test that the hook allows safe URL
    request = httpx.Request("GET", "https://google.com")
    _ssrf_redirect_hook(request) # Should not raise
