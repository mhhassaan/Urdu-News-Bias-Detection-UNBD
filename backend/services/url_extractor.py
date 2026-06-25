import ipaddress
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

MAX_DOWNLOAD_BYTES = 5_000_000


def _is_public_http_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
            )
        }
        return all(
            not (
                ipaddress.ip_address(address).is_private
                or ipaddress.ip_address(address).is_loopback
                or ipaddress.ip_address(address).is_link_local
                or ipaddress.ip_address(address).is_reserved
            )
            for address in addresses
        )
    except (OSError, ValueError):
        return False


def extract_text_from_url(url):
    """Extract paragraph text from a public HTTP(S) article URL."""
    if not _is_public_http_url(url):
        return ""

    try:
        res = requests.get(
            url,
            headers=HEADERS,
            timeout=(5, 15),
            allow_redirects=True,
            stream=True,
        )
        res.raise_for_status()
        if not _is_public_http_url(res.url):
            return ""

        content_type = res.headers.get("content-type", "").lower()
        if content_type and "text/html" not in content_type:
            return ""

        chunks = []
        downloaded = 0
        for chunk in res.iter_content(chunk_size=65_536):
            downloaded += len(chunk)
            if downloaded > MAX_DOWNLOAD_BYTES:
                return ""
            chunks.append(chunk)

        res._content = b"".join(chunks)
        res.encoding = res.apparent_encoding or res.encoding
        soup = BeautifulSoup(res.text, 'html.parser')

        for element in soup(["script", "style", "nav", "footer", "aside"]):
            element.decompose()

        paragraphs = soup.find_all('p')
        text = " ".join(
            paragraph.get_text(" ", strip=True)
            for paragraph in paragraphs
            if paragraph.get_text(strip=True)
        )
        
        return text.strip()
    except Exception as e:
        print(f"URL extraction failed: {e.__class__.__name__}")
        return ""
