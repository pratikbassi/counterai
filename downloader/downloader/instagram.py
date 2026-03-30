"""
Instagram post pages: HTML is often a client shell or error page without media.
We use the public oEmbed endpoint and collect candidate image URLs from JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

import requests

from downloader.http_constants import BROWSER_USER_AGENT
from downloader.instagram_config import load_instagram_post_url_rules

OEMBED_URL = "https://www.instagram.com/api/v1/oembed/"

# Lazy: load_instagram_post_url_rules() reads .env — avoid running at import before dotenv (see entry.py).
_rules_cache: tuple[frozenset[str], re.Pattern[str]] | None = None


def _post_url_rules() -> tuple[frozenset[str], re.Pattern[str]]:
    global _rules_cache
    if _rules_cache is None:
        _rules_cache = load_instagram_post_url_rules()
    return _rules_cache


def is_instagram_post_url(url: str) -> bool:
    try:
        p = urlparse(url.strip())
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.netloc or "").lower()
    if ":" in host:
        host = host.split(":")[0]
    host = host.removeprefix("www.")
    hosts, path_re = _post_url_rules()
    if host not in hosts:
        return False
    path = p.path or "/"
    return path_re.match(path) is not None


def _is_trusted_image_cdn(u: str) -> bool:
    try:
        host = (urlparse(u).netloc or "").lower().split(":")[0]
    except Exception:
        return False
    return (
        host.endswith(".cdninstagram.com")
        or host == "cdninstagram.com"
        or host.endswith(".fbcdn.net")
        or host == "fbcdn.net"
    )


def _collect_image_urls(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("thumbnail_url", "display_url", "url") and isinstance(v, str):
                if v.startswith("http") and _is_trusted_image_cdn(v):
                    if re.search(r"\.(jpe?g|webp)(\?|$)", v, re.I):
                        out.append(v)
            else:
                _collect_image_urls(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_image_urls(item, out)


def instagram_oembed_images(page_url: str) -> dict[str, Any]:
    """
    Fetch oEmbed JSON and return thumbnail hints plus all image URLs found in the payload.
    """
    if not is_instagram_post_url(page_url):
        raise ValueError("Not a recognized Instagram post/reel/tv URL")

    r = requests.get(
        OEMBED_URL,
        params={"url": page_url.strip()},
        headers={"User-Agent": BROWSER_USER_AGENT},
        timeout=45,
    )
    r.raise_for_status()
    try:
        data = r.json()
    except ValueError as e:
        raise LookupError("Instagram oEmbed response was not valid JSON") from e

    candidates: list[str] = []
    _collect_image_urls(data, candidates)
    raw = json.dumps(data)
    for m in re.finditer(
        r"https://[a-zA-Z0-9./?&=%_-]+\.(?:jpe?g|webp)", raw, re.I
    ):
        u = m.group(0).replace("\\/", "/")
        if _is_trusted_image_cdn(u):
            candidates.append(u)

    seen: set[str] = set()
    uniq: list[str] = []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    if not uniq:
        raise LookupError(
            "Instagram oEmbed returned no image URLs (post may be private or removed)"
        )

    tw = int(data.get("thumbnail_width") or 0)
    th = int(data.get("thumbnail_height") or 0)
    thumb = data.get("thumbnail_url")

    return {
        "candidates": uniq,
        "thumbnail_url": thumb if isinstance(thumb, str) else None,
        "thumbnail_width": tw,
        "thumbnail_height": th,
    }
