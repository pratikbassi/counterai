"""
Load Instagram post URL rules from the same JSON as the Rails app (repo config/instagram_post_url.json).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_DEFAULT: dict[str, Any] = {
    "hosts": ["instagram.com", "instagr.am"],
    "post_path_regex": r"^/(?:p|reel|tv)/[A-Za-z0-9_-]+/?$",
}

_dotenv_loaded = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_repo_dotenv() -> None:
    """Load repo-root .env before reading INSTAGRAM_POST_URL_CONFIG (import order safe)."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_repo_root() / ".env", override=False)


def _config_path() -> Path:
    raw = os.environ.get("INSTAGRAM_POST_URL_CONFIG", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _repo_root() / "config" / "instagram_post_url.json"


def load_instagram_post_url_rules() -> tuple[frozenset[str], re.Pattern[str]]:
    _ensure_repo_dotenv()
    path = _config_path()
    data = dict(_DEFAULT)
    try:
        if path.is_file():
            merged = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(merged, dict):
                data.update(merged)
    except (OSError, json.JSONDecodeError, TypeError):
        pass

    hosts_raw = data.get("hosts") or _DEFAULT["hosts"]
    hosts: set[str] = set()
    for h in hosts_raw:
        if isinstance(h, str) and h.strip():
            hn = h.strip().lower().removeprefix("www.")
            hosts.add(hn)

    pattern = data.get("post_path_regex") or _DEFAULT["post_path_regex"]
    if not isinstance(pattern, str):
        pattern = str(_DEFAULT["post_path_regex"])

    return frozenset(hosts), re.compile(pattern, re.I)
