"""
CLI for downloading image(s) from a URL. Used by DownloaderJob and make run.

With --json, only a single JSON object is written to stdout; use stderr for any diagnostics.
The Rails DownloaderJob only runs this script for Instagram permalinks (see config/instagram_post_url.json).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_repo_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_repo_root() / ".env", override=False)


# Load .env before importing downloader.* so os.environ is populated for other vars.
# Instagram rules load lazily on first URL check and call _ensure_repo_dotenv() in instagram_config.
_load_repo_dotenv()

from downloader.fetch import download_image_for_url


def _max_url_bytes() -> int:
    raw = os.environ.get("DOWNLOADER_MAX_URL_BYTES", "").strip()
    if not raw:
        return 8192
    try:
        n = int(raw)
        return n if n > 0 else 8192
    except ValueError:
        return 8192


def _output_base() -> Path:
    raw = os.environ.get("DOWNLOADER_OUTPUT_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (_repo_root() / "backend" / "storage" / "uploads").resolve()


MAX_URL_BYTES = _max_url_bytes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download an image from a URL (direct image, HTML og:image, or Instagram post). "
            "Note: DownloaderJob on the backend only invokes this for Instagram permalinks."
        )
    )
    parser.add_argument("url", type=str, help="HTTP(S) URL of an image, page, or Instagram /p|reel|tv/ link")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit a single JSON object on stdout for programmatic callers.",
    )
    return parser.parse_args()


def _validate_url(url: str) -> str | None:
    if len(url.encode("utf-8")) > MAX_URL_BYTES:
        return f"URL exceeds maximum length ({MAX_URL_BYTES} bytes)"
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return "URL must start with http:// or https://"
    if not parsed.netloc:
        return "URL is missing a host"
    return None


def run(url: str) -> dict:
    err = _validate_url(url)
    if err:
        return {"ok": False, "error": err, "url": url.strip()}

    out = download_image_for_url(url, _output_base())
    if out.get("ok"):
        out.setdefault("url", url.strip())
    return out


def main() -> None:
    args = parse_args()
    result = run(args.url)

    if args.json_output:
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        sys.exit(0 if result.get("ok") else 1)

    if result.get("ok"):
        print(f"URL: {result.get('url', args.url)}")
        print(f"Saved: {result['saved_path']}")
        print(f"SHA256: {result['sha256']} ({result['bytes']} bytes)")
        print(f"Source: {result.get('source', '')}")
    else:
        print(f"Error: {result.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
