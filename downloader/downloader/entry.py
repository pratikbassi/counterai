"""
CLI for downloading image(s) from a URL. Placeholder: validates URL and reports status.
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download image(s) from a URL (placeholder — no network fetch yet)."
    )
    parser.add_argument("url", type=str, help="HTTP(S) URL of an image or page")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit a single JSON object on stdout for programmatic callers.",
    )
    return parser.parse_args()


def _validate_url(url: str) -> str | None:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return "URL must start with http:// or https://"
    if not parsed.netloc:
        return "URL is missing a host"
    return None


def run(url: str) -> dict:
    """
    Placeholder run: validate URL and return a structured result.

    Future: fetch URL, detect content-type, save under output dir, return paths.
    """
    err = _validate_url(url)
    if err:
        return {"ok": False, "error": err, "url": url.strip()}

    return {
        "ok": True,
        "status": "placeholder",
        "url": url.strip(),
        "message": "Download not implemented yet; URL validated only.",
        "saved_paths": [],
    }


def main() -> None:
    args = parse_args()
    result = run(args.url)

    if args.json_output:
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        sys.exit(0 if result.get("ok") else 1)

    if result.get("ok"):
        print(f"URL: {result['url']}")
        print(f"Status: {result['status']}")
        print(result["message"])
    else:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
