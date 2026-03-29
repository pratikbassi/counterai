#!/usr/bin/env python3
"""
Entry point for URL → image download (placeholder). Same as downloader_entry.py;
kept as a named script for backend jobs and shell use.

  python download_url.py 'https://example.com/image.jpg'
  python download_url.py 'https://example.com/image.jpg' --json
"""

from downloader.entry import main


if __name__ == "__main__":
    main()
