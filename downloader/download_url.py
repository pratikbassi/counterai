#!/usr/bin/env python3
"""
Entry point for URL → image download. Used by DownloaderJob and make run.

  python download_url.py 'https://example.com/image.jpg'
  python download_url.py 'https://www.instagram.com/p/SHORTCODE/' --json
"""

from downloader.entry import main


if __name__ == "__main__":
    main()
