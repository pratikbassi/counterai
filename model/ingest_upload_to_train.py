#!/usr/bin/env python3
"""
Copy every pending image from ``upload/AI`` and ``upload/nonAI`` into
``train_data/`` in **batches** (default: many files per batch) for speed.

Sources are renamed to ``uploaded_<name>`` after each batch completes for that
batch’s rows (copy → append CSV rows → rename). Names starting with
``uploaded`` (any case) are skipped.

CSV format: ``, ``file_name``, ``label``; then index, ``train_data/...``, ``1`` = AI,
``0`` = not AI. Default data root is ``model/data``.

Flags: ``--dry-run``, ``--data-dir``, ``--batch-size``, ``-v`` / ``--verbose``.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import shutil
import sys
import time
import uuid
from pathlib import Path

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".jfif"}
_DEFAULT_DATA = Path(__file__).resolve().parent / "data"
_CSV_HEADER = ("", "file_name", "label")
_LABEL_AI = 1
_LABEL_NON_AI = 0
_LOG = logging.getLogger("ingest_upload_to_train")


def _already_uploaded(path: Path) -> bool:
    return path.name.lower().startswith("uploaded")


def _list_pending(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in _EXTS
        and not _already_uploaded(p)
    )


def _enumerate_jobs(ai: Path, non: Path) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    for folder, lab in ((ai, _LABEL_AI), (non, _LABEL_NON_AI)):
        for src in _list_pending(folder):
            out.append((src, lab))
    return out


def _next_row_index_scan(csv_path: Path) -> int:
    """Full-file scan (fallback if tail parse fails)."""

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0
    highest = -1
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        try:
            next(r)
        except StopIteration:
            return 0
        for row in r:
            if row and row[0].strip().isdigit():
                highest = max(highest, int(row[0].strip()))
    return highest + 1 if highest >= 0 else 0


def _next_row_index(csv_path: Path) -> int:
    """Next row index; tail-reads large CSVs to avoid scanning millions of lines."""

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0
    size = csv_path.stat().st_size
    tail_n = min(size, 262_144)
    try:
        with csv_path.open("rb") as f:
            f.seek(-tail_n, 2)  # os.SEEK_END
            raw = f.read()
        text = raw.decode("utf-8-sig", errors="replace")
        nl = text.find("\n")
        if tail_n < size and nl != -1:
            text = text[nl + 1 :]
        highest = -1
        reader = csv.reader(io.StringIO(text))
        for row in reader:
            if row and row[0].strip().isdigit():
                highest = max(highest, int(row[0].strip()))
        if highest >= 0:
            return highest + 1
    except OSError as e:
        _LOG.debug("tail index read failed: %s, falling back to full scan", e)
    return _next_row_index_scan(csv_path)


def _mark_uploaded(src: Path) -> None:
    parent = src.parent
    candidate = parent / f"uploaded_{src.name}"
    if not candidate.exists():
        src.rename(candidate)
        return
    alt = parent / f"uploaded_{uuid.uuid4().hex[:8]}_{src.name}"
    src.rename(alt)


def _build_job_list(
    jobs_src: list[tuple[Path, int]], train_dir: Path
) -> list[tuple[Path, Path, str, int]]:
    out: list[tuple[Path, Path, str, int]] = []
    for src, lab in jobs_src:
        name = f"{src.stem}_{uuid.uuid4().hex[:10]}{src.suffix.lower()}"
        dst = train_dir / name
        out.append((src, dst, f"train_data/{name}", lab))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA,
        help=f"Dataset root (default: {_DEFAULT_DATA})",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions only.")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="Files per batch: copy N, append N CSV lines, rename N (default: 256).",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging (per-batch timings).",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    root = args.data_dir.resolve()
    ai = root / "upload" / "AI"
    non = root / "upload" / "nonAI"
    train_dir = root / "train_data"
    csv_path = root / "train.csv"
    batch_size = max(1, args.batch_size)

    ai.mkdir(parents=True, exist_ok=True)
    non.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    jobs_src = _enumerate_jobs(ai, non)
    _LOG.info(
        "discovered %d pending file(s) in %.2fs",
        len(jobs_src),
        time.perf_counter() - t0,
    )
    if not jobs_src:
        _LOG.warning("No pending images in upload/AI or upload/nonAI.")
        return 0

    jobs = _build_job_list(jobs_src, train_dir)

    if args.dry_run:
        idx = _next_row_index(csv_path)
        for j, (src, dst, rel, lab) in enumerate(jobs):
            print(f"{idx + j},{rel},{lab}  <- {src} -> {dst} ; then rename -> uploaded_{src.name}")
        print(f"[dry-run] {len(jobs)} file(s).", file=sys.stderr)
        return 0

    train_dir.mkdir(parents=True, exist_ok=True)
    t_idx = time.perf_counter()
    idx = _next_row_index(csv_path)
    _LOG.info("next CSV row index=%d (resolved in %.2fs)", idx, time.perf_counter() - t_idx)

    fresh = not csv_path.exists() or csv_path.stat().st_size == 0
    enc = "utf-8-sig" if fresh else "utf-8"
    mode = "w" if fresh else "a"
    total = len(jobs)
    done = 0

    with csv_path.open(mode, encoding=enc, newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(list(_CSV_HEADER))
        for start in range(0, total, batch_size):
            batch = jobs[start : start + batch_size]
            t_batch = time.perf_counter()
            for src, dst, _rel, _lab in batch:
                shutil.copyfile(src, dst)
            t_copy = time.perf_counter()
            lines = [
                [str(idx + start + i), batch[i][2], str(batch[i][3])]
                for i in range(len(batch))
            ]
            w.writerows(lines)
            t_csv = time.perf_counter()
            for src, _dst, _rel, _lab in batch:
                _mark_uploaded(src)
            t_end = time.perf_counter()
            done += len(batch)
            _LOG.info(
                "batch %d-%d/%d copy=%.2fs csv=%.2fs rename=%.2fs total=%.2fs",
                start + 1,
                done,
                total,
                t_copy - t_batch,
                t_csv - t_copy,
                t_end - t_csv,
                t_end - t_batch,
            )

    _LOG.info("finished: copied %d file(s) -> %s; updated %s", done, train_dir, csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
