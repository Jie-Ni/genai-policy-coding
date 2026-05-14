"""Shared helpers for the IJETHE GenAI policy pipeline."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = ROOT / "data_processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PROMPTS = ROOT / "scripts" / "prompts"

USER_AGENT = (
    "IJETHE-policy-research/1.0 "
    "(University of Innsbruck Digital Science Center; "
    "contact: ni.jie@uibk.ac.at)"
)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_csv(p: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fieldnames = list(fieldnames) if fieldnames else list(rows[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def load_csv_dicts(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    return [{k.lstrip("﻿"): v for k, v in r.items()} for r in rows]


def append_csv(p: Path, row: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    is_new = not p.exists() or p.stat().st_size == 0
    with p.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def setup_logging(name: str) -> logging.Logger:
    ensure_dir(RESULTS)
    log_path = RESULTS / f"{name}.log"
    handlers = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(name)


class HostThrottler:
    """Per-host throttle: enforces a minimum interval between consecutive
    requests to the same host. Polite scraping convention is 1 req per 3 s
    per host; rate-limited sites get exponential backoff up to 60 s.
    """

    def __init__(self, min_interval_s: float = 3.0, max_interval_s: float = 60.0):
        self.min_interval_s = min_interval_s
        self.max_interval_s = max_interval_s
        self._last: dict[str, float] = {}
        self._backoff: dict[str, float] = {}

    def wait(self, host: str) -> None:
        now = time.time()
        last = self._last.get(host, 0)
        elapsed = now - last
        interval = max(self.min_interval_s, self._backoff.get(host, 0))
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last[host] = time.time()

    def back_off(self, host: str) -> None:
        cur = self._backoff.get(host, self.min_interval_s)
        self._backoff[host] = min(self.max_interval_s, cur * 2)

    def reset_backoff(self, host: str) -> None:
        self._backoff.pop(host, None)
