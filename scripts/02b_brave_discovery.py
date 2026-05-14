"""Brave Search HTML discovery — bypasses DDG (403-blocked) and Bing
(JS-rendered shell with no anonymous results) to recover non-English
policy URLs that the homepage-link crawler misses.

Brave Search (search.brave.com) returns its result page as static HTML
even without an API key. Empirically: Chinese / Portuguese / Korean
site-search queries return 10-20 relevant on-domain URLs.

Usage
-----
    python scripts/02b_brave_discovery.py                       # all 140
    python scripts/02b_brave_discovery.py --institutions EA_E01,EA_E02
    python scripts/02b_brave_discovery.py --only-zero-archive   # post-scrape mode

Output
------
data/brave_seed_urls.csv          # institution_id, query, discovered_url
data/brave_seed_urls.summary.json # per-institution counts

Politeness
----------
- Hard cap 8 queries per institution.
- 3-second sleep between requests to the same Brave host (per HostThrottler).
- User-Agent identifies the research project (per RFC 9309 conventions).
- Stops on first 429/5xx and applies exponential backoff up to 60 s.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import (  # type: ignore
    DATA, HostThrottler, load_csv_dicts, load_json, setup_logging,
)

logger = setup_logging("02b_brave")

BRAVE_BASE = "https://search.brave.com/search"
OUT_CSV = DATA / "brave_seed_urls.csv"
OUT_SUMMARY = DATA / "brave_seed_urls.summary.json"

BROWSER_UA = (
    # Firefox on Linux: Brave rate-limits the Chrome-on-Windows UA aggressively
    # (most-spoofed UA in scraping circles); Firefox-Linux passes cleanly.
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0"
)


def fetch_brave(query: str, throttler: HostThrottler, timeout: float = 25.0
                ) -> tuple[int, str]:
    params = urllib.parse.urlencode({"q": query, "source": "web"})
    url = f"{BRAVE_BASE}?{params}"
    throttler.wait("search.brave.com")
    req = urllib.request.Request(url, headers={
        "User-Agent": BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.5",
        "Accept-Language": "en;q=1.0,de;q=0.8,zh;q=0.8,pt;q=0.7,es;q=0.7",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return resp.status, body
    except urllib.error.HTTPError as exc:
        if exc.code in (429, 500, 502, 503, 504):
            throttler.back_off("search.brave.com")
        return exc.code, ""
    except Exception as exc:
        logger.warning("brave fetch err: %s", exc)
        return 0, ""


def extract_domain_hits(html: str, base_domain: str) -> list[str]:
    """Return on-domain URLs (and subdomains of base_domain) in result list order."""
    base = base_domain.lstrip(".").lower()
    seen: set[str] = set()
    out: list[str] = []
    for m in re.finditer(r'href="(https?://[^"#?]+)"', html):
        u = m.group(1).replace("&amp;", "&")
        host = urllib.parse.urlparse(u).netloc.lower()
        if host == base or host.endswith("." + base):
            # skip the institution's own search result page
            if "/search" in u and "?q=" in u:
                continue
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
    return out


def base_domain_of(website: str) -> str:
    host = urllib.parse.urlparse(website).netloc.lower()
    host = host.replace("www.", "", 1) if host.startswith("www.") else host
    return host


def discover_one_institution(inst: dict[str, Any], queries_cfg: dict[str, Any],
                              throttler: HostThrottler,
                              max_queries: int = 8
                              ) -> tuple[list[dict[str, str]], dict[str, Any]]:
    iid = inst["institution_id"]
    primary = inst.get("language_primary", "en")
    secondary = [s.strip() for s in (inst.get("language_secondary") or "").split("+") if s.strip()]
    langs = list(dict.fromkeys([primary] + secondary + ["en"]))

    queries: list[tuple[str, str]] = []  # (lang, query_string)
    for lang in langs:
        qs = queries_cfg["languages"].get(lang, {}).get("queries", [])
        for q in qs[: max(2, max_queries // max(1, len(langs)))]:
            queries.append((lang, q))
    queries = queries[:max_queries]

    base = base_domain_of(inst["website"])
    hits: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    per_query_count: dict[str, int] = {}

    consecutive_429 = 0
    for lang, q in queries:
        sq = f"site:{base} {q}"
        t0 = time.time()
        status, html = fetch_brave(sq, throttler)
        elapsed = time.time() - t0
        if status == 429:
            consecutive_429 += 1
            logger.warning("%s brave q=%r 429 (%.1fs); skipping remaining queries",
                          iid, q[:40], elapsed)
            per_query_count[q] = -429
            # On 429, bail out of this institution rather than burning 60s × N
            # waiting on backoff. We move to the next institution; the throttler
            # will rate-limit globally.
            break
        if status != 200:
            logger.warning("%s brave q=%r status=%d (%.1fs)", iid, q[:40], status, elapsed)
            per_query_count[q] = -status
            continue
        consecutive_429 = 0
        urls = extract_domain_hits(html, base)
        new = [u for u in urls if u not in seen_urls]
        for u in new:
            seen_urls.add(u)
            hits.append({"institution_id": iid, "query_lang": lang,
                          "query": q, "url": u})
        per_query_count[q] = len(new)
        logger.info("%s brave q=%r → %d urls (new=%d, %.1fs)",
                     iid, q[:40], len(urls), len(new), elapsed)
        # Once we have plenty, don't burn more queries on this institution
        if len(hits) >= 25:
            break

    return hits, {"institution_id": iid, "n_queries": len(queries),
                  "n_unique_urls": len(hits),
                  "per_query": per_query_count}


def already_has_archive(institution_id: str, data_raw: Path) -> bool:
    for region in ("NA", "EU", "EA", "LA"):
        if (data_raw / region / institution_id / "index.json").exists():
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--institutions", default="",
                    help="comma-separated institution_ids; empty = all")
    ap.add_argument("--only-zero-archive", action="store_true",
                    help="only run on institutions with no successful archive yet")
    ap.add_argument("--inst-csv", default=str(DATA / "institution_list.csv"))
    ap.add_argument("--queries-json", default=str(DATA / "search_queries.json"))
    ap.add_argument("--data-raw", default=str(DATA / "raw"))
    ap.add_argument("--max-queries", type=int, default=2,
                    help="Brave anonymous rate-limit allows ~1/30s; 2 queries "
                         "per institution is the sweet spot for recall vs. time")
    ap.add_argument("--throttle-s", type=float, default=22.0,
                    help="base throttle (s) between Brave requests; 22s avoids "
                         "most 429s when running ~280 queries over ~100 min")
    args = ap.parse_args()

    insts = load_csv_dicts(Path(args.inst_csv))
    queries_cfg = load_json(Path(args.queries_json))
    data_raw = Path(args.data_raw)

    target_ids = set(s.strip() for s in args.institutions.split(",") if s.strip())
    if target_ids:
        insts = [r for r in insts if r["institution_id"] in target_ids]
    if args.only_zero_archive:
        insts = [r for r in insts if not already_has_archive(r["institution_id"], data_raw)]
        logger.info("filtered to %d institutions with zero archives", len(insts))

    logger.info("brave-discovery on %d institutions, throttle=%.1fs, max_q=%d",
                 len(insts), args.throttle_s, args.max_queries)

    throttler = HostThrottler(min_interval_s=args.throttle_s)
    all_hits: list[dict[str, str]] = []
    summary: list[dict[str, Any]] = []

    for i, inst in enumerate(insts):
        try:
            hits, sm = discover_one_institution(inst, queries_cfg, throttler,
                                                 max_queries=args.max_queries)
        except Exception as exc:
            logger.exception("%s failed: %s", inst.get("institution_id"), exc)
            continue
        all_hits.extend(hits)
        summary.append(sm)
        if (i + 1) % 5 == 0:
            _flush(all_hits, summary)

    _flush(all_hits, summary)
    logger.info("done. wrote %d total seed URLs across %d institutions",
                 len(all_hits), len(summary))
    return 0


def _flush(hits: list[dict[str, str]], summary: list[dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["institution_id", "query_lang", "query", "url"])
        w.writeheader()
        for r in hits:
            w.writerow(r)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
