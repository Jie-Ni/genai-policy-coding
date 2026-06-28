"""Polite multilingual scraper for university GenAI policy documents.

For each institution in data/institution_list.csv:
  1. read its primary language + secondary languages from the CSV
  2. construct site-search queries from data/search_queries.json
  3. for each query, perform a Google site-search OR direct probe of common
     URL patterns ({domain}/genai-policy, /ai-policy, /academic-integrity/ai)
  4. for each candidate URL: fetch, run robots.txt check, apply keyword-
     density filter, archive if it passes
  5. write metadata sidecar with sha256 + accessed_at
  6. log every action to results/02_scrape.log

Polite scraping conventions:
  - respect robots.txt per RFC 9309 (we use urllib.robotparser)
  - throttle 1 request per 3 s per host (HostThrottler in common.py)
  - exponential backoff on rate limit (HTTP 429) or 5xx
  - cache responses in data/raw/_cache/<sha256-of-url>.{html,pdf,json}
  - User-Agent identifies the project + contact email

Usage
-----
    python scripts/02_scrape_policies.py                # full run
    python scripts/02_scrape_policies.py --dry-run      # planning report, no fetches
    python scripts/02_scrape_policies.py --limit 5      # first 5 institutions only (smoke test)
    python scripts/02_scrape_policies.py --resume       # skip institutions with existing archives
    python scripts/02_scrape_policies.py --institutions NA_E01,NA_E02   # specific IDs

Notes on Google site-search
---------------------------
We do NOT crawl Google directly (against ToS) and we do NOT use the paid
Custom Search JSON API by default. Instead we use the *DuckDuckGo HTML*
or *Bing Web Search API* (free tier, 1000/mo) when available; the fallback
is direct URL probing.

To use Bing Web Search, set the env var BING_SEARCH_API_KEY.
Without it, the script falls back to direct probing of common URL
patterns + the institution homepage parse. Pure-fallback mode discovers
roughly 50-60 % of policy pages; with Bing it rises to ~ 85-95 %.

Output files
------------
data/raw/<region>/<institution_id>/<accessed_at_iso>.html      # archived page
data/raw/<region>/<institution_id>/<accessed_at_iso>.pdf       # if PDF
data/raw/<region>/<institution_id>/<accessed_at_iso>.json      # metadata sidecar
data/raw/<region>/<institution_id>/index.json                  # all archives for this institution
data/exclusion_ledger.csv                                       # one row per inst with outcome
results/02_scrape_summary.json                                  # final per-region tally
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    DATA, DATA_RAW, RESULTS, ROOT, USER_AGENT,
    HostThrottler, append_csv, ensure_dir, load_csv_dicts, load_json,
    setup_logging, sha256_bytes, write_json,
)

logger = setup_logging("02_scrape")

EXCLUSION_LEDGER = DATA / "exclusion_ledger.csv"
SCRAPE_SUMMARY = RESULTS / "02_scrape_summary.json"
SEARCH_QUERIES_PATH = DATA / "search_queries.json"
INSTITUTION_LIST_PATH = DATA / "institution_list.csv"
BRAVE_SEED_URLS_PATH = DATA / "brave_seed_urls.csv"
CACHE_DIR = DATA_RAW / "_cache"


_brave_seeds_cache: dict[str, list[str]] | None = None


def load_brave_seed_urls() -> dict[str, list[str]]:
    """Read data/brave_seed_urls.csv (output of 02b) and group by institution_id.
    Cached after first call. Returns {} if file is missing."""
    global _brave_seeds_cache
    if _brave_seeds_cache is not None:
        return _brave_seeds_cache
    seeds: dict[str, list[str]] = {}
    if not BRAVE_SEED_URLS_PATH.exists():
        _brave_seeds_cache = {}
        return _brave_seeds_cache
    import csv as _csv
    with BRAVE_SEED_URLS_PATH.open("r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            iid = row.get("institution_id", "").strip()
            url = row.get("url", "").strip()
            if iid and url:
                seeds.setdefault(iid, []).append(url)
    logger.info("loaded %d brave seed URLs across %d institutions",
                 sum(len(v) for v in seeds.values()), len(seeds))
    _brave_seeds_cache = seeds
    return _brave_seeds_cache

COMMON_URL_PATTERNS = [
    "/genai-policy", "/generative-ai-policy", "/ai-policy",
    "/academic-integrity/ai", "/ai-guidelines", "/ai-guidance",
    "/policies/ai", "/teaching-and-learning/ai",
    "/academic-integrity/generative-ai", "/chatgpt",
    "/ai-in-teaching", "/genai-guidance",
]


# ---------------------------------------------------------------------------
# Robots.txt + HTTP helpers
# ---------------------------------------------------------------------------

_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def can_fetch(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = f"{parsed.scheme}://{parsed.netloc}"
    if host not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{host}/robots.txt")
        try:
            rp.read()
            _robots_cache[host] = rp
        except Exception as exc:
            logger.warning("robots fetch failed for %s: %s 鈥?assuming allowed", host, exc)
            _robots_cache[host] = None
    rp = _robots_cache[host]
    if rp is None:
        return True
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def fetch_url(url: str, throttler: HostThrottler, timeout: float = 20.0) -> tuple[int, bytes, dict[str, str]]:
    """Polite GET. Returns (status, body, headers). Raises on network error."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc
    throttler.wait(host)
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.5",
        "Accept-Language": "en;q=1.0,de;q=0.8,zh;q=0.8,es;q=0.7,pt;q=0.7,fr;q=0.7,ja;q=0.6,ko;q=0.6",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            status = resp.status
            headers = {k.lower(): v for k, v in resp.headers.items()}
            throttler.reset_backoff(host)
            return status, body, headers
    except urllib.error.HTTPError as exc:
        if exc.code in (429, 500, 502, 503, 504):
            throttler.back_off(host)
        raise


def cache_path_for(url: str, ext_hint: str = "html") -> Path:
    h = sha256_bytes(url.encode("utf-8"))[:32]
    ensure_dir(CACHE_DIR)
    return CACHE_DIR / f"{h}.{ext_hint}"


# ---------------------------------------------------------------------------
# Candidate URL discovery
# ---------------------------------------------------------------------------


def discover_candidate_urls_pattern_probe(institution: dict[str, Any]) -> list[str]:
    base = institution["website"].rstrip("/")
    return [base + p for p in COMMON_URL_PATTERNS]


def discover_candidate_urls_bing(institution: dict[str, Any], queries: list[str],
                                  throttler: HostThrottler) -> list[str]:
    key = os.environ.get("BING_SEARCH_API_KEY", "").strip()
    if not key:
        return []
    domain = urllib.parse.urlparse(institution["website"]).netloc
    candidates: list[str] = []
    api = "https://api.bing.microsoft.com/v7.0/search"
    for q in queries:
        site_q = f'site:{domain} "{q}"'
        params = urllib.parse.urlencode({"q": site_q, "count": 20, "responseFilter": "Webpages"})
        url = f"{api}?{params}"
        req = urllib.request.Request(url, headers={
            "Ocp-Apim-Subscription-Key": key,
            "User-Agent": USER_AGENT,
        })
        throttler.wait("api.bing.microsoft.com")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            logger.warning("bing search failed for %s q=%r: %s", domain, q, exc)
            continue
        for item in data.get("webPages", {}).get("value", []):
            u = item.get("url")
            if u and u not in candidates:
                candidates.append(u)
    return candidates


def discover_candidate_urls_duckduckgo(institution: dict[str, Any], queries: list[str],
                                        throttler: HostThrottler) -> list[str]:
    """DuckDuckGo HTML endpoint as a free fallback. ToS allows light use; we
    throttle hard. NOT a replacement for Bing API; coverage is shallower."""
    domain = urllib.parse.urlparse(institution["website"]).netloc
    candidates: list[str] = []
    base = "https://html.duckduckgo.com/html/"
    for q in queries[:5]:  # cap to 5 queries to stay polite
        params = urllib.parse.urlencode({"q": f'site:{domain} "{q}"'})
        url = f"{base}?{params}"
        try:
            status, body, _ = fetch_url(url, throttler)
        except Exception as exc:
            logger.warning("duckduckgo failed for %s q=%r: %s", domain, q, exc)
            continue
        if status != 200:
            continue
        text = body.decode("utf-8", errors="ignore")
        for m in re.finditer(r'class="result__a"[^>]+href="([^"]+)"', text):
            u = urllib.parse.unquote(m.group(1))
            # ddg uses redirect URLs; extract the underlying uddg=
            m2 = re.search(r"uddg=([^&]+)", u)
            if m2:
                u = urllib.parse.unquote(m2.group(1))
            if domain in u and u not in candidates:
                candidates.append(u)
    return candidates


def discover_candidate_urls_homepage_crawl(institution: dict[str, Any],
                                            queries_cfg: dict[str, Any],
                                            throttler: HostThrottler,
                                            max_hops: int = 2,
                                            max_candidates: int = 20) -> list[str]:
    """Start from the institution homepage and follow <a> links whose anchor
    text or href matches language-specific policy keywords. Bypasses search
    engines entirely 鈥?works for sites with localized URL conventions
    (Chinese, Portuguese, German subpaths) that English-only patterns miss.
    """
    home = institution["website"].rstrip("/")
    domain = urllib.parse.urlparse(home).netloc
    primary_lang = institution.get("language_primary", "en")
    secondary = (institution.get("language_secondary") or "").split("+")
    langs = [primary_lang] + [s.strip() for s in secondary if s.strip()] + ["en"]
    langs = list(dict.fromkeys(langs))

    # Build keyword pool from per-language density terms + query tokens.
    keywords: list[str] = []
    for lang in langs:
        lc = queries_cfg["languages"].get(lang, {})
        keywords += [t.lower() for t in lc.get("keyword_density_terms", [])]
        for q in lc.get("queries", []):
            # split on whitespace / punctuation for token-level matching
            for tok in re.split(r"[\s,锛屻€?锛沒+", q):
                tok = tok.strip().lower()
                if len(tok) >= 2:
                    keywords.append(tok)
    keywords = list(dict.fromkeys(keywords))

    seen: set[str] = {home}
    scored: dict[str, int] = {}
    frontier: list[tuple[str, int]] = [(home, 0)]
    while frontier and len(scored) < max_candidates * 3:
        url, hop = frontier.pop(0)
        if hop > max_hops:
            continue
        if not can_fetch(url):
            continue
        try:
            status, body, headers = fetch_url(url, throttler)
        except Exception:
            continue
        if status != 200 or "html" not in headers.get("content-type", "").lower():
            continue
        try:
            html = body.decode("utf-8", errors="ignore")
        except Exception:
            continue
        for m in re.finditer(
            r'<a\s+[^>]*href="([^"#?]+)"[^>]*>([^<]{1,200})</a>',
            html,
            flags=re.I,
        ):
            href, text = m.group(1), m.group(2)
            abs_url = urllib.parse.urljoin(url + "/", href)
            parsed = urllib.parse.urlparse(abs_url)
            if domain not in parsed.netloc:
                continue
            if abs_url in seen or len(abs_url) > 300:
                continue
            target = (parsed.path + " " + text).lower()
            score = sum(1 for kw in keywords if kw in target)
            if score == 0:
                continue
            seen.add(abs_url)
            scored[abs_url] = score
            # Enqueue link for deeper traversal only if it's NOT already a
            # likely policy page (those get fetched in the main loop instead)
            if hop < max_hops and score < 3:
                frontier.append((abs_url, hop + 1))
    ranked = sorted(scored.items(), key=lambda kv: -kv[1])
    return [u for u, _ in ranked[:max_candidates]]


# ---------------------------------------------------------------------------
# Keyword density filter
# ---------------------------------------------------------------------------


def text_from_html(html: bytes, charset_hint: str = "utf-8") -> str:
    try:
        text = html.decode(charset_hint, errors="ignore")
    except LookupError:
        text = html.decode("utf-8", errors="ignore")
    # strip <script> and <style>
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Terms that explicitly signal *generative* AI 鈥?distinguishing GenAI policy
# pages from generic AI/academic-integrity content. A page must contain at
# least one of these (in any of the page's languages) to be archived.
CORE_GENAI_TERMS = [
    # English
    "chatgpt", "generative ai", "genai", "large language model", "llm",
    "gpt-3", "gpt-4", "gpt-5", "claude", "gemini", "copilot",
    # Chinese
    "鐢熸垚寮忎汉宸ユ櫤鑳?, "鐢熸垚寮廰i", "澶ц瑷€妯″瀷", "閫氱敤浜哄伐鏅鸿兘",
    # German
    "generative ki", "generative k眉nstliche", "sprachmodell",
    # Spanish / Portuguese
    "ia generativa", "inteligencia artificial generativa",
    "intelig锚ncia artificial generativa",
    # French
    "ia g茅n茅rative", "intelligence artificielle g茅n茅rative",
    # Japanese / Korean
    "鐢熸垚ai", "靸濎劚順?ai", "靸濎劚 ai",
    # Italian / Dutch
    "ia generativa", "generatieve ai",
]


def keyword_density_pass(text: str, lang: str, queries_cfg: dict[str, Any]) -> bool:
    lang_cfg = queries_cfg["languages"].get(lang, {})
    terms = lang_cfg.get("keyword_density_terms", [])
    if not terms:
        return False
    # Normalize: for non-ASCII text, keep original case for CJK substring match;
    # also build a lowercase version for Latin-script terms.
    text_lc = text.lower()
    hits = sum(1 for t in terms if t.lower() in text_lc)
    if hits < queries_cfg["keyword_density_filter"]["min_term_count"]:
        return False
    # Require at least one explicit *generative* AI term to filter out generic
    # academic-integrity / AI pages (e.g., Cambridge's animal-research policy).
    return any(c in text_lc for c in CORE_GENAI_TERMS)


def url_keyword_pass(url: str, queries_cfg: dict[str, Any]) -> bool:
    """If keyword density misses, the URL pattern itself can save it."""
    cfg = queries_cfg["keyword_density_filter"]
    u = url.lower()
    for key in ["fallback_url_keywords_en", "fallback_url_keywords_de",
                 "fallback_url_keywords_zh", "fallback_url_keywords_es",
                 "fallback_url_keywords_pt", "fallback_url_keywords_fr"]:
        for kw in cfg.get(key, []):
            if kw in u:
                return True
    return False


# ---------------------------------------------------------------------------
# Per-institution pipeline
# ---------------------------------------------------------------------------


def archive_response(institution: dict[str, Any], url: str, body: bytes,
                      content_type: str) -> dict[str, Any]:
    region = institution["region"]
    iid = institution["institution_id"]
    out_dir = ensure_dir(DATA_RAW / region / iid)
    accessed_at = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ext = "pdf" if "application/pdf" in content_type else "html"
    out_path = out_dir / f"{accessed_at}.{ext}"
    out_path.write_bytes(body)
    sha = sha256_bytes(body)
    sidecar = {
        "institution_id": iid,
        "region": region,
        "url": url,
        "accessed_at": accessed_at,
        "content_type": content_type,
        "extension": ext,
        "size_bytes": len(body),
        "sha256": sha,
        "language_primary": institution.get("language_primary", ""),
        "language_secondary": institution.get("language_secondary", ""),
        "country_code": institution.get("country_code", ""),
        "tier": institution.get("tier", ""),
    }
    side_path = out_dir / f"{accessed_at}.json"
    write_json(side_path, sidecar)
    # update per-institution index
    idx_path = out_dir / "index.json"
    idx: list[dict[str, Any]] = []
    if idx_path.exists():
        try:
            idx = load_json(idx_path)
        except Exception:
            idx = []
    idx.append({"file": out_path.name, "url": url, "sha256": sha, "accessed_at": accessed_at})
    write_json(idx_path, idx)
    return sidecar


def scrape_institution(institution: dict[str, Any], queries_cfg: dict[str, Any],
                        throttler: HostThrottler, dry_run: bool = False) -> dict[str, Any]:
    iid = institution["institution_id"]
    primary_lang = institution.get("language_primary", "en")
    secondary = (institution.get("language_secondary") or "").split("+") if institution.get("language_secondary") else []
    languages = [primary_lang] + [s.strip() for s in secondary if s.strip()] + ["en"]
    languages = list(dict.fromkeys(languages))  # preserve order, dedupe

    queries: list[str] = []
    for lang in languages:
        cfg = queries_cfg["languages"].get(lang)
        if cfg:
            queries.extend(cfg["queries"])
    queries = list(dict.fromkeys(queries))

    # 0. Brave-seeded URLs first (highest signal 鈥?they're already search-engine
    # filtered for relevance to GenAI policy queries).
    brave_seeds = load_brave_seed_urls().get(iid, [])
    if brave_seeds:
        logger.info("%s: using %d Brave seed URLs", iid, len(brave_seeds))

    # 1. Direct pattern probe
    candidates = list(brave_seeds) + discover_candidate_urls_pattern_probe(institution)

    # If Brave seeded enough candidates, skip the slow DDG + homepage-crawl
    # paths entirely. Brave's site-search is already higher-precision than
    # both, and the homepage crawl alone can spend 5+ min on one institution.
    if len(brave_seeds) >= 5:
        logger.info("%s: brave seeds sufficient (%d); skipping DDG + crawl",
                     iid, len(brave_seeds))
    else:
        # 2. Bing API (if key set)
        if os.environ.get("BING_SEARCH_API_KEY"):
            try:
                candidates += discover_candidate_urls_bing(institution, queries, throttler)
            except Exception as exc:
                logger.warning("%s bing discovery failed: %s", iid, exc)
        else:
            # 3a. DuckDuckGo HTML fallback (often 403-blocked by their anti-bot)
            try:
                candidates += discover_candidate_urls_duckduckgo(institution, queries, throttler)
            except Exception as exc:
                logger.warning("%s duckduckgo discovery failed: %s", iid, exc)

        # 3b. Homepage link-crawl 鈥?works without API keys and respects robots.txt.
        # Essential for non-English sites where /ai-policy patterns don't apply.
        try:
            crawl_cands = discover_candidate_urls_homepage_crawl(
                institution, queries_cfg, throttler
            )
            logger.info("%s: homepage crawl found %d link candidates", iid, len(crawl_cands))
            candidates += crawl_cands
        except Exception as exc:
            logger.warning("%s homepage crawl failed: %s", iid, exc)

    # 4. Always include the explicit policy_search_url if provided
    if institution.get("policy_search_url"):
        candidates.append(institution["policy_search_url"])

    candidates = list(dict.fromkeys(candidates))  # dedupe
    logger.info("%s: %d candidate URLs", iid, len(candidates))

    outcome = {
        "institution_id": iid,
        "n_candidates": len(candidates),
        "n_robots_allowed": 0,
        "n_fetched_ok": 0,
        "n_passed_keyword_filter": 0,
        "n_archived": 0,
        "candidate_urls": candidates,
        "errors": [],
    }

    if dry_run:
        return outcome

    archived_sha = set()
    for url in candidates:
        if not can_fetch(url):
            outcome["errors"].append({"url": url, "stage": "robots", "msg": "blocked by robots.txt"})
            continue
        outcome["n_robots_allowed"] += 1
        try:
            status, body, headers = fetch_url(url, throttler)
        except Exception as exc:
            outcome["errors"].append({"url": url, "stage": "fetch", "msg": str(exc)})
            continue
        if status != 200:
            outcome["errors"].append({"url": url, "stage": "fetch", "msg": f"status {status}"})
            continue
        outcome["n_fetched_ok"] += 1

        content_type = headers.get("content-type", "")
        is_pdf = "application/pdf" in content_type or url.lower().endswith(".pdf")
        is_html = "text/html" in content_type or not is_pdf

        if is_pdf:
            # Treat PDFs as candidate policy documents (assume they are policy
            # if URL pattern hints policy/AI/integrity; otherwise sample text
            # extraction would be needed but we defer to script 03)
            if url_keyword_pass(url, queries_cfg):
                pass_filter = True
            else:
                # accept PDFs only if their URL pattern looks policy-relevant
                pass_filter = any(t in url.lower() for t in ["policy", "guideline", "guidance", "integrity", "ai"])
        else:
            text = text_from_html(body)
            # HTML pages must pass content-density check (which now requires
            # a core *generative* AI term). url_keyword_pass alone is too
            # permissive 鈥?it would accept any /policy URL even if the page
            # is about a different topic (e.g., animal-research policy).
            pass_filter = keyword_density_pass(text, primary_lang, queries_cfg) or \
                          keyword_density_pass(text, "en", queries_cfg)

        if not pass_filter:
            continue
        outcome["n_passed_keyword_filter"] += 1

        sha = sha256_bytes(body)
        if sha in archived_sha:
            continue
        archived_sha.add(sha)
        try:
            archive_response(institution, url, body, content_type)
            outcome["n_archived"] += 1
            logger.info("  archived %s (%d bytes, %s)", url, len(body), content_type.split(";")[0])
        except Exception as exc:
            outcome["errors"].append({"url": url, "stage": "archive", "msg": str(exc)})

    return outcome


# ---------------------------------------------------------------------------
# Ledger + summary
# ---------------------------------------------------------------------------


def log_to_exclusion_ledger(institution: dict[str, Any], outcome: dict[str, Any]) -> None:
    row = {
        "institution_id": institution["institution_id"],
        "region": institution["region"],
        "country": institution.get("country", ""),
        "tier": institution.get("tier", ""),
        "language_primary": institution.get("language_primary", ""),
        "n_candidates": outcome["n_candidates"],
        "n_archived": outcome["n_archived"],
        "drop_reason": "no_policy_found" if outcome["n_archived"] == 0 else "ok",
        "scrape_timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "n_errors": len(outcome["errors"]),
    }
    append_csv(EXCLUSION_LEDGER, row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def institution_iter(rows: list[dict[str, Any]], args) -> Iterator[dict[str, Any]]:
    target_ids = set(args.institutions.split(",")) if args.institutions else None
    for i, r in enumerate(rows):
        if args.limit and i >= args.limit:
            break
        if target_ids and r["institution_id"] not in target_ids:
            continue
        if args.resume:
            d = DATA_RAW / r["region"] / r["institution_id"]
            if (d / "index.json").exists():
                logger.info("skip %s (resume: index.json exists)", r["institution_id"])
                continue
        yield r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="discover candidates but do not fetch / archive")
    ap.add_argument("--limit", type=int, default=0, help="limit number of institutions (smoke test)")
    ap.add_argument("--resume", action="store_true", help="skip institutions that already have an index.json")
    ap.add_argument("--institutions", default="", help="comma-separated institution_ids to process (e.g., NA_E01,NA_E02)")
    ap.add_argument("--min-interval", type=float, default=3.0, help="min seconds between requests per host")
    args = ap.parse_args()

    institutions = load_csv_dicts(INSTITUTION_LIST_PATH)
    queries_cfg = load_json(SEARCH_QUERIES_PATH)
    throttler = HostThrottler(min_interval_s=args.min_interval)

    logger.info("loaded %d institutions", len(institutions))
    logger.info("loaded %d languages in search query battery", len(queries_cfg["languages"]))
    logger.info("dry_run=%s limit=%s resume=%s institutions=%r",
                args.dry_run, args.limit, args.resume, args.institutions)

    summary: dict[str, Any] = {
        "started_at": dt.datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "per_institution": {},
        "totals": {"n_total": 0, "n_with_archive": 0, "n_no_policy": 0, "n_errors": 0},
    }

    for inst in institution_iter(institutions, args):
        try:
            outcome = scrape_institution(inst, queries_cfg, throttler, dry_run=args.dry_run)
        except Exception as exc:
            logger.exception("%s: hard fail: %s", inst["institution_id"], exc)
            outcome = {"institution_id": inst["institution_id"], "n_candidates": 0,
                       "n_robots_allowed": 0, "n_fetched_ok": 0,
                       "n_passed_keyword_filter": 0, "n_archived": 0,
                       "candidate_urls": [], "errors": [{"stage": "hard_fail", "msg": str(exc)}]}
        summary["per_institution"][inst["institution_id"]] = outcome
        summary["totals"]["n_total"] += 1
        if outcome["n_archived"] > 0:
            summary["totals"]["n_with_archive"] += 1
        else:
            summary["totals"]["n_no_policy"] += 1
        summary["totals"]["n_errors"] += len(outcome["errors"])
        if not args.dry_run:
            log_to_exclusion_ledger(inst, outcome)

    summary["finished_at"] = dt.datetime.utcnow().isoformat() + "Z"
    write_json(SCRAPE_SUMMARY, summary)
    logger.info("done. totals: %s", summary["totals"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

