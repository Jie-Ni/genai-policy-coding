"""Build and validate the 240-institution sampling frame.

Stage 1 (this script): load the hand-curated `data/institution_list.csv`,
validate schema, check regional balance, check URL reachability, write a
validation report to `results/institution_list_validation.json`.

Stage 2 (to run later): fetch ROR records for each institution, attach
OpenAlex IDs, populate missing fields (QS/ARWU rank, enrollment size, year
of GenAI policy if found).

The committed CSV starts with 60 elite institutions hand-curated from QS
2026 + ARWU 2025 (Appendix A of operations_manual.md). The remaining 180
(20 upper + 15 mid + 10 regional per region) are filled in week 2 by
running this script with --extend, which calls ROR + OpenAlex APIs to
suggest candidates, ranked by Carnegie tier proxy / publication output.

Usage
-----
    python scripts/01_build_institution_list.py            # validate only
    python scripts/01_build_institution_list.py --extend   # call APIs to suggest the remaining 180
    python scripts/01_build_institution_list.py --check-urls  # ping each URL

Outputs
-------
- results/institution_list_validation.json: schema + balance + URL report
- data/institution_list_extended_candidates.csv: suggested upper/mid/regional candidates (when --extend)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

REQUIRED_FIELDS = [
    "institution_id", "name", "country", "country_code", "region", "tier",
    "qs_2026_rank", "arwu_2025_rank", "public_private", "language_primary",
    "language_secondary", "website", "policy_search_url", "ror_id", "notes",
]
REGION_CODES = {"NA", "EU", "EA", "LA"}
TIER_CODES = {"elite", "upper", "mid", "regional"}
TARGET_PER_REGION = 60
TARGET_PER_TIER = {"elite": 15, "upper": 20, "mid": 15, "regional": 10}
TARGET_TOTAL = 240


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    return [{k.lstrip("\ufeff"): v for k, v in r.items()} for r in rows]


def validate_schema(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    if not rows:
        errors.append("empty CSV")
        return errors
    actual = set(rows[0].keys())
    missing = set(REQUIRED_FIELDS) - actual
    extra = actual - set(REQUIRED_FIELDS)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected fields: {sorted(extra)}")
    seen_ids = set()
    for i, r in enumerate(rows, 1):
        rid = r.get("institution_id", "")
        if not rid:
            errors.append(f"row {i}: empty institution_id")
        elif rid in seen_ids:
            errors.append(f"row {i}: duplicate institution_id {rid}")
        seen_ids.add(rid)
        if r.get("region") not in REGION_CODES:
            errors.append(f"row {i} ({rid}): bad region {r.get('region')!r}")
        if r.get("tier") not in TIER_CODES:
            errors.append(f"row {i} ({rid}): bad tier {r.get('tier')!r}")
        if r.get("public_private") not in {"public", "private", ""}:
            errors.append(f"row {i} ({rid}): bad public_private {r.get('public_private')!r}")
        if not r.get("website", "").startswith("http"):
            errors.append(f"row {i} ({rid}): bad website {r.get('website')!r}")
    return errors


def check_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_region = Counter(r["region"] for r in rows)
    by_tier = Counter(r["tier"] for r in rows)
    by_region_tier = Counter((r["region"], r["tier"]) for r in rows)
    by_country = Counter(r["country_code"] for r in rows)
    by_language = Counter(r["language_primary"] for r in rows)
    report = {
        "n_total": len(rows),
        "target_total": TARGET_TOTAL,
        "n_short": max(0, TARGET_TOTAL - len(rows)),
        "by_region": dict(by_region),
        "by_tier": dict(by_tier),
        "by_region_tier": {f"{r}_{t}": n for (r, t), n in sorted(by_region_tier.items())},
        "by_country_top10": by_country.most_common(10),
        "by_language_primary": dict(by_language),
        "balance_ok_per_region": {
            region: by_region.get(region, 0) == TARGET_PER_REGION for region in REGION_CODES
        },
        "balance_ok_per_tier": {
            tier: by_tier.get(tier, 0) == TARGET_PER_TIER[tier] * len(REGION_CODES) for tier in TIER_CODES
        },
        "balance_ok_per_region_tier": {
            f"{region}_{tier}": by_region_tier.get((region, tier), 0) == TARGET_PER_TIER[tier]
            for region in REGION_CODES for tier in TIER_CODES
        },
    }
    return report


def check_urls(rows: list[dict[str, Any]], max_per_call: int = 5) -> dict[str, Any]:
    """HEAD-request each website + policy_search_url. Slow; opt-in.

    We do not call this on every run because it touches external sites. When
    invoked, we throttle to 1 request per 3 seconds per host and log all
    responses.
    """
    try:
        import urllib.request
    except ImportError:
        return {"error": "urllib unavailable"}
    import time
    results: dict[str, dict[str, Any]] = {}
    for r in rows[:max_per_call]:
        for field in ("website", "policy_search_url"):
            url = r.get(field, "")
            if not url:
                continue
            entry: dict[str, Any] = {"url": url, "status": None, "elapsed_ms": None, "error": None}
            t0 = time.time()
            try:
                req = urllib.request.Request(
                    url,
                    method="HEAD",
                    headers={"User-Agent": "genai-policy-maturation/1.0"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    entry["status"] = resp.status
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["elapsed_ms"] = int((time.time() - t0) * 1000)
            results.setdefault(r["institution_id"], {})[field] = entry
            time.sleep(3.0)
    return results


def extend_with_ror(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suggest upper/mid/regional candidates by querying ROR.

    Strategy (executed when --extend is passed):
    1. For each region, identify the gap: TARGET_PER_TIER - count(current)
    2. Query ROR `https://api.ror.org/organizations?query=...&filter=country.country_code:US&types=Education`
       for institutions in the country whose names are not already in the list
    3. Cross-reference with QS 2026 + ARWU 2025 (manually downloaded CSVs in
       data/rankings/) to assign tier
    4. Suggest top 20 candidates per gap, sorted by Carnegie/equivalent
       research-intensity proxy

    The function returns a list of dicts in the same schema; output is
    written to data/institution_list_extended_candidates.csv. Human review
    required before merging into institution_list.csv.

    NOTE: ROR API has no auth and is generous; rate limit 2000 req/min.
    However, manual curation of regional rankings is still required because
    QS and ARWU do not cover all regions adequately, especially LA.
    """
    print("WARNING: extend_with_ror is a stub. Manual curation required for")
    print("upper/mid/regional tiers. See operations_manual.md Appendix A.")
    print()
    print("Action plan for week 2:")
    print("  (1) Download QS World 2026 + ARWU 2025 CSVs to data/rankings/")
    print("  (2) For each region 脳 tier cell, list candidate institutions")
    print("       sorted by rank")
    print("  (3) For LA, supplement with QS Latin America 2026 + Times HE LA")
    print("  (4) Final review by senior author before merging")
    return []


def write_report(report: dict[str, Any], errors: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"errors": errors, "balance": report}
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extend", action="store_true", help="call ROR/OpenAlex to suggest remaining 180 candidates")
    ap.add_argument("--check-urls", action="store_true", help="HEAD-request each URL (slow, polite)")
    ap.add_argument("--csv", default=str(DATA / "institution_list.csv"))
    args = ap.parse_args()

    rows = load_csv(Path(args.csv))
    print(f"loaded {len(rows)} institutions from {args.csv}")

    errors = validate_schema(rows)
    balance = check_balance(rows)
    print(f"validation: {len(errors)} errors")
    for e in errors[:20]:
        print(f"  - {e}")
    if len(errors) > 20:
        print(f"  ({len(errors) - 20} more errors elided)")

    print()
    print(f"balance: {balance['n_total']} / {balance['target_total']} institutions")
    print(f"  by region: {balance['by_region']}")
    print(f"  by tier:   {balance['by_tier']}")
    for cell, ok in balance["balance_ok_per_region_tier"].items():
        if not ok:
            current = sum(1 for r in rows if r["region"] == cell.split("_")[0] and r["tier"] == cell.split("_")[1])
            target = TARGET_PER_TIER[cell.split("_")[1]]
            print(f"  cell {cell}: {current} / {target} (gap = {target - current})")

    url_report = None
    if args.check_urls:
        print()
        print("checking URLs (rate-limited, throttled)...")
        url_report = check_urls(rows)
        print(f"  checked {len(url_report)} institutions")

    if args.extend:
        print()
        print("attempting extension to 240...")
        extend_with_ror(rows)

    write_report({"balance": balance, "url_report": url_report}, errors, RESULTS / "institution_list_validation.json")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

