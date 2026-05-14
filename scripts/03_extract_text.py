"""Extract clean text from the raw archived HTML and PDF documents.

Walks data/raw/<region>/<institution_id>/*.{html,pdf}, extracts text, runs
a battery of cleaning + encoding-validity checks, writes one row per
extracted document to data_processed/policy_documents.jsonl.

Key concerns this script addresses
----------------------------------
1. **HTML noise**: navigation, footers, cookie banners, scripts, styles.
   Solution: trafilatura (preferred) or readability-lxml (fallback) or
   a minimal regex-strip if neither library is installed.

2. **PDF mojibake** (the v15.x problem with Chinese 培养方案 PDFs that
   had broken CMap-to-Unicode mappings): detect by looking for the U+FFFD
   replacement character or by a high rate of non-printable bytes; flag
   for OCR fallback.

3. **Language detection**: each extracted document gets a language tag
   from langdetect (or a multilingual heuristic if langdetect is unavailable).
   This catches cases where an institution's English-language sub-page
   is in fact untranslated source-language content.

4. **Encoding detection**: HTML charset from <meta> tag; fallback to
   utf-8 with errors=replace.

Usage
-----
    python scripts/03_extract_text.py                    # process all archives
    python scripts/03_extract_text.py --institutions NA_E01,NA_E02
    python scripts/03_extract_text.py --force            # re-process even if existing output
    python scripts/03_extract_text.py --ocr-fallback     # try OCR on PDFs with mojibake

Outputs
-------
data_processed/policy_documents.jsonl   # one line per extracted document
data_processed/extraction_report.json   # summary statistics
results/03_extract.log
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    DATA, DATA_PROCESSED, DATA_RAW, RESULTS, ROOT,
    ensure_dir, load_csv_dicts, setup_logging, sha256_bytes, write_json,
)

logger = setup_logging("03_extract")

INSTITUTION_LIST_PATH = DATA / "institution_list.csv"
EXTRACT_OUT = DATA_PROCESSED / "policy_documents.jsonl"
EXTRACT_REPORT = DATA_PROCESSED / "extraction_report.json"

# ---------------------------------------------------------------------------
# Optional dependencies (best-effort imports)
# ---------------------------------------------------------------------------

try:
    import trafilatura  # type: ignore
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from readability import Document as ReadabilityDoc  # type: ignore
    HAS_READABILITY = True
except ImportError:
    HAS_READABILITY = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 11
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


def detect_language(text: str, fallback: str = "und") -> str:
    """Best-effort language detection. Falls back to script-based heuristic."""
    sample = text[:2000].strip()
    if not sample:
        return fallback
    if HAS_LANGDETECT:
        try:
            return detect(sample)
        except Exception:
            pass
    # Script heuristic
    cjk = sum(1 for c in sample if '一' <= c <= '鿿')
    if cjk > 30:
        return "zh"
    hira = sum(1 for c in sample if '぀' <= c <= 'ゟ')
    kata = sum(1 for c in sample if '゠' <= c <= 'ヿ')
    if hira + kata > 30:
        return "ja"
    hangul = sum(1 for c in sample if '가' <= c <= '힣')
    if hangul > 30:
        return "ko"
    # Latin-script
    if any(c in sample for c in "äöüÄÖÜß"):
        return "de"
    if any(c in sample for c in "ñáéíóúü¿¡"):
        return "es"
    if any(c in sample for c in "ãõçáéíóúâêô"):
        return "pt"
    if any(c in sample for c in "àâçéèêëîïôûùü"):
        return "fr"
    return fallback


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------


_HTML_CHARSET_RE = re.compile(
    rb'<meta[^>]+charset[^=]*=\s*["\']?([\w-]+)', re.I
)


def detect_html_charset(html_bytes: bytes) -> str:
    m = _HTML_CHARSET_RE.search(html_bytes[:4096])
    if m:
        try:
            return m.group(1).decode("ascii", errors="ignore").lower()
        except Exception:
            pass
    return "utf-8"


def extract_html(html_bytes: bytes, source_url: str = "") -> dict[str, Any]:
    charset = detect_html_charset(html_bytes)
    try:
        html_text = html_bytes.decode(charset, errors="replace")
    except LookupError:
        html_text = html_bytes.decode("utf-8", errors="replace")

    main_text = ""
    title = ""
    extraction_method = "regex"

    if HAS_TRAFILATURA:
        try:
            main_text = trafilatura.extract(
                html_text, url=source_url, include_comments=False,
                include_tables=True, deduplicate=True,
            ) or ""
            extraction_method = "trafilatura"
        except Exception:
            main_text = ""

    if not main_text and HAS_READABILITY:
        try:
            doc = ReadabilityDoc(html_text)
            title = doc.title() or ""
            content_html = doc.summary(html_partial=True)
            main_text = re.sub(r"<[^>]+>", " ", content_html)
            extraction_method = "readability"
        except Exception:
            pass

    if not main_text:
        # last-ditch regex strip
        t = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.I)
        t = re.sub(r"<style[\s\S]*?</style>", " ", t, flags=re.I)
        t = re.sub(r"<nav[\s\S]*?</nav>", " ", t, flags=re.I)
        t = re.sub(r"<footer[\s\S]*?</footer>", " ", t, flags=re.I)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"\s+", " ", t)
        main_text = t.strip()
        if not title:
            tm = re.search(r"<title[^>]*>([\s\S]*?)</title>", html_text, flags=re.I)
            title = tm.group(1).strip() if tm else ""

    main_text = re.sub(r"\s+", " ", main_text).strip()
    return {
        "title": title.strip(),
        "text": main_text,
        "extraction_method": extraction_method,
        "charset": charset,
    }


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def mojibake_score(text: str) -> float:
    """Fraction of characters that are U+FFFD or non-printable.

    > 0.05 = likely mojibake.
    """
    if not text:
        return 1.0
    bad = sum(1 for c in text if c == "�" or unicodedata.category(c) in ("Cc", "Co", "Cn"))
    return bad / max(1, len(text))


def extract_pdf(pdf_bytes: bytes) -> dict[str, Any]:
    if not HAS_FITZ:
        return {
            "text": "",
            "title": "",
            "extraction_method": "none",
            "n_pages": 0,
            "warning": "PyMuPDF (fitz) not installed; pip install pymupdf",
        }
    text_parts: list[str] = []
    title = ""
    n_pages = 0
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        title = doc.metadata.get("title", "") or ""
        n_pages = len(doc)
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
    except Exception as exc:
        return {"text": "", "title": "", "extraction_method": "fitz_failed",
                "n_pages": 0, "warning": f"fitz error: {exc}"}
    full = "\n\n".join(text_parts)
    moji = mojibake_score(full)
    return {
        "title": title,
        "text": full,
        "extraction_method": "fitz",
        "n_pages": n_pages,
        "mojibake_score": moji,
        "mojibake_flagged": moji > 0.05,
    }


# ---------------------------------------------------------------------------
# Per-archive processing
# ---------------------------------------------------------------------------


def walk_archives(institutions: list[dict[str, Any]], target_ids: set[str] | None
                  ) -> Iterator[tuple[dict[str, Any], Path]]:
    by_id = {r["institution_id"]: r for r in institutions}
    for region_dir in sorted(DATA_RAW.iterdir()):
        if not region_dir.is_dir() or region_dir.name.startswith("_"):
            continue
        for inst_dir in sorted(region_dir.iterdir()):
            if not inst_dir.is_dir():
                continue
            iid = inst_dir.name
            if target_ids and iid not in target_ids:
                continue
            inst = by_id.get(iid, {"institution_id": iid, "region": region_dir.name})
            for f in sorted(inst_dir.glob("*.html")):
                yield inst, f
            for f in sorted(inst_dir.glob("*.pdf")):
                yield inst, f


def load_sidecar(archive_path: Path) -> dict[str, Any]:
    sc = archive_path.with_suffix(".json")
    if sc.exists():
        try:
            return json.loads(sc.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def process_archive(institution: dict[str, Any], archive_path: Path,
                     ocr_fallback: bool = False) -> dict[str, Any]:
    sidecar = load_sidecar(archive_path)
    ext = archive_path.suffix.lstrip(".").lower()
    raw = archive_path.read_bytes()

    if ext == "pdf":
        extraction = extract_pdf(raw)
        if extraction.get("mojibake_flagged") and ocr_fallback:
            extraction["text"] = ""
            extraction["warning"] = "mojibake detected; OCR fallback not implemented (paddleocr/tesseract); flagged for manual review"
            extraction["extraction_method"] = "fitz_then_ocr_skipped"
    elif ext == "html":
        extraction = extract_html(raw, source_url=sidecar.get("url", ""))
    else:
        return {"institution_id": institution["institution_id"], "file": str(archive_path),
                "error": f"unknown extension {ext}"}

    text = extraction.get("text", "")
    lang_detected = detect_language(text) if text else "und"
    lang_declared = institution.get("language_primary", "")
    return {
        "institution_id": institution["institution_id"],
        "region": institution.get("region", ""),
        "country_code": institution.get("country_code", ""),
        "tier": institution.get("tier", ""),
        "language_declared": lang_declared,
        "language_detected": lang_detected,
        "language_match": lang_detected == lang_declared,
        "file": str(archive_path.relative_to(ROOT)),
        "extension": ext,
        "sha256": sha256_bytes(raw),
        "size_bytes": len(raw),
        "url": sidecar.get("url", ""),
        "accessed_at": sidecar.get("accessed_at", ""),
        "title": extraction.get("title", ""),
        "text": text,
        "text_length_chars": len(text),
        "text_length_words": len(text.split()),
        "extraction_method": extraction.get("extraction_method", ""),
        "mojibake_score": extraction.get("mojibake_score"),
        "mojibake_flagged": extraction.get("mojibake_flagged", False),
        "n_pages": extraction.get("n_pages"),
        "warning": extraction.get("warning"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--institutions", default="", help="comma-separated institution_ids to process")
    ap.add_argument("--force", action="store_true", help="re-process even if already in output")
    ap.add_argument("--ocr-fallback", action="store_true", help="flag mojibake PDFs for OCR (not implemented yet)")
    args = ap.parse_args()

    institutions = load_csv_dicts(INSTITUTION_LIST_PATH)
    target_ids = set(args.institutions.split(",")) if args.institutions else None

    # Load existing output to skip processed (unless --force)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    if EXTRACT_OUT.exists() and not args.force:
        for line in EXTRACT_OUT.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                seen.add(r.get("sha256", ""))
                rows.append(r)
            except Exception:
                continue
        logger.info("loaded %d existing extractions; skipping duplicates", len(rows))

    new_count = 0
    flagged_mojibake = 0
    flagged_lang_mismatch = 0
    for inst, archive in walk_archives(institutions, target_ids):
        # quick skip if sha already in output
        raw_sha = sha256_bytes(archive.read_bytes())
        if raw_sha in seen and not args.force:
            continue
        rec = process_archive(inst, archive, ocr_fallback=args.ocr_fallback)
        rows.append(rec)
        seen.add(rec["sha256"])
        new_count += 1
        if rec.get("mojibake_flagged"):
            flagged_mojibake += 1
            logger.warning("%s mojibake flag: %s", inst["institution_id"], archive.name)
        if not rec.get("language_match") and rec.get("text_length_chars", 0) > 200:
            flagged_lang_mismatch += 1
            logger.info("%s lang mismatch (declared %s vs detected %s): %s",
                         inst["institution_id"], rec.get("language_declared"),
                         rec.get("language_detected"), archive.name)

    # write JSONL output
    ensure_dir(DATA_PROCESSED)
    with EXTRACT_OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("wrote %d rows to %s (added %d new)", len(rows), EXTRACT_OUT, new_count)

    # summary report
    report = {
        "n_archives_total": len(rows),
        "n_archives_new_this_run": new_count,
        "n_mojibake_flagged": flagged_mojibake,
        "n_lang_mismatch": flagged_lang_mismatch,
        "n_html": sum(1 for r in rows if r.get("extension") == "html"),
        "n_pdf": sum(1 for r in rows if r.get("extension") == "pdf"),
        "n_per_region": {},
        "n_per_language_declared": {},
        "n_per_language_detected": {},
        "avg_text_length_chars": int(sum(r.get("text_length_chars", 0) for r in rows) / max(1, len(rows))),
        "extraction_dependencies": {
            "trafilatura": HAS_TRAFILATURA,
            "readability": HAS_READABILITY,
            "fitz_pymupdf": HAS_FITZ,
            "langdetect": HAS_LANGDETECT,
        },
    }
    for r in rows:
        for k in ("region", "language_declared", "language_detected"):
            d = report["n_per_" + ("region" if k == "region" else k)]
            v = r.get(k, "")
            d[v] = d.get(v, 0) + 1
    write_json(EXTRACT_REPORT, report)
    logger.info("extraction report: %s", json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
