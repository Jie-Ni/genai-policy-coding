"""Chunk extracted policy text into ~500-token windows with metadata.

Inputs
------
data_processed/policy_documents.jsonl
    Output of 03_extract_text.py 鈥?one row per extracted document with
    fields: institution_id, source_url, text, language_detected,
    archive_sha (or similar archive identifier), and optional section
    structure markers.

data/institution_list.csv
    For institution-level metadata (region, tier, country, language_primary).

Outputs
-------
data_processed/shards/policy_chunks_shard_NN.jsonl   (NN = 00..07)
    Each line is one chunk ready for the coder:
        {
          "chunk_id": "<institution_id>_<archive_id>_<chunk_seq>",
          "institution_id": "NA_E01",
          "institution_name": "Massachusetts Institute of Technology",
          "region": "NA", "tier": "elite",
          "language_declared": "en",
          "section_header": "Faculty Guidance",
          "source_url": "https://...",
          "text": "<chunk body, 300-600 tokens approx>",
          "n_words": 412,
          "sha256_chunk": "..."
        }

Chunking strategy
-----------------
1. Split the extracted text on Markdown-style headings (#, ##, ###) and on
   double-newline paragraph boundaries.
2. Greedily pack paragraphs into target_tokens-sized windows, preserving the
   most recent heading as `section_header`.
3. Filter out chunks that lack any GenAI-specific term (CORE_GENAI_TERMS).
4. Shard the resulting chunks deterministically by sha256(chunk_id) % N
   so a 4-region 脳 4-tier corpus is balanced across shards (not all of one
   region landing in shard 0).

Token approximation: words 脳 1.3 (English) / 0.7 (Chinese) is the rough
rule we use; vLLM's tokenizer is the actual gate but we don't need to
import it here. The 500-token target is the sweet spot for the coder
prompt budget (system + 16 few-shot examples + chunk + JSON output stays
under 8K context).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import load_csv_dicts  # type: ignore

POLICY_DOCS_JSONL = ROOT / "data_processed" / "policy_documents.jsonl"
SHARDS_DIR = ROOT / "data_processed" / "shards"

# Mirrors 02_scrape_policies.CORE_GENAI_TERMS. Kept inline so this script
# can run without importing the scraper module.
CORE_GENAI_TERMS = [
    "chatgpt", "generative ai", "genai", "large language model", "llm",
    "gpt-3", "gpt-4", "gpt-5", "claude", "gemini", "copilot",
    "鐢熸垚寮忎汉宸ユ櫤鑳?, "鐢熸垚寮廰i", "澶ц瑷€妯″瀷", "閫氱敤浜哄伐鏅鸿兘",
    "generative ki", "generative k眉nstliche", "sprachmodell",
    "ia generativa", "inteligencia artificial generativa",
    "intelig锚ncia artificial generativa",
    "ia g茅n茅rative", "intelligence artificielle g茅n茅rative",
    "鐢熸垚ai", "靸濎劚順?ai", "靸濎劚 ai", "generatieve ai",
]


def approx_tokens(text: str, lang: str) -> int:
    """Coarse token count; English ~ 1.3 tokens/word, CJK ~ 1 char/token."""
    if lang.startswith("zh") or lang in ("ja", "ko"):
        # CJK: roughly 1 token per CJK character
        cjk = sum(1 for c in text if "銆€" <= c <= "榭?)
        latin_words = len(re.findall(r"[A-Za-z]+", text))
        return cjk + latin_words
    return int(len(text.split()) * 1.3)


def split_into_blocks(text: str) -> list[tuple[str, str]]:
    """Split text into (heading, body) blocks.

    A 'block' is a section delimited by Markdown-style ## headings OR by
    visible heading markers we standardize in script 03. If no heading is
    present, the entire document is one block with heading = "".
    """
    blocks: list[tuple[str, str]] = []
    cur_heading = ""
    cur_body: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if m:
            if cur_body:
                blocks.append((cur_heading, "\n".join(cur_body).strip()))
                cur_body = []
            cur_heading = m.group(2).strip()
            continue
        cur_body.append(line)
    if cur_body:
        blocks.append((cur_heading, "\n".join(cur_body).strip()))
    return [(h, b) for h, b in blocks if b]


def pack_paragraphs(blocks: list[tuple[str, str]], lang: str,
                     target_tokens: int = 500,
                     min_tokens: int = 120,
                     max_tokens: int = 750
                     ) -> list[tuple[str, str]]:
    """Greedy pack paragraphs into ~target_tokens chunks. Yields (heading, body)."""
    out: list[tuple[str, str]] = []
    for heading, body in blocks:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", body) if p.strip()]
        if not paragraphs:
            continue
        buf: list[str] = []
        buf_tokens = 0
        for para in paragraphs:
            ptok = approx_tokens(para, lang)
            # Hard limit: if single paragraph exceeds max, split on sentences
            if ptok > max_tokens:
                if buf:
                    out.append((heading, "\n\n".join(buf)))
                    buf, buf_tokens = [], 0
                for sent_chunk in _split_long_paragraph(para, lang, max_tokens):
                    out.append((heading, sent_chunk))
                continue
            if buf_tokens + ptok > target_tokens and buf_tokens >= min_tokens:
                out.append((heading, "\n\n".join(buf)))
                buf, buf_tokens = [], 0
            buf.append(para)
            buf_tokens += ptok
        if buf and buf_tokens >= min_tokens:
            out.append((heading, "\n\n".join(buf)))
        elif buf and out:
            # Tail too small; merge into previous chunk
            prev_h, prev_b = out[-1]
            out[-1] = (prev_h, prev_b + "\n\n" + "\n\n".join(buf))
    return out


def _split_long_paragraph(p: str, lang: str, max_tokens: int) -> list[str]:
    """Split a paragraph that exceeds max_tokens on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?銆傦紒锛焆)\s+", p) if lang != "ja" \
        else re.split(r"(?<=[銆傦紒锛焆)", p)
    out: list[str] = []
    buf: list[str] = []
    buf_tok = 0
    for s in sentences:
        st = approx_tokens(s, lang)
        if buf_tok + st > max_tokens and buf:
            out.append(" ".join(buf))
            buf, buf_tok = [], 0
        buf.append(s)
        buf_tok += st
    if buf:
        out.append(" ".join(buf))
    return out


def contains_core_genai_term(text: str) -> bool:
    lc = text.lower()
    return any(t in lc for t in CORE_GENAI_TERMS)


def chunk_id_of(institution_id: str, archive_id: str, seq: int) -> str:
    return f"{institution_id}_{archive_id[:8]}_{seq:03d}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-docs", default=str(POLICY_DOCS_JSONL),
                    help="path to data_processed/policy_documents.jsonl from 03")
    ap.add_argument("--shards-dir", default=str(SHARDS_DIR))
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--inst-csv", default="data/institution_list.csv")
    ap.add_argument("--target-tokens", type=int, default=500)
    ap.add_argument("--require-genai-term", action="store_true", default=True,
                    help="drop chunks lacking any GenAI-specific term")
    ap.add_argument("--no-require-genai-term", dest="require_genai_term",
                    action="store_false")
    ap.add_argument("--limit-per-institution", type=int, default=12,
                    help="cap chunks per institution to limit dominance")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    policy_docs = Path(args.policy_docs)
    shards_dir = Path(args.shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    institutions = {row["institution_id"]: row
                    for row in load_csv_dicts(Path(args.inst_csv))}
    if not policy_docs.exists():
        print(f"ERROR: policy docs JSONL not found: {policy_docs}", file=sys.stderr)
        return 1

    docs: list[dict[str, Any]] = []
    with policy_docs.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    docs.append(json.loads(line))
                except Exception as exc:
                    print(f"[chunk] bad jsonl row: {exc}", file=sys.stderr)
    print(f"[chunk] {len(docs)} extracted documents loaded")
    if not docs:
        return 2

    per_inst_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_kept = 0
    total_dropped = 0

    for doc in docs:
        institution_id = doc.get("institution_id", "")
        inst = institutions.get(institution_id)
        if not inst:
            print(f"[chunk] unknown institution {institution_id}, skipping", file=sys.stderr)
            continue
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        lang = doc.get("language_detected") or inst.get("language_primary", "en")
        # Archive identifier: 03 writes 'sha256' (body hash); fall back to URL hash
        archive_id = doc.get("sha256") or doc.get("archive_sha") or \
                     hashlib.sha256(
                         (doc.get("url") or doc.get("source_url", "")).encode("utf-8")
                     ).hexdigest()
        source_url = doc.get("url") or doc.get("source_url", "")

        blocks = split_into_blocks(text)
        packed = pack_paragraphs(blocks, lang, target_tokens=args.target_tokens)

        for seq, (heading, body) in enumerate(packed):
            if args.require_genai_term and not contains_core_genai_term(body):
                total_dropped += 1
                continue
            cid = chunk_id_of(institution_id, archive_id, seq)
            rec = {
                "chunk_id": cid,
                "institution_id": institution_id,
                "institution_name": inst.get("name", ""),
                "region": inst.get("region", ""),
                "tier": inst.get("tier", ""),
                "country_code": inst.get("country_code", ""),
                "language_declared": lang,
                "section_header": heading,
                "source_url": source_url,
                "text": body,
                "n_words": len(body.split()),
                "approx_tokens": approx_tokens(body, lang),
                "sha256_chunk": sha256_text(body),
            }
            per_inst_chunks[institution_id].append(rec)
            total_kept += 1

    # Apply per-institution cap (preserves diversity over dominance)
    if args.limit_per_institution > 0:
        capped: list[dict[str, Any]] = []
        for iid, recs in per_inst_chunks.items():
            recs.sort(key=lambda r: -r["approx_tokens"])  # prefer longer/richer chunks
            capped.extend(recs[: args.limit_per_institution])
        all_chunks = capped
    else:
        all_chunks = [r for recs in per_inst_chunks.values() for r in recs]

    print(f"[chunk] kept {total_kept}, dropped {total_dropped} "
          f"(no GenAI term), final after cap = {len(all_chunks)}")

    # Stratified sharding: hash(chunk_id) -> shard index
    shards: list[list[dict[str, Any]]] = [[] for _ in range(args.n_shards)]
    for r in all_chunks:
        h = int(hashlib.sha256(r["chunk_id"].encode("utf-8")).hexdigest(), 16)
        shards[h % args.n_shards].append(r)

    if args.dry_run:
        for i, s in enumerate(shards):
            print(f"  shard {i:02d}: {len(s)} chunks "
                  f"({len({r['institution_id'] for r in s})} institutions, "
                  f"{len({r['region'] for r in s})} regions)")
        return 0

    for i, s in enumerate(shards):
        out_path = shards_dir / f"policy_chunks_shard_{i:02d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in s:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[chunk] wrote {out_path.name} ({len(s)} chunks)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

