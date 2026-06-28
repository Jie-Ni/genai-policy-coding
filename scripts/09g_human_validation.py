"""Cohen's kappa between expert hand-coded gold standard and Qwen3.6 codings,
per theme + overall, with bootstrap 95% CIs.

Inputs:
  data_processed/expert_gold_codes.csv  (40 chunks 脳 8 themes 脳 4 sentiments)
  results/codes/qwen36/shard_*.jsonl   (full coding output)

Output:
  results/stats/human_validation_kappa.csv
  results/stats/human_validation_summary.md
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPERT = ROOT / "data_processed" / "expert_gold_codes.csv"
QWEN_DIR = ROOT / "results" / "codes" / "qwen36"
OUT = ROOT / "results" / "stats"

THEMES = ["T1_integration", "T2_multimodal", "T3_privacy", "T4_integrity",
          "T5_disclosure", "T6_equity", "T7_vendor_governance",
          "T8_pedagogical_redesign"]
SENTS = ["sent_assessment", "sent_research", "sent_teaching", "sent_administration"]
QWEN_SENTS = ["assessment", "research", "teaching", "administration"]


def cohen_kappa(a, b):
    if not a:
        return float("nan")
    n = len(a)
    p_obs = sum(1 for x, y in zip(a, b) if x == y) / n
    classes = sorted(set(a) | set(b))
    p_exp = sum((a.count(c) / n) * (b.count(c) / n) for c in classes)
    return (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else 1.0


def quadratic_kappa(a, b):
    """Weighted 魏 with quadratic weights for ordinal sentiment."""
    if not a:
        return float("nan")
    n = len(a)
    classes = sorted(set(a) | set(b))
    k2 = len(classes)
    class_idx = {c: i for i, c in enumerate(classes)}
    obs = np.zeros((k2, k2))
    for x, y in zip(a, b):
        obs[class_idx[x], class_idx[y]] += 1
    obs /= n
    row = obs.sum(axis=1)
    col = obs.sum(axis=0)
    exp = np.outer(row, col)
    w = np.array([[(i - j) ** 2 / max(1, (k2 - 1) ** 2)
                    for j in range(k2)] for i in range(k2)])
    num = (w * obs).sum()
    den = (w * exp).sum()
    return 1 - num / den if den > 0 else float("nan")


def bootstrap_ci(fn, a, b, B=2000, seed=11):
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    arr_a = np.array(a)
    arr_b = np.array(b)
    boots = np.empty(B)
    idx = np.arange(n)
    for i in range(B):
        s = rng.choice(idx, size=n, replace=True)
        boots[i] = fn(arr_a[s].tolist(), arr_b[s].tolist())
    finite = boots[np.isfinite(boots)]
    if len(finite) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(finite, 2.5)),
            float(np.percentile(finite, 97.5)))


def main():
    # Load expert codings
    expert = {}
    with EXPERT.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expert[row["chunk_id"]] = row

    # Load Qwen codings for same chunk_ids
    qwen = {}
    for sp in sorted(QWEN_DIR.glob("shard_*.jsonl")):
        for line in sp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("ok") and r["chunk_id"] in expert:
                qwen[r["chunk_id"]] = r

    common = sorted(set(expert) & set(qwen))
    print(f"[H] {len(common)} chunks have both expert + Qwen codings")

    rows = []
    # Themes (binary)
    for theme in THEMES:
        a = [int(expert[c][theme]) for c in common]
        b = [int(qwen[c]["themes"].get(theme, 0)) for c in common]
        k = cohen_kappa(a, b)
        lo, hi = bootstrap_ci(cohen_kappa, a, b, B=2000)
        agree = sum(1 for x, y in zip(a, b) if x == y) / len(a)
        rows.append({"variable": theme, "type": "binary", "n": len(a),
                      "kappa": k, "ci95_lo": lo, "ci95_hi": hi,
                      "raw_agreement": agree,
                      "expert_pos": sum(a), "qwen_pos": sum(b)})

    # Sentiments (ordinal, quadratic-weighted)
    for esent, qsent in zip(SENTS, QWEN_SENTS):
        a = [int(expert[c][esent]) for c in common]
        b = [int(qwen[c]["sentiment"].get(qsent, 0)) for c in common]
        k = quadratic_kappa(a, b)
        lo, hi = bootstrap_ci(quadratic_kappa, a, b, B=2000)
        agree = sum(1 for x, y in zip(a, b) if x == y) / len(a)
        rows.append({"variable": qsent, "type": "ordinal_weighted",
                      "n": len(a), "kappa": k, "ci95_lo": lo, "ci95_hi": hi,
                      "raw_agreement": agree,
                      "expert_mean": sum(a) / len(a),
                      "qwen_mean": sum(b) / len(b)})

    # Write CSV
    keys = list(rows[0].keys())
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "human_validation_kappa.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + sorted(set(k for r in rows for k in r.keys()) - set(keys)))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[H] wrote {out_csv}")

    # Summary markdown
    md = ["# Human-Expert vs Qwen3.6 Agreement (n = " + str(len(common)) + " chunks)", ""]
    md.append("## Binary themes (Cohen's 魏, unweighted)")
    md.append("")
    md.append("| Theme | n | 魏 | 95% CI | raw agreement | expert+ | qwen+ |")
    md.append("|---|---|---|---|---|---|---|")
    bin_ks = []
    for r in rows:
        if r["type"] == "binary":
            bin_ks.append(r["kappa"])
            md.append(f"| {r['variable']} | {r['n']} | {r['kappa']:.3f} | "
                       f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}] | "
                       f"{r['raw_agreement']:.2%} | {r['expert_pos']} | {r['qwen_pos']} |")
    md.append(f"\n**Mean theme 魏 vs human expert = {np.mean(bin_ks):.3f}**")
    md.append("")
    md.append("## Ordinal sentiment (quadratic-weighted Cohen's 魏)")
    md.append("")
    md.append("| Use case | n | 魏 | 95% CI | raw agreement |")
    md.append("|---|---|---|---|---|")
    sent_ks = []
    for r in rows:
        if r["type"] == "ordinal_weighted":
            sent_ks.append(r["kappa"])
            md.append(f"| {r['variable']} | {r['n']} | {r['kappa']:.3f} | "
                       f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}] | "
                       f"{r['raw_agreement']:.2%} |")
    md.append(f"\n**Mean sentiment quadratic-魏 vs human expert = {np.mean(sent_ks):.3f}**")
    (OUT / "human_validation_summary.md").write_text("\n".join(md) + "\n",
                                                      encoding="utf-8")
    print(f"[H] mean theme 魏 vs human = {np.mean(bin_ks):.3f}")
    print(f"[H] mean sentiment 魏 vs human = {np.mean(sent_ks):.3f}")


if __name__ == "__main__":
    main()

