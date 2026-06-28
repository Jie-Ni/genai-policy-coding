"""Compare ablation conditions to the 16-fewshot/T=0/seed=11 baseline.

Inputs
------
results/codes/ablation/<condition>/codes.jsonl   for each condition

Outputs
-------
results/stats/ablation_B_fewshot_kappa.csv     # per-theme 魏 vs baseline
results/stats/ablation_B_temperature_kappa.csv # T=0.3 vs T=0
results/stats/ablation_B_seed_kappa.csv        # seed 22/42 vs 11
results/stats/ablation_B_prevalence.csv        # per-region theme prevalence in each condition
results/stats/ablation_B_summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util as _iu
_spec = _iu.spec_from_file_location("stats_mod", ROOT / "scripts" / "09_stats_analysis.py")
_stats = _iu.module_from_spec(_spec)
sys.modules["stats_mod"] = _stats
_spec.loader.exec_module(_stats)
cohen_kappa = _stats.cohen_kappa
THEMES = _stats.THEMES
REGIONS = _stats.REGIONS

BASELINE = "baseline_16fs"


def load_codes(p: Path) -> dict[str, dict]:
    """Return {chunk_id -> coded_row} for an ablation condition."""
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("ok"):
            out[r["chunk_id"]] = r
    return out


def per_theme_kappa(a: dict, b: dict) -> list[dict]:
    common = sorted(set(a) & set(b))
    rows = []
    for t in THEMES:
        xa = [a[cid]["themes"].get(t, 0) for cid in common]
        xb = [b[cid]["themes"].get(t, 0) for cid in common]
        k = cohen_kappa(xa, xb)
        rows.append({"theme": t, "n": k.get("n", 0),
                     "kappa": k.get("kappa"),
                     "p_obs": k.get("p_obs")})
    return rows


def region_prevalence(codes: dict) -> dict[str, dict[str, float]]:
    """For each region, fraction of chunks where each theme = 1."""
    by_r: dict[str, list[dict]] = {r: [] for r in REGIONS}
    for cid, row in codes.items():
        by_r.setdefault(row["region"], []).append(row)
    out: dict[str, dict[str, float]] = {}
    for r, rows in by_r.items():
        if not rows:
            continue
        out[r] = {}
        for t in THEMES:
            pos = sum(1 for x in rows if x["themes"].get(t, 0) == 1)
            out[r][t] = pos / len(rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", default="results/codes/ablation")
    ap.add_argument("--out-dir", default="results/stats")
    args = ap.parse_args()

    ablation_dir = Path(args.ablation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = sorted(p.name for p in ablation_dir.iterdir() if p.is_dir())
    print(f"[B] conditions found: {conditions}")
    if BASELINE not in conditions:
        print(f"ERROR: baseline {BASELINE} not found", file=sys.stderr)
        return 1

    codes_by_cond: dict[str, dict] = {}
    for c in conditions:
        p = ablation_dir / c / "codes.jsonl"
        if p.exists():
            codes_by_cond[c] = load_codes(p)
            print(f"  {c}: {len(codes_by_cond[c])} chunks")

    base = codes_by_cond[BASELINE]

    # B1: per-condition 魏 vs baseline
    rows: list[dict] = []
    for c in conditions:
        if c == BASELINE:
            continue
        ks = per_theme_kappa(base, codes_by_cond[c])
        mean_k = sum(k["kappa"] for k in ks if k["kappa"] is not None) / len(ks)
        for k in ks:
            rows.append({"condition": c, "theme": k["theme"],
                         "n": k["n"], "kappa_vs_baseline": k["kappa"]})
        rows.append({"condition": c, "theme": "MEAN",
                     "n": ks[0]["n"], "kappa_vs_baseline": mean_k})
    _write_csv(out_dir / "ablation_B_kappa_vs_baseline.csv", rows)

    # B2: regional prevalence stability
    prev_rows: list[dict] = []
    for c in conditions:
        rp = region_prevalence(codes_by_cond[c])
        for region, themes in rp.items():
            for t, v in themes.items():
                prev_rows.append({"condition": c, "region": region,
                                   "theme": t, "prevalence": v})
    _write_csv(out_dir / "ablation_B_region_prevalence.csv", prev_rows)

    # B3: T7 NA>LA gap robustness 鈥?does the most striking finding hold under all conditions?
    t7_rows: list[dict] = []
    for c in conditions:
        rp = region_prevalence(codes_by_cond[c])
        na_t7 = rp.get("NA", {}).get("T7_vendor_governance", 0)
        la_t7 = rp.get("LA", {}).get("T7_vendor_governance", 0)
        eu_t7 = rp.get("EU", {}).get("T7_vendor_governance", 0)
        ea_t7 = rp.get("EA", {}).get("T7_vendor_governance", 0)
        t7_rows.append({"condition": c,
                         "T7_NA": na_t7, "T7_EU": eu_t7,
                         "T7_EA": ea_t7, "T7_LA": la_t7,
                         "NA_minus_LA": na_t7 - la_t7,
                         "rank_order_correct": int(na_t7 >= eu_t7 >= ea_t7 >= la_t7),
                         })
    _write_csv(out_dir / "ablation_B_T7_gradient.csv", t7_rows)

    # Markdown summary
    lines = ["# Ablation Studies 鈥?Few-shot, Temperature, Seed", ""]
    lines.append(f"All ablation runs use 80-chunk stratified subset (20/region) "
                 f"of the production corpus. Baseline = {BASELINE} (16-shot, T=0, seed=11).")
    lines.append("")
    lines.append("## B1. Cohen's 魏 vs baseline (mean across 8 themes)")
    lines.append("")
    lines.append("| Condition | Mean 魏 vs baseline |")
    lines.append("|-----------|---------------------|")
    mean_per_cond = {}
    for r in rows:
        if r["theme"] == "MEAN":
            mean_per_cond[r["condition"]] = r["kappa_vs_baseline"]
    for c, k in sorted(mean_per_cond.items(), key=lambda x: -x[1] if x[1] is not None else 0):
        lines.append(f"| {c} | {k:.3f} |" if k is not None else f"| {c} | nan |")
    lines.append("")
    lines.append("## B2. T7 vendor-governance NA-LA gradient across conditions")
    lines.append("")
    lines.append("Does NA > EU > EA > LA hold under every ablation? (Rank-order check.)")
    lines.append("")
    lines.append("| Condition | T7 NA | T7 EU | T7 EA | T7 LA | NA-LA | Rank OK? |")
    lines.append("|-----------|-------|-------|-------|-------|-------|----------|")
    for r in t7_rows:
        lines.append(f"| {r['condition']} | {r['T7_NA']:.2f} | {r['T7_EU']:.2f} | "
                      f"{r['T7_EA']:.2f} | {r['T7_LA']:.2f} | {r['NA_minus_LA']:+.2f} | "
                      f"{'yes' if r['rank_order_correct'] else 'no'} |")
    (out_dir / "ablation_B_summary.md").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")
    print(f"[B] wrote {out_dir / 'ablation_B_summary.md'}")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            s = str(v) if v is not None else ""
            if "," in s or '"' in s:
                s = '"' + s.replace('"', '""') + '"'
            cells.append(s)
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

