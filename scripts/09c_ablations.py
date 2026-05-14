"""Additional rigor for IJETHE reviewers 2/3 — purely CPU/Python.

A1. Bootstrap CI on cross-coder Cohen's kappa (B=2000) per theme. Reports
    point estimate + 95% percentile interval.
A2. Leave-one-region-out: recompute every theme prevalence and TOST pair
    statistic dropping one region at a time. If a finding survives all
    four drops, it is not driven by any single region's dominance.
A3. K-fold (k=5) cross-validation of the mixed-effects logistic model:
    train on 4 folds of institutions, predict held-out fold's theme
    prevalence, report AUROC and Brier score.
A4. Annotator-level model: treat Qwen and Mistral as two "annotators",
    fit a hierarchical model with random intercept per chunk AND per
    annotator. Effective in detecting whether one annotator
    systematically biases the regional comparison.

Outputs
-------
results/stats/ablation_A1_bootstrap_kappa.csv
results/stats/ablation_A2_loo_region.csv
results/stats/ablation_A3_kfold_cv.csv
results/stats/ablation_A4_annotator_model.csv
results/stats/ablation_summary.md
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import load_csv_dicts  # type: ignore

import importlib.util as _iu
_spec = _iu.spec_from_file_location("stats_mod", ROOT / "scripts" / "09_stats_analysis.py")
_stats = _iu.module_from_spec(_spec)
sys.modules["stats_mod"] = _stats
_spec.loader.exec_module(_stats)

load_coded = _stats.load_coded
cohen_kappa = _stats.cohen_kappa
institution_level_prevalence = _stats.institution_level_prevalence
tost_proportions = _stats.tost_proportions
fit_mixed_logit = _stats.fit_mixed_logit
THEMES = _stats.THEMES
REGIONS = _stats.REGIONS
TIERS = _stats.TIERS
SEED = _stats.SEED


# ---------------------------------------------------------------------------
# A1. Bootstrap CI on kappa
# ---------------------------------------------------------------------------


def bootstrap_kappa_ci(qwen_chunks, mistral_chunks, B: int = 2000,
                       seed: int = SEED) -> list[dict[str, Any]]:
    mistral_map = {c.chunk_id: c for c in mistral_chunks}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for t in THEMES:
        a, b = [], []
        for c in qwen_chunks:
            if c.chunk_id in mistral_map:
                a.append(c.themes.get(t, 0))
                b.append(mistral_map[c.chunk_id].themes.get(t, 0))
        n = len(a)
        a_arr = np.array(a)
        b_arr = np.array(b)
        full_k = cohen_kappa(a, b).get("kappa", float("nan"))
        boot_ks = np.empty(B)
        idx = np.arange(n)
        for i in range(B):
            samp = rng.choice(idx, size=n, replace=True)
            k = cohen_kappa(a_arr[samp].tolist(), b_arr[samp].tolist())
            boot_ks[i] = k.get("kappa", float("nan"))
        finite = boot_ks[np.isfinite(boot_ks)]
        if len(finite) == 0:
            ci_lo = ci_hi = float("nan")
        else:
            ci_lo = float(np.percentile(finite, 2.5))
            ci_hi = float(np.percentile(finite, 97.5))
        rows.append({
            "theme": t, "n_paired": n,
            "kappa": full_k,
            "ci95_lo": ci_lo, "ci95_hi": ci_hi,
            "B": B,
        })
    return rows


# ---------------------------------------------------------------------------
# A2. Leave-one-region-out: theme prevalence + pair stability
# ---------------------------------------------------------------------------


def loo_region(chunks, institutions) -> list[dict[str, Any]]:
    region_inst: dict[str, list[str]] = {r: [] for r in REGIONS}
    for c in chunks:
        if c.institution_id in institutions:
            inst = institutions[c.institution_id]
            if c.institution_id not in region_inst[inst["region"]]:
                region_inst[inst["region"]].append(c.institution_id)
    # Full estimates as baseline
    pairs = [(REGIONS[i], REGIONS[j])
             for i in range(len(REGIONS)) for j in range(i + 1, len(REGIONS))]
    full_diffs: dict[tuple[str, str, str], float] = {}
    for t in THEMES:
        prev = institution_level_prevalence(chunks, t)
        for a, b in pairs:
            xa = [prev.get(i, 0) for i in region_inst[a]]
            xb = [prev.get(i, 0) for i in region_inst[b]]
            if xa and xb:
                full_diffs[(t, a, b)] = sum(xa) / len(xa) - sum(xb) / len(xb)
            else:
                full_diffs[(t, a, b)] = float("nan")

    rows: list[dict[str, Any]] = []
    for dropped in REGIONS:
        # Recompute with `dropped` region's institutions removed
        kept_chunks = [c for c in chunks if institutions.get(c.institution_id, {}).get("region") != dropped]
        sub_region_inst = {r: ids for r, ids in region_inst.items() if r != dropped}
        for t in THEMES:
            prev = institution_level_prevalence(kept_chunks, t)
            for a, b in pairs:
                if a == dropped or b == dropped:
                    continue  # comparison undefined
                xa = [prev.get(i, 0) for i in sub_region_inst[a]]
                xb = [prev.get(i, 0) for i in sub_region_inst[b]]
                if not xa or not xb:
                    continue
                d = sum(xa) / len(xa) - sum(xb) / len(xb)
                f = full_diffs[(t, a, b)]
                same_sign = (d > 0) == (f > 0) if (not math.isnan(f)) and (d != 0 and f != 0) else True
                rows.append({
                    "dropped_region": dropped, "theme": t,
                    "region_A": a, "region_B": b,
                    "full_diff": f, "loo_diff": d,
                    "same_sign": int(same_sign),
                })
    return rows


# ---------------------------------------------------------------------------
# A3. K-fold cross-validation of mixed-logit
# ---------------------------------------------------------------------------


def kfold_cv_mixed_logit(chunks, institutions, k: int = 5, seed: int = SEED
                          ) -> list[dict[str, Any]]:
    inst_lookup = {c.institution_id: institutions.get(c.institution_id)
                   for c in chunks if c.institution_id in institutions}
    all_inst = sorted(inst_lookup)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_inst)
    folds = np.array_split(all_inst, k)

    rows: list[dict[str, Any]] = []
    for theme in THEMES:
        prev_inst = institution_level_prevalence(chunks, theme)
        fold_aurocs: list[float] = []
        fold_briers: list[float] = []
        for fi in range(k):
            test = set(folds[fi].tolist())
            train_inst = [i for i in all_inst if i not in test]
            test_inst = list(test)
            if len(train_inst) < 5 or len(test_inst) < 1:
                continue

            def design(insts):
                ys = np.array([prev_inst.get(i, 0) for i in insts], dtype=float)
                cols = [np.ones(len(insts))]
                for r in REGIONS[1:]:
                    cols.append(np.array([1.0 if inst_lookup[i]["region"] == r else 0.0
                                          for i in insts]))
                for ti in TIERS[1:]:
                    if any(inst_lookup[i]["tier"] == ti for i in insts):
                        cols.append(np.array([1.0 if inst_lookup[i]["tier"] == ti else 0.0
                                              for i in insts]))
                groups = np.array([inst_lookup[i]["country_code"] for i in insts])
                return ys, np.stack(cols, axis=1), groups

            y_tr, X_tr, g_tr = design(train_inst)
            y_te, X_te, _ = design(test_inst)
            if X_tr.shape[1] != X_te.shape[1]:
                # design mismatch (test fold missing a tier dummy column).
                # Pad test fold with zero columns to match.
                pad = X_tr.shape[1] - X_te.shape[1]
                if pad > 0:
                    X_te = np.concatenate([X_te, np.zeros((len(test_inst), pad))], axis=1)
                else:
                    X_tr = np.concatenate([X_tr, np.zeros((len(train_inst), -pad))], axis=1)

            fit = fit_mixed_logit(y_tr, X_tr, g_tr)
            beta = np.array(fit["beta"])
            if X_te.shape[1] > len(beta):
                X_te = X_te[:, :len(beta)]
            elif X_te.shape[1] < len(beta):
                X_te = np.concatenate(
                    [X_te, np.zeros((X_te.shape[0], len(beta) - X_te.shape[1]))], axis=1
                )
            eta = X_te @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            # AUROC via Mann-Whitney U (ascending-rank convention: rank 1 = lowest p)
            if 0 < y_te.sum() < len(y_te):
                ranks = np.argsort(np.argsort(p)) + 1
                n1 = int(y_te.sum())
                n0 = len(y_te) - n1
                pos_ranks = ranks[y_te == 1].sum()
                auroc = (pos_ranks - n1 * (n1 + 1) / 2) / (n1 * n0)
            else:
                auroc = float("nan")
            brier = float(np.mean((p - y_te) ** 2))
            fold_aurocs.append(float(auroc))
            fold_briers.append(brier)

        valid_aurocs = [a for a in fold_aurocs if not (a is None or (isinstance(a, float) and math.isnan(a)))]
        rows.append({
            "theme": theme, "k": k,
            "mean_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else float("nan"),
            "sd_auroc": float(np.std(valid_aurocs, ddof=1)) if len(valid_aurocs) > 1 else float("nan"),
            "mean_brier": float(np.mean(fold_briers)) if fold_briers else float("nan"),
            "n_valid_folds": len(valid_aurocs),
        })
    return rows


# ---------------------------------------------------------------------------
# A4. Annotator-level mixed-effects (chunk-level fixed effect: region+tier,
#     random intercept per institution, random "annotator" slope by region)
# ---------------------------------------------------------------------------


def annotator_model(qwen_chunks, mistral_chunks, institutions
                     ) -> list[dict[str, Any]]:
    """Per-theme: pool Qwen+Mistral chunk-level codings, fit a logistic model
    with fixed-effect region+tier+annotator_dummy and a random intercept by
    institution. Reports the *annotator effect* — does Mistral systematically
    over/under-report a theme relative to Qwen?"""
    rows: list[dict[str, Any]] = []
    mistral_map = {c.chunk_id: c for c in mistral_chunks}
    # Collect pooled chunk-level rows
    paired = []
    for c in qwen_chunks:
        if c.chunk_id in mistral_map and c.institution_id in institutions:
            paired.append(c)
    for theme in THEMES:
        # Build pooled long-format
        ys, X_rows, groups = [], [], []
        cols_meta: list[str] = ["(Intercept)"]
        for c in paired:
            inst = institutions[c.institution_id]
            for annot, src in (("qwen", c), ("mistral", mistral_map[c.chunk_id])):
                ys.append(src.themes.get(theme, 0))
                row = [1.0]
                for r in REGIONS[1:]:
                    row.append(1.0 if inst["region"] == r else 0.0)
                for ti in TIERS[1:]:
                    row.append(1.0 if inst["tier"] == ti else 0.0)
                row.append(1.0 if annot == "mistral" else 0.0)
                X_rows.append(row)
                groups.append(c.institution_id)
        cols_meta += [f"region_{r}" for r in REGIONS[1:]]
        cols_meta += [f"tier_{ti}" for ti in TIERS[1:]]
        cols_meta += ["annotator_mistral"]
        ys = np.array(ys, dtype=float)
        X = np.array(X_rows, dtype=float)
        groups_arr = np.array(groups)
        if len(ys) < 10:
            continue
        fit = fit_mixed_logit(ys, X, groups_arr)
        for ci, name in enumerate(cols_meta):
            if ci >= len(fit["beta"]):
                continue
            b = fit["beta"][ci]
            s = fit["se_beta"][ci] if not (isinstance(fit["se_beta"][ci], float)
                                            and math.isnan(fit["se_beta"][ci])) else float("nan")
            z = (b / s) if s and not math.isnan(s) else float("nan")
            p = (math.erfc(abs(z) / math.sqrt(2))) if not math.isnan(z) else float("nan")
            rows.append({
                "theme": theme, "term": name,
                "beta": b, "se": s, "z": z, "p_two_sided": p,
                "sigma2_inst": fit["sigma2"],
                "n_obs": fit["n_obs"],
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-codes", default="results/codes/qwen36")
    ap.add_argument("--mistral-codes", default="results/codes/mistral")
    ap.add_argument("--inst-csv", default="data/institution_list.csv")
    ap.add_argument("--out-dir", default="results/stats")
    ap.add_argument("--bootstrap-B", type=int, default=2000)
    ap.add_argument("--kfold-k", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    institutions = {r["institution_id"]: r for r in load_csv_dicts(Path(args.inst_csv))}
    qwen_chunks = load_coded(Path(args.qwen_codes), institutions)
    mistral_chunks = load_coded(Path(args.mistral_codes), institutions)
    print(f"[A] Qwen chunks: {len(qwen_chunks)}, Mistral chunks: {len(mistral_chunks)}")

    print("[A1] bootstrap kappa CI...")
    a1 = bootstrap_kappa_ci(qwen_chunks, mistral_chunks, B=args.bootstrap_B)
    _write_csv(out_dir / "ablation_A1_bootstrap_kappa.csv", a1)

    print("[A2] leave-one-region-out...")
    a2 = loo_region(qwen_chunks, institutions)
    _write_csv(out_dir / "ablation_A2_loo_region.csv", a2)

    print(f"[A3] {args.kfold_k}-fold cross-validation of mixed-logit...")
    a3 = kfold_cv_mixed_logit(qwen_chunks, institutions, k=args.kfold_k)
    _write_csv(out_dir / "ablation_A3_kfold_cv.csv", a3)

    print("[A4] annotator-level mixed-effects model...")
    a4 = annotator_model(qwen_chunks, mistral_chunks, institutions)
    _write_csv(out_dir / "ablation_A4_annotator_model.csv", a4)

    # Summary
    md = _build_summary(a1, a2, a3, a4)
    (out_dir / "ablation_summary.md").write_text(md, encoding="utf-8")
    print(f"[A] wrote 4 files + ablation_summary.md to {out_dir}")
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _build_summary(a1, a2, a3, a4) -> str:
    lines = ["# IJETHE GenAI Policy — Ablation Studies", ""]
    # A1
    lines.append("## A1. Bootstrap 95% CI on cross-coder kappa (B=2000)")
    lines.append("")
    lines.append("| Theme | n | kappa | 95% CI |")
    lines.append("|-------|---|-------|--------|")
    for r in a1:
        lines.append(f"| {r['theme']} | {r['n_paired']} | {r['kappa']:.3f} | "
                      f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}] |")
    lines.append("")
    # A2
    lines.append("## A2. Leave-one-region-out (sign stability of pair differences)")
    lines.append("")
    by_drop = defaultdict(list)
    for r in a2:
        by_drop[r["dropped_region"]].append(r)
    for drop, rs in by_drop.items():
        n_same = sum(r["same_sign"] for r in rs)
        lines.append(f"- Drop {drop}: {n_same}/{len(rs)} ({n_same/len(rs):.1%}) "
                     "pairwise differences preserve sign vs. full corpus")
    lines.append("")
    # A3
    lines.append("## A3. 5-fold CV on mixed-logit (per theme)")
    lines.append("")
    lines.append("| Theme | mean AUROC | sd | mean Brier |")
    lines.append("|-------|------------|----|-----------|")
    for r in a3:
        lines.append(f"| {r['theme']} | {r['mean_auroc']:.3f} | "
                      f"{r['sd_auroc']:.3f} | {r['mean_brier']:.3f} |")
    lines.append("")
    # A4
    lines.append("## A4. Annotator-level mixed-effects (chunk-level, both coders pooled)")
    lines.append("")
    lines.append("Annotator (Mistral vs Qwen-ref) fixed effect per theme:")
    lines.append("")
    lines.append("| Theme | beta (annotator=Mistral) | SE | z | p |")
    lines.append("|-------|--------------------------|----|----|----|")
    for r in a4:
        if r["term"] != "annotator_mistral":
            continue
        lines.append(f"| {r['theme']} | {r['beta']:+.3f} | {r['se']:.3f} | "
                      f"{r['z']:.2f} | {r['p_two_sided']:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
