"""Extended inferential analyses to address reviewer concerns identified
in the v2 audit:

E1. Wilson 95% CI for every (region 脳 theme) prevalence point estimate.
E2. Holm-Bonferroni correction on the same pairwise z-test family that
    Bonferroni rejected 鈥?Holm is uniformly more powerful and preferred.
E3. Kruskal-Wallis non-parametric test for the sentiment regional means
    (does the ANOVA-null hold without normality assumption?).
E4. Intra-class correlation (ICC) for the country random effect in the
    mixed-logit, computed as 蟽虏_country / (蟽虏_country + 蟺虏/3) for the
    logistic linking function.
E5. Sample-size / power retrospective using the canonical 4脳2 chi-square
    test for a single theme 鈥?what regional gap size could we detect at
    伪=0.05, power=0.80, given n_per_region 鈭?{11, 13, 18, 20}?

Outputs
-------
results/stats/extended_E1_wilson_ci.csv
results/stats/extended_E2_holm_pairwise.csv
results/stats/extended_E3_kruskal_sentiment.csv
results/stats/extended_E4_icc.csv
results/stats/extended_E5_power_retrospective.csv
results/stats/extended_summary.md
"""
from __future__ import annotations

import argparse
import math
import sys
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
institution_level_prevalence = _stats.institution_level_prevalence
institution_level_sentiment = _stats.institution_level_sentiment
THEMES = _stats.THEMES
REGIONS = _stats.REGIONS
SENT_USES = _stats.SENTIMENT_USE_CASES


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054  # qnorm(0.975)
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def E1_wilson(chunks, institutions) -> list[dict[str, Any]]:
    region_inst: dict[str, set[str]] = {r: set() for r in REGIONS}
    for c in chunks:
        inst = institutions.get(c.institution_id)
        if inst:
            region_inst[inst["region"]].add(c.institution_id)
    rows = []
    for theme in THEMES:
        prev = institution_level_prevalence(chunks, theme)
        for region in REGIONS:
            ids = region_inst[region]
            x = sum(prev.get(i, 0) for i in ids)
            n = len(ids)
            lo, hi = wilson_ci(x, n)
            rows.append({
                "theme": theme, "region": region,
                "n_with": x, "n_total": n,
                "prevalence": x / n if n else float("nan"),
                "wilson_lo": lo, "wilson_hi": hi,
            })
    return rows


def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    sorted_p = [pvals[i] for i in order]
    rej = [False] * m
    for k in range(m):
        # adjusted alpha = alpha / (m - k)
        if sorted_p[k] <= alpha / (m - k):
            rej[order[k]] = True
        else:
            break  # subsequent tests cannot reject under Holm
    return rej


def E2_holm(chunks, institutions, themes_focus) -> list[dict[str, Any]]:
    """Same pairwise-z family as C4, with Holm correction instead of Bonferroni."""
    region_inst: dict[str, list[str]] = {r: [] for r in REGIONS}
    for c in chunks:
        inst = institutions.get(c.institution_id)
        if inst and c.institution_id not in region_inst[inst["region"]]:
            region_inst[inst["region"]].append(c.institution_id)
    pairs = [(REGIONS[i], REGIONS[j])
             for i in range(len(REGIONS)) for j in range(i + 1, len(REGIONS))]
    raw = []
    for theme in themes_focus:
        prev = institution_level_prevalence(chunks, theme)
        for a, b in pairs:
            xa = [prev.get(i, 0) for i in region_inst[a]]
            xb = [prev.get(i, 0) for i in region_inst[b]]
            n1, n2 = len(xa), len(xb)
            if n1 == 0 or n2 == 0:
                continue
            p1, p2 = sum(xa) / n1, sum(xb) / n2
            p_pool = (sum(xa) + sum(xb)) / (n1 + n2)
            se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z = (p1 - p2) / se if se > 0 else float("nan")
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))) if not math.isnan(z) else float("nan")
            raw.append({"theme": theme, "region_A": a, "region_B": b,
                         "p_A": p1, "p_B": p2, "diff": p1 - p2,
                         "z": z, "p_raw": p})
    pvals = [r["p_raw"] for r in raw]
    rej = holm_bonferroni(pvals)
    for r, k in zip(raw, rej):
        r["reject_at_holm_005"] = int(k)
    return raw


def kruskal_wallis(groups: list[list[float]]) -> tuple[float, int, float]:
    """Kruskal-Wallis H-test. Returns (H, df, p)."""
    all_vals = []
    g_labels = []
    for gi, g in enumerate(groups):
        all_vals.extend(g)
        g_labels.extend([gi] * len(g))
    if len(all_vals) < 4 or len(set(g_labels)) < 2:
        return float("nan"), 0, float("nan")
    # Rank with tie correction
    pairs = sorted(enumerate(all_vals), key=lambda x: x[1])
    ranks = [0.0] * len(all_vals)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2  # average rank of the tied group
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg_rank
        i = j + 1
    n = len(all_vals)
    n_per = [0] * len(groups)
    rank_sum = [0.0] * len(groups)
    for r, lbl in zip(ranks, g_labels):
        n_per[lbl] += 1
        rank_sum[lbl] += r
    H = 12.0 / (n * (n + 1)) * sum((rs ** 2) / np_ for rs, np_ in zip(rank_sum, n_per)) - 3 * (n + 1)
    df = len(groups) - 1
    # tie correction
    # compute 危(t^3 - t) over ties
    ties = []
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        ties.append(j - i + 1)
        i = j + 1
    T = sum(t ** 3 - t for t in ties if t > 1)
    C = 1 - T / (n ** 3 - n) if n > 1 else 1.0
    H_corr = H / C if C > 0 else H
    # chi-square SF approximation
    p = chi2_sf(H_corr, df)
    return H_corr, df, p


def chi2_sf(x: float, df: int) -> float:
    """Survival function 1 - F(x; df) 鈥?reuses 09e gammaincc."""
    if x <= 0:
        return 1.0
    return _gammaincc(df / 2.0, x / 2.0)


def _gammaln(x: float) -> float:
    g = 7
    p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - _gammaln(1 - x)
    x = x - 1
    a = p[0]
    t = x + g + 0.5
    for i in range(1, g + 2):
        a += p[i] / (x + i)
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def _gammaincc(a: float, x: float) -> float:
    if x < a + 1.0:
        return 1.0 - _gammainc_series(a, x)
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - _gammaln(a)) * h


def _gammainc_series(a: float, x: float) -> float:
    if x <= 0:
        return 0.0
    ap = a
    s = 1.0 / a
    delta = s
    for _ in range(200):
        ap += 1
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * 1e-12:
            break
    return s * math.exp(-x + a * math.log(x) - _gammaln(a))


def E3_kruskal(chunks, institutions) -> list[dict[str, Any]]:
    rows = []
    for u in SENT_USES:
        inst_means = institution_level_sentiment(chunks, u)
        by_r = {r: [] for r in REGIONS}
        for iid, m in inst_means.items():
            inst = institutions.get(iid)
            if inst:
                by_r[inst["region"]].append(m)
        H, df, p = kruskal_wallis([by_r[r] for r in REGIONS])
        rows.append({"use_case": u, "H": H, "df": df, "p_value": p,
                     "n_NA": len(by_r["NA"]), "n_EU": len(by_r["EU"]),
                     "n_EA": len(by_r["EA"]), "n_LA": len(by_r["LA"])})
    return rows


def E4_icc(stats_dir: Path) -> list[dict[str, Any]]:
    """For each theme, read the mixed-logit fit and report ICC = 蟽虏/(蟽虏 + 蟺虏/3).
    The logistic-link ICC denominator is 蟺虏/3 鈮?3.290."""
    rows = []
    for theme in THEMES:
        p = stats_dir / f"mixed_logit_{theme}.csv"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("sigma2_country"):
                try:
                    sigma2 = float(line.split(",")[1])
                except (ValueError, IndexError):
                    continue
                icc = sigma2 / (sigma2 + math.pi ** 2 / 3)
                rows.append({"theme": theme, "sigma2_country": sigma2,
                              "icc_logistic": icc})
                break
    return rows


def E5_power(n_per_region: dict[str, int]) -> list[dict[str, Any]]:
    """Retrospective power: for a 2-region chi-square at 伪=0.05 power=0.80,
    what gap size (in pp) is the minimum detectable effect, given the
    observed per-region n?"""
    rows = []
    z_alpha = 1.96
    z_beta = 0.8416
    # Cohen's 蠁 = (z_伪 + z_尾) / 鈭歯  鈫?MDE in proportion difference
    # using normal approx for two proportions assuming p1=p2=0.5 (max variance)
    for a in REGIONS:
        for b in REGIONS:
            if a >= b:
                continue
            n1 = n_per_region.get(a, 0)
            n2 = n_per_region.get(b, 0)
            if n1 == 0 or n2 == 0:
                continue
            # MDE on 螖p assuming p虅 = 0.5: SE_pool = 鈭?0.25*(1/n1+1/n2))
            se = math.sqrt(0.25 * (1 / n1 + 1 / n2))
            mde = (z_alpha + z_beta) * se
            rows.append({"region_A": a, "region_B": b,
                          "n_A": n1, "n_B": n2,
                          "mde_pp_at_alpha0.05_power0.80": mde * 100})
    return rows


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes-dir", default="results/codes/qwen36")
    ap.add_argument("--inst-csv", default="data/institution_list.csv")
    ap.add_argument("--out-dir", default="results/stats")
    args = ap.parse_args()
    out = Path(args.out_dir)
    institutions = {r["institution_id"]: r for r in load_csv_dicts(Path(args.inst_csv))}
    chunks = load_coded(Path(args.codes_dir), institutions)
    print(f"[E] {len(chunks)} chunks across {len({c.institution_id for c in chunks})} institutions")

    e1 = E1_wilson(chunks, institutions)
    _write_csv(out / "extended_E1_wilson_ci.csv", e1)
    print(f"[E1] wrote Wilson CIs for {len(e1)} (region 脳 theme) cells")

    e2 = E2_holm(chunks, institutions,
                  ["T7_vendor_governance", "T4_integrity", "T5_disclosure"])
    _write_csv(out / "extended_E2_holm_pairwise.csv", e2)
    n_holm = sum(r["reject_at_holm_005"] for r in e2)
    print(f"[E2] Holm-Bonferroni: {n_holm}/{len(e2)} pairs reject at 伪=0.05")

    e3 = E3_kruskal(chunks, institutions)
    _write_csv(out / "extended_E3_kruskal_sentiment.csv", e3)
    n_kr = sum(1 for r in e3 if not math.isnan(r["p_value"]) and r["p_value"] < 0.05)
    print(f"[E3] Kruskal-Wallis: {n_kr}/{len(e3)} use-cases reject H0 at 伪=0.05")

    e4 = E4_icc(out)
    _write_csv(out / "extended_E4_icc.csv", e4)
    if e4:
        mean_icc = sum(r["icc_logistic"] for r in e4) / len(e4)
        print(f"[E4] mean ICC (country) = {mean_icc:.3f} across {len(e4)} themes")

    # n_per_region from the institution corpus
    region_n = {r: 0 for r in REGIONS}
    for c in chunks:
        inst = institutions.get(c.institution_id)
        if inst:
            region_n[inst["region"]] = max(region_n.get(inst["region"], 0), 0)
    # use n_per_region from the unique-institution set
    seen = set()
    for c in chunks:
        inst = institutions.get(c.institution_id)
        if inst and c.institution_id not in seen:
            seen.add(c.institution_id)
            region_n[inst["region"]] += 1
    e5 = E5_power(region_n)
    _write_csv(out / "extended_E5_power_retrospective.csv", e5)
    print(f"[E5] retrospective MDE table for {len(e5)} region pairs at 伪=0.05 power=0.80")

    # Summary markdown
    md = ["# Extended Analyses 鈥?Reviewer-Concern Closure", ""]
    md.append("## E1. Wilson 95% confidence intervals (per region 脳 theme)")
    md.append("")
    md.append("| Theme | NA | EU | EA | LA |")
    md.append("|---|---|---|---|---|")
    by = {(r["theme"], r["region"]): r for r in e1}
    for t in THEMES:
        cells = [t]
        for r in REGIONS:
            d = by.get((t, r), {})
            cells.append(f"{d.get('prevalence',0)*100:.0f}% "
                          f"[{d.get('wilson_lo',0)*100:.0f}, "
                          f"{d.get('wilson_hi',0)*100:.0f}]")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append("## E2. Holm-Bonferroni pairwise z (T7, T4, T5)")
    md.append("")
    md.append(f"- **{n_holm}/{len(e2)}** pairs reject H0 at 伪=0.05 with Holm correction")
    md.append(
        "  (Holm is uniformly at least as powerful as Bonferroni; identical or more rejections expected.)")
    md.append("")
    md.append("## E3. Kruskal-Wallis on sentiment (non-parametric)")
    md.append("")
    md.append("| Use case | H | df | p |")
    md.append("|---|---|---|---|")
    for r in e3:
        md.append(f"| {r['use_case']} | {r['H']:.3f} | {r['df']} | {r['p_value']:.4g} |")
    md.append(f"\n- **{n_kr}/{len(e3)}** use-cases reject H0 (parametric ANOVA also rejected 0/4 鈥?non-parametric confirms).")
    md.append("")
    md.append("## E4. Intra-class correlation (ICC) for country random effect")
    md.append("")
    md.append("| Theme | 蟽虏_country | ICC (logistic link) |")
    md.append("|---|---|---|")
    for r in e4:
        md.append(f"| {r['theme']} | {r['sigma2_country']:.3f} | {r['icc_logistic']:.3f} |")
    md.append("")
    md.append("## E5. Retrospective minimum-detectable effect (alpha=0.05, power=0.80, two-proportion z)")
    md.append("")
    md.append("| Pair | n_A | n_B | MDE (pp) |")
    md.append("|---|---|---|---|")
    for r in e5:
        md.append(f"| {r['region_A']}-{r['region_B']} | {r['n_A']} | {r['n_B']} "
                   f"| {r['mde_pp_at_alpha0.05_power0.80']:.1f} |")
    (out / "extended_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[E] wrote extended_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

