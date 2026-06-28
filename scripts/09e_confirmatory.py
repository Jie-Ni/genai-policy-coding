"""Confirmatory inferential tests with adequate statistical power.

Replaces (or supplements) the TOST framework, which is fundamentally
underpowered at n=13-20 per region with 蔚=0.10 (Schuirmann 1987 + BH-FDR
correction).

The TOST family answered the wrong question ("are the regions equivalent
within 卤10pp?"). The substantive question is "are the regions different?"
which is reachable via:

C1. Chi-square test of independence on the 4-region 脳 8-theme contingency
    table. H0: theme prevalence is independent of region. Cramer's V as
    effect size.

C2. Per-theme Wald tests on region fixed-effect coefficients in the
    mixed-effects logistic regression already fit by 09. Reports 尾, SE,
    z, p for each region (NA reference) per theme.

C3. Cochran-Armitage trend test for the N-S gradient on each theme. With
    regions ordered NA > EU > EA > LA, tests whether theme prevalence
    monotonically decreases (or increases). Powerful when the gradient
    is real.

C4. Pairwise two-proportion z-tests for the headline finding (T7 vendor
    governance and T4 integrity), with Bonferroni correction for
    6 pairwise comparisons (伪/6 = 0.0083).

C5. ANOVA-style F-test for regional mean differences in 4 sentiment
    use-cases. Reports F, df, p per use-case.

Outputs
-------
results/stats/confirmatory_C1_chi2.csv
results/stats/confirmatory_C2_mixed_logit_region.csv
results/stats/confirmatory_C3_cochran_armitage.csv
results/stats/confirmatory_C4_pairwise_z.csv
results/stats/confirmatory_C5_sentiment_anova.csv
results/stats/confirmatory_summary.md
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
institution_level_prevalence = _stats.institution_level_prevalence
institution_level_sentiment = _stats.institution_level_sentiment
THEMES = _stats.THEMES
REGIONS = _stats.REGIONS
SENT_USES = _stats.SENTIMENT_USE_CASES
SEED = _stats.SEED


# ---------------------------------------------------------------------------
# C1. Chi-square 4-region 脳 8-theme contingency
# ---------------------------------------------------------------------------


def chi2_region_theme(chunks, institutions) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Build observed: rows=region, cols=theme; cell = # institutions in region with theme=1
    region_inst: dict[str, set[str]] = {r: set() for r in REGIONS}
    for c in chunks:
        inst = institutions.get(c.institution_id)
        if inst:
            region_inst[inst["region"]].add(c.institution_id)
    rows: list[dict[str, Any]] = []
    obs = np.zeros((len(REGIONS), len(THEMES)), dtype=float)
    region_totals = np.array([len(region_inst[r]) for r in REGIONS], dtype=float)
    for ti, t in enumerate(THEMES):
        prev_map = institution_level_prevalence(chunks, t)
        for ri, r in enumerate(REGIONS):
            obs[ri, ti] = sum(prev_map.get(i, 0) for i in region_inst[r])
            rows.append({"region": r, "theme": t,
                          "n_with": int(obs[ri, ti]),
                          "n_total": int(region_totals[ri]),
                          "prevalence": float(obs[ri, ti] / max(region_totals[ri], 1))})
    # Expected under independence treating each (region, theme) as Bernoulli with
    # marginal pooled probability per theme (pooled across regions):
    theme_p = obs.sum(axis=0) / region_totals.sum()  # pooled prevalence per theme
    chi2 = 0.0
    df = (len(REGIONS) - 1) * len(THEMES)
    for ti in range(len(THEMES)):
        for ri in range(len(REGIONS)):
            n_i = region_totals[ri]
            p_t = theme_p[ti]
            # Expected count of theme=1 in region ri: n_i * p_t
            e_yes = n_i * p_t
            e_no = n_i * (1 - p_t)
            o_yes = obs[ri, ti]
            o_no = n_i - o_yes
            if e_yes > 0:
                chi2 += (o_yes - e_yes) ** 2 / e_yes
            if e_no > 0:
                chi2 += (o_no - e_no) ** 2 / e_no
    # df = (rows-1)*(cols-1) for a true contingency; but we collapse each theme
    # into its own 2-row table with shared region structure, so we sum 8
    # independent 2x4 tables, each with df=(2-1)*(4-1)=3 鈫?total df = 24.
    df = 3 * len(THEMES)
    pval = chi2_sf(chi2, df)
    n = int(region_totals.sum())
    # Cramer's V (collapsed): sqrt(chi2 / (n * (min_dim - 1))); here min_dim
    # across an 8-table family is heuristic 鈥?report normalized by n*8 instead
    cramers_v = math.sqrt(chi2 / (n * 8)) if n > 0 else float("nan")
    head = {"chi2": chi2, "df": df, "p_value": pval, "cramers_v": cramers_v,
            "n_institutions": n}
    return head, rows


def chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) of chi-square(df) at x. Series via gammainc."""
    if x <= 0:
        return 1.0
    # Regularized upper incomplete gamma Q(k/2, x/2) for integer or half-integer df
    return _gammaincc(df / 2.0, x / 2.0)


def _gammaln(x: float) -> float:
    # Stirling-Lanczos approximation
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
    """Regularized upper incomplete gamma Q(a, x). Continued fraction (Press et al.)."""
    if x < a + 1.0:
        return 1.0 - _gammainc_series(a, x)
    # Continued fraction expansion
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
    """Regularized lower incomplete gamma P(a, x) via series expansion."""
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


# ---------------------------------------------------------------------------
# C2. Mixed-logit region p-values (already computed in 09)
# ---------------------------------------------------------------------------


def parse_mixed_logit_csvs(stats_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for theme in THEMES:
        p = stats_dir / f"mixed_logit_{theme}.csv"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines()[1:]:
            cells = line.split(",")
            if len(cells) < 5:
                continue
            term, beta, se, z, pv = cells[:5]
            if not term.startswith("region_"):
                continue
            try:
                rows.append({"theme": theme, "term": term,
                              "beta": float(beta),
                              "se": float(se) if se else float("nan"),
                              "z": float(z) if z else float("nan"),
                              "p_two_sided": float(pv) if pv else float("nan")})
            except ValueError:
                continue
    return rows


# ---------------------------------------------------------------------------
# C3. Cochran-Armitage trend test (per theme, with N-S ranks)
# ---------------------------------------------------------------------------


def cochran_armitage(chunks, institutions) -> list[dict[str, Any]]:
    # Region scores: NA=4, EU=3, EA=2, LA=1 (ordering by "policy maturity" descriptive)
    region_score = {"NA": 4, "EU": 3, "EA": 2, "LA": 1}
    region_inst: dict[str, list[str]] = {r: [] for r in REGIONS}
    for c in chunks:
        inst = institutions.get(c.institution_id)
        if inst and c.institution_id not in region_inst[inst["region"]]:
            region_inst[inst["region"]].append(c.institution_id)
    rows: list[dict[str, Any]] = []
    for theme in THEMES:
        prev = institution_level_prevalence(chunks, theme)
        ys, scores = [], []
        for r in REGIONS:
            for i in region_inst[r]:
                ys.append(prev.get(i, 0))
                scores.append(region_score[r])
        ys = np.array(ys, dtype=float)
        s = np.array(scores, dtype=float)
        n = len(ys)
        if n == 0 or ys.sum() in (0, n):
            rows.append({"theme": theme, "Z": float("nan"), "p_two_sided": float("nan")})
            continue
        p_hat = ys.mean()
        num = (ys * s).sum() - n * p_hat * s.mean()
        var = p_hat * (1 - p_hat) * ((s ** 2).sum() - n * (s.mean() ** 2))
        z = num / math.sqrt(var) if var > 0 else float("nan")
        p_two = 2 * (1 - _norm_cdf(abs(z))) if not math.isnan(z) else float("nan")
        rows.append({"theme": theme, "Z": z, "p_two_sided": p_two})
    return rows


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# C4. Pairwise two-proportion z-test for T7 and T4
# ---------------------------------------------------------------------------


def pairwise_z_proportions(chunks, institutions, themes: list[str]
                            ) -> list[dict[str, Any]]:
    region_inst: dict[str, list[str]] = {r: [] for r in REGIONS}
    for c in chunks:
        inst = institutions.get(c.institution_id)
        if inst and c.institution_id not in region_inst[inst["region"]]:
            region_inst[inst["region"]].append(c.institution_id)
    pairs = [(REGIONS[i], REGIONS[j])
             for i in range(len(REGIONS)) for j in range(i + 1, len(REGIONS))]
    rows: list[dict[str, Any]] = []
    n_pairs = len(pairs)
    alpha_bonf = 0.05 / n_pairs  # 0.0083 for 6 pairs
    for theme in themes:
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
            p_two = 2 * (1 - _norm_cdf(abs(z))) if not math.isnan(z) else float("nan")
            rows.append({"theme": theme, "region_A": a, "region_B": b,
                          "p_A": p1, "p_B": p2, "diff": p1 - p2,
                          "z": z, "p_two_sided": p_two,
                          "bonferroni_alpha": alpha_bonf,
                          "reject_at_bonferroni": int(p_two < alpha_bonf if not math.isnan(p_two) else 0)})
    return rows


# ---------------------------------------------------------------------------
# C5. ANOVA F-test for sentiment means
# ---------------------------------------------------------------------------


def anova_sentiment(chunks, institutions) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for u in SENT_USES:
        inst_means = institution_level_sentiment(chunks, u)
        by_r: dict[str, list[float]] = {r: [] for r in REGIONS}
        for iid, m in inst_means.items():
            inst = institutions.get(iid)
            if inst:
                by_r[inst["region"]].append(m)
        groups = [v for v in by_r.values() if len(v) > 1]
        if len(groups) < 2:
            continue
        # One-way ANOVA
        n = sum(len(g) for g in groups)
        k = len(groups)
        grand = sum(sum(g) for g in groups) / n
        ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
        ss_within = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)
        df_b = k - 1
        df_w = n - k
        ms_b = ss_between / df_b
        ms_w = ss_within / df_w if df_w > 0 else float("nan")
        F = ms_b / ms_w if ms_w and not math.isnan(ms_w) and ms_w > 0 else float("nan")
        p = _f_sf(F, df_b, df_w) if not math.isnan(F) else float("nan")
        rows.append({"use_case": u, "F": F, "df_between": df_b,
                      "df_within": df_w, "p_value": p})
    return rows


def _f_sf(F: float, df1: int, df2: int) -> float:
    """Survival function 1 - CDF of F(df1, df2). Uses regularized inc. beta."""
    if F <= 0:
        return 1.0
    x = df2 / (df2 + df1 * F)
    # 1 - I_x(df2/2, df1/2)
    return _betainc(df2 / 2.0, df1 / 2.0, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via continued fraction (Press et al.)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_beta = _gammaln(a) + _gammaln(b) - _gammaln(a + b)
    bt = math.exp(a * math.log(x) + b * math.log(1 - x) - ln_beta)
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    fpmin = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes-dir", default="results/codes/qwen36")
    ap.add_argument("--inst-csv", default="data/institution_list.csv")
    ap.add_argument("--out-dir", default="results/stats")
    args = ap.parse_args()
    out = Path(args.out_dir)
    institutions = {r["institution_id"]: r for r in load_csv_dicts(Path(args.inst_csv))}
    chunks = _stats.load_coded(Path(args.codes_dir), institutions)
    print(f"[C] {len(chunks)} chunks across "
          f"{len({c.institution_id for c in chunks})} institutions")

    # C1
    head, c1 = chi2_region_theme(chunks, institutions)
    _write_csv(out / "confirmatory_C1_chi2.csv", c1)
    (out / "confirmatory_C1_chi2_overall.csv").write_text(
        "chi2,df,p_value,cramers_v,n_institutions\n"
        f"{head['chi2']:.4f},{head['df']},{head['p_value']:.4g},"
        f"{head['cramers_v']:.4f},{head['n_institutions']}\n",
        encoding="utf-8")
    print(f"[C1] chi2={head['chi2']:.2f}, df={head['df']}, "
          f"p={head['p_value']:.4g}, V={head['cramers_v']:.3f}")

    # C2
    c2 = parse_mixed_logit_csvs(out)
    _write_csv(out / "confirmatory_C2_mixed_logit_region.csv", c2)
    n_sig = sum(1 for r in c2 if r["p_two_sided"] is not None and r["p_two_sided"] < 0.05)
    print(f"[C2] mixed-logit region effects: {n_sig}/{len(c2)} significant at 伪=0.05")

    # C3
    c3 = cochran_armitage(chunks, institutions)
    _write_csv(out / "confirmatory_C3_cochran_armitage.csv", c3)
    n_sig_c3 = sum(1 for r in c3 if r["p_two_sided"] is not None
                    and not math.isnan(r["p_two_sided"]) and r["p_two_sided"] < 0.05)
    print(f"[C3] Cochran-Armitage trend: {n_sig_c3}/{len(c3)} themes p<0.05")

    # C4
    c4 = pairwise_z_proportions(chunks, institutions,
                                  themes=["T7_vendor_governance",
                                          "T4_integrity",
                                          "T5_disclosure"])
    _write_csv(out / "confirmatory_C4_pairwise_z.csv", c4)
    n_bonf_sig = sum(r["reject_at_bonferroni"] for r in c4)
    print(f"[C4] pairwise z-tests (T7+T4+T5): {n_bonf_sig}/{len(c4)} reject at Bonferroni")

    # C5
    c5 = anova_sentiment(chunks, institutions)
    _write_csv(out / "confirmatory_C5_sentiment_anova.csv", c5)
    n_sig_c5 = sum(1 for r in c5 if r["p_value"] is not None
                    and not math.isnan(r["p_value"]) and r["p_value"] < 0.05)
    print(f"[C5] sentiment ANOVA: {n_sig_c5}/{len(c5)} use-cases p<0.05")

    # Summary
    lines = ["# Confirmatory Tests 鈥?Are Regions Different?", ""]
    lines.append(f"## C1. Chi-square: region 脳 theme")
    lines.append(f"- 蠂虏({head['df']}) = **{head['chi2']:.2f}**, "
                 f"**p = {head['p_value']:.4g}**, Cramer's V = {head['cramers_v']:.3f}")
    lines.append("")
    lines.append(f"## C2. Mixed-logit region coefficients (NA reference)")
    lines.append(f"- **{n_sig}/{len(c2)}** region coefficients significant at 伪=0.05")
    lines.append("")
    lines.append("| Theme | Term | 尾 | SE | z | p |")
    lines.append("|-------|------|----|----|----|----|")
    for r in c2:
        lines.append(f"| {r['theme']} | {r['term']} | {r['beta']:+.3f} | "
                      f"{r['se']:.3f} | {r['z']:+.2f} | {r['p_two_sided']:.4g} |")
    lines.append("")
    lines.append(f"## C3. Cochran-Armitage N-S trend")
    lines.append(f"- **{n_sig_c3}/{len(c3)}** themes show significant N-S trend (p<0.05)")
    lines.append("")
    lines.append("| Theme | Z | p (two-sided) |")
    lines.append("|-------|---|---------------|")
    for r in c3:
        lines.append(f"| {r['theme']} | {r['Z']:+.3f} | {r['p_two_sided']:.4g} |")
    lines.append("")
    lines.append(f"## C4. Pairwise z-tests for T7/T4/T5 (Bonferroni 伪/6=0.0083)")
    lines.append(f"- **{n_bonf_sig}/{len(c4)}** pairs reject at Bonferroni-corrected level")
    lines.append("")
    lines.append("| Theme | Pair | diff | z | p | Reject? |")
    lines.append("|-------|------|------|----|----|---------|")
    for r in c4:
        lines.append(f"| {r['theme']} | {r['region_A']}-{r['region_B']} | "
                      f"{r['diff']:+.2f} | {r['z']:+.2f} | {r['p_two_sided']:.4g} | "
                      f"{'鉁? if r['reject_at_bonferroni'] else '路'} |")
    lines.append("")
    lines.append(f"## C5. ANOVA on sentiment means (per use-case)")
    lines.append(f"- **{n_sig_c5}/{len(c5)}** use-cases show significant regional differences")
    lines.append("")
    lines.append("| Use-case | F | df | p |")
    lines.append("|----------|---|----|----|")
    for r in c5:
        lines.append(f"| {r['use_case']} | {r['F']:.2f} | "
                      f"({r['df_between']}, {r['df_within']}) | {r['p_value']:.4g} |")
    (out / "confirmatory_summary.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
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


if __name__ == "__main__":
    sys.exit(main())

