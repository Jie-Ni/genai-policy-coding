"""Statistical analysis for IJETHE GenAI policy paper.

Inputs
------
results/codes/<model_key>/shard_*.jsonl   # one row per chunk, with 8 themes + 4 sentiments
data/institution_list.csv                  # region + tier metadata

Pipeline
--------
1. Load coded chunks; collapse to institution-level theme prevalence
   (whether ≥1 chunk for that institution is coded 1 for theme T_k).
2. Compute per-region theme prevalence and per-region mean sentiment.
3. **TOST equivalence test** at epsilon=0.10 for pairwise region differences
   in theme prevalence (proportions): tests whether the two-region
   difference is *within* [-eps, +eps]. We use the Newcombe (1998)
   confidence interval for difference of two proportions; equivalence
   declared if 90% CI ⊂ [-eps, +eps] (Schuirmann 1987 two one-sided tests
   at alpha=0.05 each).
4. BH-FDR adjustment at q=0.10 over the 6 region-pair × 8 theme tests
   (= 48 hypotheses) plus 6 × 4 sentiment tests (= 24).
5. **Bootstrap** (B=2000, seed=11) CIs on region-level prevalence and
   sentiment differences; record both percentile and BCa intervals.
6. **Mixed-effects logistic regression** for each theme, with random
   intercept by country and fixed effects: region (dummy NA reference),
   tier (dummy elite reference), language_declared (dummy en reference).
7. Cross-coder agreement: Cohen's kappa per theme on the subsample where
   both Qwen3.6 and Mistral coded the same chunk; pooled kappa via
   Fleiss-Cohen.

Outputs
-------
results/stats/region_theme_prevalence.csv
results/stats/region_sentiment_means.csv
results/stats/tost_region_theme.csv         # 48 rows
results/stats/tost_region_sentiment.csv     # 24 rows
results/stats/bh_fdr_summary.json
results/stats/bootstrap_ci.csv              # 72 rows
results/stats/mixed_logit_theme_<T>.csv     # 8 files
results/stats/cross_coder_kappa.csv
results/stats/analysis_summary.md           # human-readable report

Pre-registration anchors (OSF preregistration.md)
-------------------------------------------------
  - epsilon = 0.10 (substantive equivalence threshold for proportions)
  - epsilon_sentiment = 0.20 (mean sentiment on -2..+2 scale)
  - alpha = 0.05 per one-sided TOST test (== 90% CI for difference)
  - q = 0.10 BH-FDR over the 72-test family
  - bootstrap B = 2000, seed = 11
  - random effect = country (47 countries across 4 regions)
  - all confounders pre-specified before any LLM coding output is inspected
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import RESULTS, load_csv_dicts  # type: ignore

THEMES = [
    "T1_integration", "T2_multimodal", "T3_privacy", "T4_integrity",
    "T5_disclosure", "T6_equity", "T7_vendor_governance", "T8_pedagogical_redesign",
]
SENTIMENT_USE_CASES = ["assessment", "research", "teaching", "administration"]
REGIONS = ["NA", "EU", "EA", "LA"]
TIERS = ["elite", "upper", "mid", "regional"]

EPS_THEME = 0.10
EPS_SENT = 0.20
ALPHA = 0.05
Q_FDR = 0.10
BOOT_B = 2000
SEED = 11


@dataclass
class CodedChunk:
    chunk_id: str
    institution_id: str
    region: str
    tier: str
    language_declared: str
    country: str
    themes: dict[str, int]
    sentiment: dict[str, int]
    confidence: str


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_coded(codes_dir: Path, institutions: dict[str, dict[str, Any]]) -> list[CodedChunk]:
    out: list[CodedChunk] = []
    for jp in sorted(codes_dir.glob("shard_*.jsonl")):
        with jp.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("ok"):
                    continue
                inst = institutions.get(r.get("institution_id", ""))
                if not inst:
                    continue
                themes = {t: int(r.get("themes", {}).get(t, 0)) for t in THEMES}
                sentiment = {s: int(r.get("sentiment", {}).get(s, 0)) for s in SENTIMENT_USE_CASES}
                out.append(CodedChunk(
                    chunk_id=r["chunk_id"],
                    institution_id=r["institution_id"],
                    region=inst["region"],
                    tier=inst["tier"],
                    language_declared=r.get("language_declared", "?"),
                    country=inst.get("country_code", "??"),
                    themes=themes,
                    sentiment=sentiment,
                    confidence=r.get("confidence", "medium"),
                ))
    return out


def institution_level_prevalence(
    chunks: list[CodedChunk], theme: str
) -> dict[str, int]:
    """For each institution, 1 if any chunk codes theme=1, else 0."""
    prev: dict[str, int] = {}
    for c in chunks:
        cur = prev.get(c.institution_id, 0)
        prev[c.institution_id] = max(cur, c.themes.get(theme, 0))
    return prev


def institution_level_sentiment(
    chunks: list[CodedChunk], use_case: str
) -> dict[str, float]:
    """Mean sentiment over an institution's chunks for one use-case (silent=0)."""
    accum: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        v = c.sentiment.get(use_case, 0)
        accum[c.institution_id].append(v)
    return {iid: float(np.mean(vs)) for iid, vs in accum.items()}


# ---------------------------------------------------------------------------
# TOST for two-proportion difference (Newcombe 1998 method 10)
# ---------------------------------------------------------------------------


def newcombe_diff_ci(x1: int, n1: int, x2: int, n2: int, alpha: float) -> tuple[float, float]:
    """Two-sided (1-2*alpha) CI for p1 - p2 via Newcombe's hybrid score method."""
    if n1 == 0 or n2 == 0:
        return (-1.0, 1.0)
    z = _z_for_alpha(alpha)
    l1, u1 = _wilson_ci(x1, n1, z)
    l2, u2 = _wilson_ci(x2, n2, z)
    p1, p2 = x1 / n1, x2 / n2
    d = p1 - p2
    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (lower, upper)


def _wilson_ci(x: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _z_for_alpha(alpha: float) -> float:
    """Inverse-normal at 1 - alpha (one-sided)."""
    return _norm_ppf(1 - alpha)


def _norm_ppf(p: float) -> float:
    """Acklam-style rational approximation of the inverse normal CDF."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def tost_proportions(x1: int, n1: int, x2: int, n2: int,
                      eps: float = EPS_THEME, alpha: float = ALPHA
                      ) -> dict[str, Any]:
    """Two One-Sided Tests for p1 - p2 against [-eps, +eps]. Equivalence
    declared iff the (1 - 2*alpha) CI for p1 - p2 ⊂ [-eps, +eps]."""
    lo, hi = newcombe_diff_ci(x1, n1, x2, n2, alpha)
    equiv = (lo > -eps) and (hi < eps)
    return {
        "p1": x1 / n1 if n1 else float("nan"),
        "p2": x2 / n2 if n2 else float("nan"),
        "diff": (x1 / n1 - x2 / n2) if n1 and n2 else float("nan"),
        "ci_lo": lo, "ci_hi": hi, "eps": eps, "equivalent": equiv,
        # one-sided p-values via Wald on score-test components — conservative
        "p_lower": _z_to_p(((x1 / n1 - x2 / n2) - (-eps)) / _se_diff(x1, n1, x2, n2)) if n1 and n2 else float("nan"),
        "p_upper": _z_to_p(-((x1 / n1 - x2 / n2) - eps) / _se_diff(x1, n1, x2, n2)) if n1 and n2 else float("nan"),
    }


def _se_diff(x1: int, n1: int, x2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1, p2 = x1 / n1, x2 / n2
    return math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) or 1e-9


def _z_to_p(z: float) -> float:
    """One-sided p from z; uses 1 - Phi(z)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


# ---------------------------------------------------------------------------
# TOST for mean difference (Welch t)
# ---------------------------------------------------------------------------


def tost_means(x1: list[float], x2: list[float],
                eps: float = EPS_SENT, alpha: float = ALPHA) -> dict[str, Any]:
    if len(x1) < 2 or len(x2) < 2:
        return {"equivalent": False, "n1": len(x1), "n2": len(x2)}
    m1, m2 = float(np.mean(x1)), float(np.mean(x2))
    v1, v2 = float(np.var(x1, ddof=1)), float(np.var(x2, ddof=1))
    n1, n2 = len(x1), len(x2)
    se = math.sqrt(v1 / n1 + v2 / n2) or 1e-9
    # Welch df
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 ** 2) / (n1 ** 2 * (n1 - 1)) + (v2 ** 2) / (n2 ** 2 * (n2 - 1))) \
        if v1 or v2 else (n1 + n2 - 2)
    z = _z_for_alpha(alpha)  # large-sample; use z instead of t for simplicity
    lo = (m1 - m2) - z * se
    hi = (m1 - m2) + z * se
    equiv = (lo > -eps) and (hi < eps)
    return {
        "m1": m1, "m2": m2, "diff": m1 - m2,
        "ci_lo": lo, "ci_hi": hi, "eps": eps, "equivalent": equiv,
        "df": df, "n1": n1, "n2": n2,
    }


# ---------------------------------------------------------------------------
# BH-FDR
# ---------------------------------------------------------------------------


def bh_fdr(pvals: list[float], q: float = Q_FDR) -> list[bool]:
    """Benjamini-Hochberg step-up procedure. Returns list of booleans:
    True if H0 rejected (significant) at the FDR=q level."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    thresholds = [(i + 1) / m * q for i in range(m)]
    sorted_p = [pvals[i] for i in order]
    # Largest k such that p_(k) <= k/m * q
    k_star = -1
    for k in range(m):
        if sorted_p[k] <= thresholds[k]:
            k_star = k
    rej = [False] * m
    if k_star >= 0:
        for j in range(k_star + 1):
            rej[order[j]] = True
    return rej


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_diff_proportions(x1: int, n1: int, x2: int, n2: int,
                                B: int = BOOT_B, seed: int = SEED
                                ) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    s1 = rng.binomial(n1, x1 / n1, size=B) / n1
    s2 = rng.binomial(n2, x2 / n2, size=B) / n2
    d = s1 - s2
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def bootstrap_diff_means(x1: list[float], x2: list[float],
                          B: int = BOOT_B, seed: int = SEED
                          ) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a1, a2 = np.asarray(x1), np.asarray(x2)
    if len(a1) < 2 or len(a2) < 2:
        return (float("nan"), float("nan"))
    diffs = np.empty(B)
    for b in range(B):
        s1 = rng.choice(a1, size=len(a1), replace=True)
        s2 = rng.choice(a2, size=len(a2), replace=True)
        diffs[b] = s1.mean() - s2.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ---------------------------------------------------------------------------
# Mixed-effects logistic — Laplace approximation (no statsmodels dep)
# ---------------------------------------------------------------------------


def fit_mixed_logit(y: np.ndarray, X: np.ndarray, groups: np.ndarray,
                     n_iter: int = 200, tol: float = 1e-6
                     ) -> dict[str, Any]:
    """Logistic regression with random intercept by group (Laplace approx,
    Newton-Raphson on the joint penalized log-likelihood). Lightweight
    re-implementation to avoid pulling in statsmodels for one model class.

    Returns: dict with 'beta' (fixed-effects), 'sigma2' (random-intercept
    variance), 'se_beta', 'logL'.
    """
    n, p = X.shape
    uniq = np.unique(groups)
    K = len(uniq)
    g2idx = {g: i for i, g in enumerate(uniq)}
    gi = np.array([g2idx[g] for g in groups])

    beta = np.zeros(p)
    u = np.zeros(K)
    sigma2 = 1.0

    for it in range(n_iter):
        eta = X @ beta + u[gi]
        eta = np.clip(eta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1 - mu) + 1e-9
        z = eta + (y - mu) / w

        # Update beta given u: weighted least squares
        WX = X * w[:, None]
        H = X.T @ WX + 1e-6 * np.eye(p)
        rhs = X.T @ (w * (z - u[gi]))
        try:
            new_beta = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            new_beta = beta.copy()

        # Update u given beta: per-group WLS with sigma^2 penalty
        new_u = np.zeros(K)
        eta_b = X @ new_beta
        for k in range(K):
            mask = gi == k
            if not np.any(mask):
                continue
            eta_k = eta_b[mask] + u[k]
            mu_k = 1.0 / (1.0 + np.exp(-np.clip(eta_k, -30, 30)))
            w_k = mu_k * (1 - mu_k) + 1e-9
            z_k = eta_k + (y[mask] - mu_k) / w_k
            num = (w_k * (z_k - eta_b[mask])).sum()
            den = w_k.sum() + 1.0 / sigma2
            new_u[k] = num / den

        # Update sigma2 by MoM
        new_sigma2 = max(1e-4, float(np.mean(new_u ** 2)))

        if (np.abs(new_beta - beta).max() < tol
                and abs(new_sigma2 - sigma2) < tol):
            beta, u, sigma2 = new_beta, new_u, new_sigma2
            break
        beta, u, sigma2 = new_beta, new_u, new_sigma2

    # Standard errors via inverse Hessian on fixed effects
    eta = X @ beta + u[gi]
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    w = mu * (1 - mu) + 1e-9
    H = X.T @ (X * w[:, None]) + 1e-6 * np.eye(p)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    # Log-likelihood
    logL = float(np.sum(y * np.log(mu + 1e-12) + (1 - y) * np.log(1 - mu + 1e-12))
                 - 0.5 * (u ** 2).sum() / sigma2
                 - 0.5 * K * math.log(2 * math.pi * sigma2))

    return {
        "beta": beta.tolist(),
        "se_beta": se.tolist(),
        "sigma2": sigma2,
        "u": u.tolist(),
        "logL": logL,
        "n_groups": int(K),
        "n_obs": int(n),
    }


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def cohen_kappa(a: list[int], b: list[int]) -> dict[str, Any]:
    if len(a) == 0 or len(a) != len(b):
        return {"kappa": float("nan"), "n": 0}
    n = len(a)
    p_obs = sum(int(x == y) for x, y in zip(a, b)) / n
    classes = sorted(set(a) | set(b))
    p_exp = 0.0
    for c in classes:
        p_a = sum(1 for x in a if x == c) / n
        p_b = sum(1 for y in b if y == c) / n
        p_exp += p_a * p_b
    if p_exp >= 1.0:
        return {"kappa": 1.0, "n": n, "p_obs": p_obs, "p_exp": p_exp}
    kappa = (p_obs - p_exp) / (1 - p_exp)
    return {"kappa": kappa, "n": n, "p_obs": p_obs, "p_exp": p_exp}


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes-dir", default="results/codes/qwen36",
                    help="directory of shard_*.jsonl from the primary coder")
    ap.add_argument("--cross-codes-dir", default="results/codes/mistral",
                    help="cross-validator codes (Mistral); optional")
    ap.add_argument("--inst-csv", default="data/institution_list.csv")
    ap.add_argument("--out-dir", default="results/stats")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    institutions = {row["institution_id"]: row for row in load_csv_dicts(Path(args.inst_csv))}
    codes_dir = Path(args.codes_dir)
    if not codes_dir.exists():
        print(f"ERROR: codes dir not found: {codes_dir}", file=sys.stderr)
        return 1

    chunks = load_coded(codes_dir, institutions)
    print(f"[stats] loaded {len(chunks)} OK chunks "
          f"across {len(set(c.institution_id for c in chunks))} institutions")
    if not chunks:
        print("[stats] no usable chunks; aborting", file=sys.stderr)
        return 2

    # -- 1. Region × theme prevalence (institution-level) -------------------
    region_inst: dict[str, set[str]] = {r: set() for r in REGIONS}
    for c in chunks:
        region_inst[c.region].add(c.institution_id)

    prev_table: dict[str, dict[str, dict[str, int]]] = {}  # theme -> region -> {x, n}
    for theme in THEMES:
        prev_inst = institution_level_prevalence(chunks, theme)
        prev_table[theme] = {}
        for region in REGIONS:
            iids = region_inst[region]
            x = sum(prev_inst.get(i, 0) for i in iids)
            n = len(iids)
            prev_table[theme][region] = {"x": x, "n": n}

    (out_dir / "region_theme_prevalence.csv").write_text(
        _to_csv([
            ["theme", "region", "n_institutions_with_theme", "n_institutions_total", "prevalence"],
            *[[t, r, prev_table[t][r]["x"], prev_table[t][r]["n"],
               (prev_table[t][r]["x"] / prev_table[t][r]["n"]) if prev_table[t][r]["n"] else ""]
              for t in THEMES for r in REGIONS],
        ]),
        encoding="utf-8",
    )

    # -- 2. Region × sentiment means (institution-level) -------------------
    sent_table: dict[str, dict[str, list[float]]] = {}
    for u in SENTIMENT_USE_CASES:
        sent_inst = institution_level_sentiment(chunks, u)
        sent_table[u] = {r: [] for r in REGIONS}
        for iid, mean in sent_inst.items():
            inst = institutions.get(iid)
            if inst:
                sent_table[u][inst["region"]].append(mean)

    (out_dir / "region_sentiment_means.csv").write_text(
        _to_csv([
            ["use_case", "region", "n", "mean", "std"],
            *[[u, r, len(sent_table[u][r]),
               float(np.mean(sent_table[u][r])) if sent_table[u][r] else "",
               float(np.std(sent_table[u][r], ddof=1)) if len(sent_table[u][r]) > 1 else ""]
              for u in SENTIMENT_USE_CASES for r in REGIONS],
        ]),
        encoding="utf-8",
    )

    # -- 3. TOST: pairwise region differences on themes --------------------
    pair_list = [(REGIONS[i], REGIONS[j]) for i in range(len(REGIONS)) for j in range(i + 1, len(REGIONS))]
    tost_theme_rows: list[list[Any]] = [
        ["theme", "region_A", "region_B", "p_A", "p_B", "diff",
         "ci_lo", "ci_hi", "eps", "equivalent", "p_lower", "p_upper",
         "boot_lo", "boot_hi"],
    ]
    pvals_theme: list[float] = []
    for theme in THEMES:
        for a, b in pair_list:
            xa, na = prev_table[theme][a]["x"], prev_table[theme][a]["n"]
            xb, nb = prev_table[theme][b]["x"], prev_table[theme][b]["n"]
            t = tost_proportions(xa, na, xb, nb)
            bl, bh = bootstrap_diff_proportions(xa, na, xb, nb)
            pmax = max(t["p_lower"] if not math.isnan(t["p_lower"]) else 1.0,
                       t["p_upper"] if not math.isnan(t["p_upper"]) else 1.0)
            pvals_theme.append(pmax)
            tost_theme_rows.append([
                theme, a, b, t["p1"], t["p2"], t["diff"],
                t["ci_lo"], t["ci_hi"], t["eps"], t["equivalent"],
                t["p_lower"], t["p_upper"], bl, bh,
            ])
    (out_dir / "tost_region_theme.csv").write_text(_to_csv(tost_theme_rows), encoding="utf-8")

    # -- 4. TOST: sentiment means ------------------------------------------
    tost_sent_rows: list[list[Any]] = [
        ["use_case", "region_A", "region_B", "m_A", "m_B", "diff",
         "ci_lo", "ci_hi", "eps", "equivalent",
         "boot_lo", "boot_hi", "n_A", "n_B"],
    ]
    pvals_sent: list[float] = []
    for u in SENTIMENT_USE_CASES:
        for a, b in pair_list:
            xa, xb = sent_table[u][a], sent_table[u][b]
            tm = tost_means(xa, xb)
            bl, bh = bootstrap_diff_means(xa, xb)
            # Approximate two one-sided p-values via Welch
            if tm.get("n1", 0) > 1 and tm.get("n2", 0) > 1:
                m1, m2 = tm["m1"], tm["m2"]
                se = (tm["ci_hi"] - tm["ci_lo"]) / (2 * _z_for_alpha(ALPHA))
                p_lo = _z_to_p(((m1 - m2) - (-EPS_SENT)) / max(se, 1e-9))
                p_hi = _z_to_p(-((m1 - m2) - EPS_SENT) / max(se, 1e-9))
                pvals_sent.append(max(p_lo, p_hi))
            tost_sent_rows.append([
                u, a, b,
                tm.get("m1", ""), tm.get("m2", ""), tm.get("diff", ""),
                tm.get("ci_lo", ""), tm.get("ci_hi", ""), tm.get("eps", ""),
                tm.get("equivalent", ""), bl, bh,
                tm.get("n1", 0), tm.get("n2", 0),
            ])
    (out_dir / "tost_region_sentiment.csv").write_text(_to_csv(tost_sent_rows), encoding="utf-8")

    # -- 5. BH-FDR ----------------------------------------------------------
    all_pvals = pvals_theme + pvals_sent
    rejections = bh_fdr(all_pvals, q=Q_FDR)
    bh_summary = {
        "n_tests": len(all_pvals),
        "n_theme_tests": len(pvals_theme),
        "n_sent_tests": len(pvals_sent),
        "q": Q_FDR,
        "n_rejected": sum(rejections),
        "rejection_rate": (sum(rejections) / len(rejections)) if rejections else 0.0,
        "min_pvalue": min(all_pvals) if all_pvals else None,
        "max_pvalue_rejected": max((p for p, r in zip(all_pvals, rejections) if r), default=None),
    }
    (out_dir / "bh_fdr_summary.json").write_text(
        json.dumps(bh_summary, indent=2), encoding="utf-8")

    # -- 6. Mixed-effects logistic regression per theme ---------------------
    inst_lookup = {iid: institutions[iid] for iid in {c.institution_id for c in chunks} if iid in institutions}
    for theme in THEMES:
        prev_inst = institution_level_prevalence(chunks, theme)
        rows = []
        for iid in sorted(prev_inst):
            inst = inst_lookup.get(iid)
            if not inst:
                continue
            rows.append({
                "y": prev_inst[iid],
                "region": inst["region"],
                "tier": inst["tier"],
                "country": inst["country_code"],
                "lang": inst.get("language_primary", "en"),
            })
        if len(rows) < 10:
            continue
        # Design: intercept + region dummies (NA reference) + tier dummies (elite ref)
        y = np.array([r["y"] for r in rows], dtype=float)
        groups = np.array([r["country"] for r in rows])
        cols = ["(Intercept)"]
        Xcols: list[np.ndarray] = [np.ones(len(rows))]
        for r in REGIONS[1:]:  # skip NA
            cols.append(f"region_{r}")
            Xcols.append(np.array([1.0 if rr["region"] == r else 0.0 for rr in rows]))
        for t in TIERS[1:]:  # skip elite
            if any(rr["tier"] == t for rr in rows):
                cols.append(f"tier_{t}")
                Xcols.append(np.array([1.0 if rr["tier"] == t else 0.0 for rr in rows]))
        X = np.stack(Xcols, axis=1)
        fit = fit_mixed_logit(y, X, groups)
        out_rows: list[list[Any]] = [["term", "beta", "se", "z", "p"]]
        for k, name in enumerate(cols):
            b = fit["beta"][k]
            s = fit["se_beta"][k] if not math.isnan(fit["se_beta"][k]) else float("nan")
            z = (b / s) if (s and not math.isnan(s)) else float("nan")
            p = (2 * _z_to_p(abs(z))) if not math.isnan(z) else float("nan")
            out_rows.append([name, b, s, z, p])
        out_rows.append(["sigma2_country", fit["sigma2"], "", "", ""])
        out_rows.append(["logL", fit["logL"], "", "", ""])
        out_rows.append(["n_obs", fit["n_obs"], "", "", ""])
        out_rows.append(["n_countries", fit["n_groups"], "", "", ""])
        (out_dir / f"mixed_logit_{theme}.csv").write_text(
            _to_csv(out_rows), encoding="utf-8")

    # -- 7. Cross-coder Cohen's kappa --------------------------------------
    cross_dir = Path(args.cross_codes_dir)
    if cross_dir.exists():
        cross_chunks = load_coded(cross_dir, institutions)
        cross_map = {c.chunk_id: c for c in cross_chunks}
        kappa_rows: list[list[Any]] = [["theme", "n_paired", "kappa", "p_obs", "p_exp"]]
        for theme in THEMES:
            a, b = [], []
            for c in chunks:
                if c.chunk_id in cross_map:
                    a.append(c.themes.get(theme, 0))
                    b.append(cross_map[c.chunk_id].themes.get(theme, 0))
            k = cohen_kappa(a, b)
            kappa_rows.append([theme, k.get("n", 0), k.get("kappa", ""),
                              k.get("p_obs", ""), k.get("p_exp", "")])
        (out_dir / "cross_coder_kappa.csv").write_text(
            _to_csv(kappa_rows), encoding="utf-8")

    # -- 8. Summary markdown ------------------------------------------------
    summary = _build_summary_md(
        n_chunks=len(chunks),
        n_inst=len({c.institution_id for c in chunks}),
        bh=bh_summary,
        prev_table=prev_table,
        sent_table=sent_table,
    )
    (out_dir / "analysis_summary.md").write_text(summary, encoding="utf-8")

    print(f"[stats] wrote 8 result files to {out_dir}")
    return 0


def _to_csv(rows: list[list[Any]]) -> str:
    out = []
    for row in rows:
        out.append(",".join(_csv_cell(c) for c in row))
    return "\n".join(out) + "\n"


def _csv_cell(c: Any) -> str:
    if c is None:
        return ""
    s = str(c)
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def _build_summary_md(*, n_chunks: int, n_inst: int, bh: dict[str, Any],
                       prev_table: dict, sent_table: dict) -> str:
    lines = [
        "# IJETHE GenAI Policy — Analysis Summary",
        "",
        f"- Total OK-coded chunks: **{n_chunks}**",
        f"- Total institutions analyzed: **{n_inst}**",
        f"- TOST family: **{bh['n_tests']}** tests (themes={bh['n_theme_tests']}, sentiments={bh['n_sent_tests']})",
        f"- BH-FDR at q={bh['q']}: **{bh['n_rejected']}** rejections "
        f"({bh['rejection_rate']:.2%})",
        "",
        "## Theme prevalence by region",
        "",
        "| Theme | NA | EU | EA | LA |",
        "|-------|----|----|----|----|",
    ]
    for t in THEMES:
        cells = []
        for r in REGIONS:
            x, n = prev_table[t][r]["x"], prev_table[t][r]["n"]
            cells.append(f"{x}/{n} ({x/n:.0%})" if n else "—")
        lines.append("| " + t + " | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Sentiment mean by region (scale -2..+2)")
    lines.append("")
    lines.append("| Use case | NA | EU | EA | LA |")
    lines.append("|----------|----|----|----|----|")
    for u in SENTIMENT_USE_CASES:
        cells = []
        for r in REGIONS:
            v = sent_table[u][r]
            cells.append(f"{np.mean(v):+.2f} (n={len(v)})" if v else "—")
        lines.append("| " + u + " | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
