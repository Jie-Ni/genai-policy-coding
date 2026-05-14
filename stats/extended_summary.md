# Extended Analyses — Reviewer-Concern Closure

## E1. Wilson 95% confidence intervals (per region × theme)

| Theme | NA | EU | EA | LA |
|---|---|---|---|---|
| T1_integration | 95% [76, 99] | 89% [67, 97] | 77% [50, 92] | 77% [50, 92] |
| T2_multimodal | 75% [53, 89] | 67% [44, 84] | 69% [42, 87] | 46% [23, 71] |
| T3_privacy | 85% [64, 95] | 67% [44, 84] | 69% [42, 87] | 38% [18, 64] |
| T4_integrity | 85% [64, 95] | 67% [44, 84] | 46% [23, 71] | 92% [67, 99] |
| T5_disclosure | 90% [70, 97] | 61% [39, 80] | 38% [18, 64] | 46% [23, 71] |
| T6_equity | 55% [34, 74] | 28% [12, 51] | 23% [8, 50] | 38% [18, 64] |
| T7_vendor_governance | 80% [58, 92] | 67% [44, 84] | 31% [13, 58] | 15% [4, 42] |
| T8_pedagogical_redesign | 70% [48, 85] | 33% [16, 56] | 62% [36, 82] | 38% [18, 64] |

## E2. Holm-Bonferroni pairwise z (T7, T4, T5)

- **2/18** pairs reject H0 at α=0.05 with Holm correction
  (Holm is uniformly at least as powerful as Bonferroni; identical or more rejections expected.)

## E3. Kruskal-Wallis on sentiment (non-parametric)

| Use case | H | df | p |
|---|---|---|---|
| assessment | 0.801 | 3 | 0.8492 |
| research | 0.177 | 3 | 0.9812 |
| teaching | 1.375 | 3 | 0.7113 |
| administration | 7.764 | 3 | 0.05115 |

- **0/4** use-cases reject H0 (parametric ANOVA also rejected 0/4 — non-parametric confirms).

## E4. Intra-class correlation (ICC) for country random effect

| Theme | σ²_country | ICC (logistic link) |
|---|---|---|
| T1_integration | 0.000 | 0.000 |
| T2_multimodal | 0.000 | 0.000 |
| T3_privacy | 0.000 | 0.000 |
| T4_integrity | 0.000 | 0.000 |
| T5_disclosure | 0.000 | 0.000 |
| T6_equity | 0.000 | 0.000 |
| T7_vendor_governance | 0.000 | 0.000 |
| T8_pedagogical_redesign | 0.000 | 0.000 |

## E5. Retrospective minimum-detectable effect (α=0.05, power=0.80, two-proportion z)

| Pair | n_A | n_B | MDE (pp) |
|---|---|---|---|
| EU–NA | 18 | 20 | 45.5 |
| EU–LA | 18 | 13 | 51.0 |
| EA–NA | 13 | 20 | 49.9 |
| EA–EU | 13 | 18 | 51.0 |
| EA–LA | 13 | 13 | 54.9 |
| LA–NA | 13 | 20 | 49.9 |
