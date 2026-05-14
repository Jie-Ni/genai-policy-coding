# OSF Pre-registration — IJETHE GenAI Policy Paper

**Project title.** Isomorphism or institutional logic divergence? A multilingual NLP analysis of GenAI policies across 240 universities, with equivalence testing under multiple-comparison control.

**Authors.**
- Ni Jie (Principal Investigator). University of Innsbruck, Digital Science Center. ORCID: [to be added before submission to OSF].
- Adam Jatowt (Senior Supervisor). University of Innsbruck, Digital Science Center. ORCID: [to be added].

**Date of pre-registration.** [To be filled by OSF on submission. Pre-registration is timestamped before any inferential statistic is computed; target date end of operations-manual week 4, no later than 2026-06-11.]

**Target journal.** International Journal of Educational Technology in Higher Education (IJETHE), Springer Open. Regular issue (continuous publication), no fixed deadline.

**OSF submission category.** Pre-registration of original analysis (not a registered report). Closest fit is the OSF AsPredicted template, adapted for content-analysis methodology.

---

## 1. Hypotheses

This study is guided by three competing theoretical hypotheses about cross-national university GenAI policy variation:

- **H1 (Strong isomorphism).** Following DiMaggio and Powell (1983), coercive, mimetic, and normative pressures produce convergent institutional policies across regions. *Operationalisation:* at least 60% of policy themes are TOST-equivalent across the six region-pair comparisons at $\varepsilon = 0.10$, after Benjamini–Hochberg false-discovery-rate control at $q = 0.10$.

- **H2 (Strong institutional-logic divergence).** Following Thornton, Ocasio, and Lounsbury (2012), institutions follow nationally inherited rationalities (state, market, profession, community); their policy responses reflect dominant national logics rather than transnational pressure. *Operationalisation:* at least 60% of policy themes are non-equivalent across the six region-pair comparisons, and the non-equivalent themes cluster on regulatory and sovereignty axes (themes T3, T7).

- **H3 (Mixed regime — PRIMARY pre-registered hypothesis).** Convergence and divergence operate on different policy dimensions. *Operationalisation:* (a) integrity-related themes T4 (academic integrity) and T5 (disclosure) pass TOST in at least 75% of inter-regional comparisons; (b) integration/pedagogy-related themes T1 (integration) and T8 (pedagogical redesign) fail TOST in at least 75%.

H3 is pre-registered as the **primary** hypothesis. H1 and H2 are pre-registered as bounding alternatives so that the field of falsifiable outcomes is documented in advance.

A fourth, exploratory hypothesis (H4) is pre-registered without pass/fail criteria: **sentiment toward GenAI use varies systematically by use-case (assessment / research / teaching / administration)**, with more restrictive sentiment on assessment and more permissive sentiment on research and administration.

---

## 2. Sample

**Target sample size.** 240 universities, stratified across four regions × four tiers.

| Region | Code | Countries | n | Languages |
|---|---|---|---|---|
| North America | NA | USA, Canada | 60 | English (+ Quebec French) |
| Europe | EU | EU27 + UK + Switzerland + Norway | 60 | English, German, French, Spanish, Italian, Dutch |
| East Asia | EA | China, Japan, Korea, Singapore, Hong Kong, Taiwan | 60 | Chinese (simplified + traditional), Japanese, Korean, English |
| Latin America | LA | Brazil, Mexico, Argentina, Chile, Colombia, Peru | 60 | Portuguese, Spanish |

| Tier | Definition | n per region | Total |
|---|---|---|---|
| Elite | QS World 2026 top-50 globally | 15 | 60 |
| Upper | QS 51–200 globally | 20 | 80 |
| Mid | QS 201–500 globally | 15 | 60 |
| Regional | Highest-ranked regional accreditor lists | 10 | 40 |

**Acceptable yield floor.** ≥ 192 / 240 institutions (80%) must yield at least one publicly accessible GenAI policy document for the planned analysis to proceed. If yield falls below 192, the paper will report the actual N achieved and explicitly discuss geographic-availability bias.

**Replacement protocol.** If a sampled institution lacks a discoverable public GenAI policy after the documented search protocol, replace it with the next-ranked institution within the same tier × region cell. Every replacement is logged in `data/exclusion_ledger.csv`.

---

## 3. Inclusion / exclusion criteria

**Document inclusion criteria.** A document enters the corpus if:
1. It is publicly accessible at the institution's official domain without login or paywall.
2. It was published or last revised after 2023-01-01 (i.e., post-ChatGPT general availability).
3. It addresses generative AI in higher education in one or more of the following contexts: teaching, learning, student assessment, research conduct, or administrative use.
4. It is written in one of the supported languages (English, Chinese simplified or traditional, German, French, Spanish, Portuguese, Italian, Dutch, Japanese, Korean). Other languages excluded.

**Document exclusion criteria.** A document is excluded if:
1. It pre-dates the 2022-11-30 ChatGPT general availability.
2. It is behind authentication, paywall, or login.
3. It addresses non-GenAI topics only (general academic integrity, technology in education broadly, etc.).
4. It is a draft, working document, or explicit "consultation in progress".

---

## 4. Data collection

**Collection window.** 2026-05-15 to 2026-06-04 (three weeks).

**Source-of-truth.** Each institution's official `.edu` / `.ac.*` / national-equivalent domain only. Third-party aggregators (e.g., AI policy databases hosted outside the institution) are excluded as primary source but may inform discovery.

**Search protocol.** Per institution, execute the search-query battery documented in `operations_manual.md` Appendix B for the institution's primary language. Use Google site-search (`site:<institution_domain> "<query>"`). Capture top-20 results per query, deduplicate by URL, fetch each, and apply the keyword-density relevance filter (≥ 3 query keywords per page).

**Fallback protocol.** For institutions where automated discovery fails, perform a 5-minute manual search. Log time spent and final outcome in the exclusion ledger.

**Translation.** Non-English originals are translated by GPT-4o using the documented translation prompt. A 5% sub-sample is back-translated by Claude Sonnet 4.6 for validation. Both original and English-translated text are archived. All inferential analyses run on English text.

---

## 5. Pre-registered codebook (binary themes + 5-point ordinal sentiment)

Eight themes, fixed before any data is coded. Detailed inclusion/exclusion criteria and worked examples are in `operations_manual.md` Appendix C.

| ID | Theme |
|---|---|
| T1 | Integration in learning and assessment |
| T2 | Multimodal and creative use |
| T3 | Security, privacy, and data protection |
| T4 | Academic integrity and misconduct |
| T5 | Disclosure and transparency |
| T6 | Equity, accessibility, and digital divide |
| T7 | Vendor governance, sovereignty, and institutional control |
| T8 | Pedagogical redesign |

Each chunk receives a binary label (0/1) per theme, plus a 5-point ordinal sentiment label (−2 to +2) per each of four use-cases (assessment / research / teaching / administration). Multi-label across themes is permitted.

---

## 6. Coding protocol (human-LLM hybrid)

**Pilot (10% of corpus, 24 institutions, ~ 1200 chunks).** Two human coders (PI Ni Jie + recruited Coder 2) independently code each pilot chunk using the v1 codebook. Disagreements are adjudicated by Prof. Adam Jatowt blind to coder identity. The adjudicated set is the gold standard.

**Inter-coder agreement targets.** Per-theme Cohen's κ ≥ 0.45 (Landis & Koch, 1977, "moderate" floor); pooled κ ≥ 0.60 ("substantial") across all 8 themes. If thresholds are not met, the codebook is revised to v2 and a 200-chunk re-pilot is run.

**LLM calibration.** GPT-4o is applied to the 1200-chunk pilot using the documented zero-shot + 16-example few-shot prompt. Per-theme F1 of GPT-4o against the human-adjudicated gold is computed. Acceptable threshold: per-theme F1 ≥ 0.75; macro-F1 ≥ 0.80.

**Production coding.** Calibrated GPT-4o is then applied to the remaining 90% of the corpus. Cross-validation: Claude Sonnet 4.6 is run on the pilot only; per-theme Cohen's κ between GPT-4o and Claude is reported as a sensitivity analysis.

---

## 7. Analytic plan

### 7.1 Descriptive analysis

For each institution, compute the **per-institution theme share** = fraction of policy chunks coded positive for that theme. For each region, compute the **regional mean share** (unweighted across institutions). All descriptives reported with bootstrap 95% confidence intervals at the institution level (B = 2000, fixed seed = 11).

### 7.2 Unsupervised cross-check

Run LDA with $k \in \{4, 6, 8, 10, 12\}$ topics over the translated English corpus. Select the optimal $k$ by C_v coherence score. Report the topic-share-by-region matrix and topic-share-by-tier matrix. Compare the LDA topics to the pre-registered 8-theme codebook by computing cosine similarity of top-20 word lists. This serves as a structural-independence check; the 8-theme codebook remains the confirmatory analysis.

### 7.3 Primary inferential analysis — TOST equivalence

For each region-pair (6 pairs: NA-EU, NA-EA, NA-LA, EU-EA, EU-LA, EA-LA) and each theme (8), compute the absolute share difference $|\hat\pi_a - \hat\pi_b|$ and apply two one-sided tests (TOST; Schuirmann, 1987; Lakens, 2017) at $\varepsilon = 0.10$. The test pair: $H_{01}: \pi_a - \pi_b \le -\varepsilon$ versus $H_{a1}$; and $H_{02}: \pi_a - \pi_b \ge \varepsilon$ versus $H_{a2}$. Equivalence is supported when both null hypotheses are rejected.

**Multiple-comparison correction.** Apply Benjamini–Hochberg false-discovery-rate (Benjamini & Hochberg, 1995) at $q = 0.10$ across the full family of $6 \times 8 = 48$ tests.

**Bootstrap CI.** Each share difference is reported with a 90% percentile bootstrap CI (B = 2000, institution-level resampling, seed = 11). The 90% CI is used because TOST at $\alpha = 0.05$ in each one-sided test corresponds to a 90% two-sided CI.

**Effect size.** Cohen's *h* (arcsine difference) reported alongside each share difference. Threshold conventions (Cohen, 1988): small $h < 0.2$; medium $h \in [0.2, 0.5]$; large $h > 0.5$.

### 7.4 Secondary inferential analysis — predictors of policy variation

For each theme $T$, fit a mixed-effects logistic regression:
$$
\mathrm{logit}\, P(T = 1) = \beta_0 + \beta_R \cdot \mathrm{Region} + \beta_T \cdot \mathrm{Tier} + \beta_P \cdot \mathrm{Public} + \beta_S \cdot \log(\mathrm{Enrollment}) + u_i + \epsilon
$$
with random intercept $u_i$ for institution. Region and Tier are categorical with treatment coding (NA + Elite as reference). Report odds ratios with 95% Wald CIs. Test for region effect via likelihood-ratio test against a null model with only $(1 \mid i)$.

### 7.5 Sentiment analysis

For each use-case (assessment / research / teaching / administration), fit an ordinal logistic regression of the 5-point sentiment score on the same covariates as 7.4. Apply BH-FDR at $q = 0.10$ across the $4 \times 4 = 16$ tests of region effect on use-case sentiment. Where omnibus is significant, run Dunn's post-hoc test with Bonferroni correction.

### 7.6 Robustness experiments

Four pre-registered robustness checks:
1. **Hold-one-region-out cross-validation** for GPT-4o-as-coder: train calibration on 3 regions, evaluate on the 4th. Report F1 drop.
2. **Per-language F1 breakdown** on the test set: separate F1 for each of the supported languages.
3. **Bootstrap 95% CI on theme shares** at the institution level.
4. **Inter-method Cohen's κ** between the keyword baseline and GPT-4o-as-coder, per theme.

---

## 8. Pre-registered decision rules

- **Sample yield**: if < 192 / 240 (80%) institutions yield documents, report actual N + bias discussion.
- **Inter-coder κ**: if pooled κ < 0.60, revise codebook v2, re-pilot 200 chunks.
- **LLM-coder F1**: if per-theme F1 < 0.75 after up to two prompt revisions, report results as exploratory for that theme.
- **TOST primary**: H3 considered supported if T4 + T5 pass TOST in ≥ 9 of 12 cells (3 pairs × 2 themes for "≥75%"); H3 considered supported on integration side if T1 + T8 fail TOST in ≥ 9 of 12 cells.
- **TOST overall**: H1 supported if ≥ 29/48 (60%) tests pass equivalence after BH-FDR; H2 supported if ≥ 29/48 fail equivalence; H1 and H2 are mutually exclusive at the 60% threshold but both can be partially supported.

---

## 9. Sample size justification and power

With $n_a = n_b = 60$ institutions per region, the standard error of a difference in binary proportions is approximately $\sqrt{0.5 \cdot 0.5 / 60 + 0.5 \cdot 0.5 / 60} = 0.0913$ at the maximum-variance case $\pi = 0.5$. The 90% CI half-width is $1.645 \times 0.0913 = 0.150$. The TOST equivalence margin is $\varepsilon = 0.10$. **This is a conservative power calculation**: the TOST is well-powered to reject equivalence when the true difference exceeds 0.10 + the CI half-width, i.e., $|\pi_a - \pi_b| > 0.25$. Detecting equivalence requires the true difference to be < 0.10 and the empirical CI to fall within ±0.10.

In practice, the power to declare equivalence is highest when the actual difference is small (≤ 0.03) and the empirical 90% CI is narrow. The power to declare non-equivalence is high for differences > 0.15 — which is what we expect for cross-regional policy variation under H3.

---

## 10. Data availability and reproducibility

Upon acceptance:
- All code released under CC-BY-4.0 on a public GitHub repository, archived to Software Heritage.
- Derived chunk-level codes, theme-share matrices, and statistical outputs released on Zenodo (DOI on acceptance), under CC-BY-4.0.
- Raw policy documents NOT redistributed (per institution's terms of use); instead, the exclusion ledger + URL + accessed-date timestamps are released to enable third-party re-scraping.
- Reproducibility deposit hashes (sha256) committed before submission.

The pre-registration itself is public on OSF upon timestamping.

---

## 11. Deviations from pre-registration

Any deviation from this pre-registration is logged with timestamp, rationale, and decision-maker in `supplement/appendix_G_preregistration_deviations.md` of the final manuscript. Deviations do not invalidate the pre-registration — they are reported transparently per Open Science conventions.

---

## OSF submission checklist

Before timestamping on OSF, confirm:

- [ ] All hypotheses (H1, H2, H3, H4) explicitly named and operationalised
- [ ] Sample frame fully documented (`data/institution_list.csv` linked)
- [ ] Search query battery committed (`data/search_queries.json` linked)
- [ ] Codebook v1 committed (Appendix C of operations manual)
- [ ] Statistical analysis plan complete
- [ ] Decision rules pre-registered
- [ ] Power calculation included
- [ ] Both authors approve text
- [ ] No inferential statistics computed yet on the planned analysis

After ticking all boxes, submit to OSF and timestamp. The resulting OSF DOI must be included in the manuscript Methods section under the heading "Pre-registration".

---

## References (cited above)

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300.
- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum.
- DiMaggio, P. J., & Powell, W. W. (1983). The iron cage revisited: Institutional isomorphism and collective rationality in organizational fields. *American Sociological Review, 48*(2), 147–160.
- Lakens, D. (2017). Equivalence tests: A practical primer for *t* tests, correlations, and meta-analyses. *Social Psychological and Personality Science, 8*(4), 355–362.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics, 33*(1), 159–174.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics, 15*(6), 657–680.
- Thornton, P. H., Ocasio, W., & Lounsbury, M. (2012). *The institutional logics perspective: A new approach to culture, structure, and process.* Oxford University Press.
