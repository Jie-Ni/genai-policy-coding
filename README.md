# Cross-National GenAI Policy Coding

A reproducible open-weight dual-LLM coding pipeline for analysing university
generative-AI policy text across world regions and languages, with the
accompanying 64-institution corpus and the full statistical analysis
underlying the manuscript currently under review.

> **Status (as of repo creation).** The manuscript is under double-anonymous
> peer review. Journal identity, manuscript number, and DOI will be added to
> this README at acceptance.

---

## What's in this repository

```
genai-policy-coding/
├── codebook.md                       # the 8-theme codebook v1.0 + sentiment rubric
│                                     # + 16 worked few-shot examples + JSON schema
├── institution_list.csv              # 64-institution sampling frame
│                                     # (region, tier, language, ranking, ROR ID)
├── search_queries.json               # 22-query multilingual search battery
│                                     # (pre-registered before any scraping)
├── preregistration.md                # OSF pre-registration document
├── codebook.md                       # operational coding definitions
├── chunks_metadata/                  # chunk metadata (text stripped — see Note 1)
│   ├── ablation_chunks.metadata.jsonl       # 80 stratified ablation chunks
│   ├── expert_gold_subset.metadata.jsonl    # 40 human gold-standard chunks
│   ├── smoke_test_chunks.metadata.jsonl     # 5 pipeline smoke-test chunks
│   └── expert_gold_codes.csv                # hand-coded labels (no text)
├── codings/                          # dual-LLM coding outputs (no text needed)
│   └── shard_{00..07}.jsonl          # 338 Qwen3.6-27B codings, schema-constrained
├── stats/                            # full analysis CSVs (37 files)
│   ├── region_theme_prevalence.csv          # Table 1 in manuscript
│   ├── region_sentiment_means.csv           # Table 2
│   ├── ablation_A1_bootstrap_kappa.csv      # Table 3
│   ├── confirmatory_C2_mixed_logit_region.csv  # Table 5
│   ├── extended_E*_…csv                     # extended inferential tables
│   ├── sensitivity_S*_…csv                  # robustness probes
│   ├── tost_*.csv                           # TOST equivalence tests
│   ├── extended_summary.md
│   └── human_validation_summary.md
├── scripts/                          # analysis pipeline (Python)
│   ├── 01_build_institution_list.py
│   ├── 02_scrape_policies.py
│   ├── 03_extract_text.py
│   ├── 04_chunk_policies.py
│   ├── 07_local_llm_coder.py         # vLLM-based schema-constrained coder
│   ├── 09_stats_analysis.py          # all confirmatory tests
│   ├── 09b_sensitivity.py            # S1 resample, S2 stratification, S3 per-coder κ
│   ├── 09c_ablations.py              # 7-condition prompt ablation
│   ├── 09e_confirmatory.py           # chi-square, Cochran-Armitage, mixed-logit
│   ├── 09f_extended_stats.py         # Wilson, Holm, Kruskal, ICC, MDE
│   ├── 09g_human_validation.py       # human-vs-LLM κ
│   └── common.py
└── manuscript/                       # paper sources
    ├── manuscript.md                 # main text (markdown)
    ├── tables/                       # 5 generated tables (markdown)
    └── figures/                      # 7 figures (PNG)
```

---

## Reproducing the analysis

```bash
# 1. clone the repo
git clone <this-repo-url> && cd genai-policy-coding

# 2. set up env (single-machine, no GPU needed for the statistics)
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas pyyaml  # only stats stack required; no statsmodels

# 3. regenerate all statistical CSVs from the 338 codings
python scripts/09_stats_analysis.py
python scripts/09b_sensitivity.py
python scripts/09e_confirmatory.py
python scripts/09f_extended_stats.py
python scripts/09g_human_validation.py

# 4. regenerate manuscript tables
python -c "import sys; sys.path.insert(0, 'manuscript'); from make_tables import main; main()"
```

Re-running the **LLM coding** itself requires GPU and the open-weight model
weights (Qwen3.6-27B, ~52 GB; Mistral-Small-3.2-24B). Coding scripts read
chunk text not redistributed here (see Note 1); a SLURM array launcher is
provided in `scripts/` for reference.

---

## Notes on reproducibility

### Note 1 — Source policy text

The verbatim policy-chunk text is **not** redistributed in this repository.
University policy documents are subject to per-institution copyright and
some jurisdictions' fair-use provisions do not extend to bulk
redistribution. The repository ships:

- **Chunk metadata** (institution, region, language, section, source URL,
  word count, approximate token count) — sufficient for reviewers to
  re-fetch each chunk from its source URL.
- **The full LLM coding outputs** (theme labels, sentiment scores,
  confidence flags, coder notes) for every chunk.
- **The codebook** with eight worked few-shot examples that *are* synthetic
  and freely redistributable.

The raw policy-chunk text is available **from the corresponding author on
reasonable request**, subject to a fair-use review.

### Note 2 — Models

The two coders named in the manuscript (Qwen3.6-27B and
Mistral-Small-3.2-24B-Instruct-2506) are publicly hosted on Hugging Face
under permissive licences (Apache-2.0). The repository does **not** include
the model weights — re-download via:

```bash
hf download Qwen/Qwen3.6-27B --local-dir ./models/Qwen_Qwen3.6-27B
hf download mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
        --local-dir ./models/mistralai_Mistral-Small-3.2-24B-Instruct-2506
```

### Note 3 — Pre-registration

The codebook, search battery, eight a-priori hypotheses, sample
stratification rules, and the full analysis plan were time-stamped on OSF
prior to any LLM coding execution. See `preregistration.md`.

---

## Licensing

- **Source code** (everything under `scripts/` and `manuscript/`):
  [MIT licence](LICENSE).
- **Data and analysis outputs** (everything under `chunks_metadata/`,
  `codings/`, `stats/`, plus `institution_list.csv`, `search_queries.json`):
  [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA).
- **Codebook and few-shot examples** (`codebook.md`): CC BY 4.0.

---

## Citation

A formal citation will be added at journal acceptance. A
forward-compatible Zenodo DOI will also be deposited at that time. Until
then, please cite by repository URL.

---

## Contact

Open an issue on this repository, or contact the corresponding author
(see `manuscript/manuscript.md`).
