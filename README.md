# GenAI Policy Maturation

Minimal reproducibility repository for the manuscript:

**From Interim Rules to Institutional Futures: A 200-University Cross-National
Analysis of Generative AI Policy Maturation**

This repository is intentionally lean. It contains the codebook, the main
analysis pipeline scripts, and aggregate derived result tables needed to
reproduce the numerical tables reported in the manuscript. It does **not**
contain manuscript drafts, submission files, figures, model weights, local
cluster logs, or raw university policy text.

Repository URL: <https://github.com/Jie-Ni/genai-policy-coding>

## Contents

```text
codebook.md
    Eight-theme GenAI policy codebook and coding schema.

data/
    pfe_measurement_summary.csv
    pfe_country_icc.csv
    pfe_regional_contrasts.csv
    pfe_temporal_drift.csv
    README.md

scripts/
    01_build_institution_list.py
    02_scrape_policies.py
    03_extract_text.py
    04_chunk_policies.py
    07_local_llm_coder.py
    09_stats_analysis.py
    09d_ablation_analysis.py
    reproduce_pfe_locked_results.py
```

## Reproducing the locked manuscript tables

The current public package ships aggregate derived results rather than raw
policy text. To regenerate the manuscript result tables from those locked
aggregate results:

```bash
python scripts/reproduce_pfe_locked_results.py --data-dir data --out-dir tables
```

This writes:

```text
tables/table_measurement_summary.md
tables/table_regional_contrasts.md
tables/table_temporal_drift.md
tables/table_country_icc.md
```

No GPU is required for this summary-level reproduction.

## Data boundary

The study uses public university policy documents. The repository does not
redistribute verbatim source policy text, because such documents may be subject
to institution-specific copyright, reuse, or website terms. The public release
therefore separates:

- released: codebook, analysis scripts, derived aggregate result tables;
- not released here: raw policy-page text, model weights, local cluster logs,
  manuscript drafts, submission packages, and generated figures.

The source-discovery and coding scripts document the computational workflow used
for the larger corpus. The locked aggregate CSV files are the authoritative
public values for the current PFE manuscript package.

## Compute acknowledgement

The original coding workflow used MUSICA high-performance computing resources
provided through the Austrian Scientific Computing infrastructure.

## License

- Source code: MIT license.
- Derived aggregate tables and codebook: CC BY 4.0.
