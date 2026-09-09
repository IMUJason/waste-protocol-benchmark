# Waste-forecasting evaluation-protocol benchmark

Reproduction code and frozen input panels for a full-factorial benchmark of
evaluation protocols in waste-generation forecasting, together with a coded
survey of 110 published waste-forecasting studies (2015–2026).

The benchmark crosses eight models (persistence, entity mean, OLS, Lasso,
random forest, XGBoost, GP, MR-GP) with four protocol factors
(validation design: rolling origin / LOOCV / random 5-fold; covariate
timing: lagged / contemporaneous; target-derived features: off / on;
preprocessing scope: fold-internal / global) on four panels:

| Panel | Units | Description |
|---|---|---|
| A  | 21 Chinese cities | CDW generation, 2015–2024, four-level provenance flags |
| A17 | 18 cities | robustness subset of A (screened cities dropped) |
| B  | 33 European countries | Eurostat municipal waste, 2004–2023 |
| C0 | Beijing + Shenzhen | case study carrying CDW sub-stream covariates |

All comparisons use a common (rolling-origin) evaluation window, paired
entity bootstrap, and same-station persistence baselines.

## Layout

```
data/    frozen input panels (see provenance below)
src/     pipeline scripts (run in order via run_all.py)
output/  generated at run time (analysis CSVs, numbers.tex)
```

## Reproduce

```
pip install -r requirements.txt
python src/run_all.py              # fetches Eurostat, then runs everything
python src/run_all.py --skip-download   # use the frozen Eurostat copy instead
```

Runtime is under one CPU-hour. Random seeds are fixed; results are
deterministic. Key outputs:

- `output/analysis_inflation.csv` — honest-vs-common R2 with paired bootstrap CIs
- `output/analysis_ranking.csv` — Kendall tau between protocol rankings
- `output/analysis_persistence.csv` — model-vs-persistence win counts per station
- `output/analysis_factor_decomp.csv` — one-factor-at-a-time decomposition
- `output/analysis_tuned.csv` — fold-internal tuned-tree control station
- `output/analysis_mechanism.csv` — per-entity protocol gain vs series structure
- `output/numbers.tex` — every headline number as a LaTeX macro

As a guard against silent breakage, the persistence implementation
reproduces reference values for the C0 panel exactly (pooled LOOCV
R2 = 0.862; rolling-origin R2 = 0.807).

## Data provenance

- **Panel A / A17**: derived from Chinese provincial environmental and
  statistical yearbook channels. Each city-year carries a four-level
  provenance flag (reported / interpolated-assisted / suspected-estimated /
  contradicted). Original government records are not redistributed; only the
  derived analysis panel ships here.
- **Panel B**: Eurostat `env_wasmun` (municipal waste generated), plus
  `nama_10_gdp` and `tps00001` covariates; redistributable under the
  Eurostat reuse policy. `src/wp2_fetch_dataset_b.py` re-downloads it.
- **Panel C0**: derived from Beijing and Shenzhen public environmental
  records; the 2015–2019 Shenzhen segment is interpolated-assisted (flagged).
- **Survey** (`data/protocol_survey.csv`): 110 coded studies; every record
  carries a verified DOI or an archival identifier (five records without
  DOI). Coding fields: validation design, temporal vs cross-sectional data,
  persistence baseline, target-derivation risk, uncertainty reporting.
  `data/reliability_agreement.csv` documents the blind re-coding check on a
  seeded 13-study subsample.

## Requirements

Python ≥ 3.10 with numpy, pandas, scikit-learn, xgboost, scipy, pyarrow
(see `requirements.txt`).

## License

Code: MIT. Data: see the provenance section; the Eurostat panel is subject
to the Eurostat reuse policy; panels A/A17/C0 are provided for research
reproduction with source attribution to the yearbook channels listed in the
accompanying article.

## Citation

Please cite the accompanying article (Cao, 2026).
