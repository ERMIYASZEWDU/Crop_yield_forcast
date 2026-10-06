# Team Adwa - Ethiopian Smallholder Crop-Yield Challenge

**Qiyas / IADE Training Program, Addis Ababa University - 5 October 2026**

**Team:** Team Adwa - *[add the three or four member names here before submitting]*

## Summary

We predict `yield_tons_per_ha` for 3,750 Ethiopian smallholder plots from a raw survey export,
a regional monthly weather table and a market price table. All three files were cleaned with
statistics fit on the train file only (label standardisation, `-999` sentinels, blanks, outlier
caps, a weather duplicate-key fix and a per-kilo price unit fix), joined many-to-one using a
growing-season window of **planting month + the following three months**, and engineered into a
21-feature model table with six weather-derived features. Five models were compared on one
80/20 split (seed 42) with 5-fold cross-validation, an out-of-time 2024 check, a weather
ablation and a 16-trial tuning search. The final tuned model reaches **validation RMSE
0.478 t/ha (MAE 0.347, R2 0.885)** against a mean-predictor baseline of 1.413 t/ha; removing the
weather features costs 6.6% accuracy. Deliverables A-D are written to `reports/`, the twelve
graded figures live in `figures/`, the deck in `presentation/`, and a Streamlit demo predicts
yield **and** revenue with weather and price looked up automatically.

## Setup

```bash
python -m venv .venv            # optional
# Windows: .venv\Scripts\activate   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ (developed on 3.14). All paths in the code are relative to this folder.

## Run order

From the project root:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_cleaning_and_integration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_analysis_report.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_visualizations.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_modeling_and_evaluation.ipynb
```

(or open them in JupyterLab/VS Code and run top to bottom - the order matters: 01 builds the
master tables, 02 analyses them, 03 draws the figures and trains/loads the evaluation artifacts,
04 reports the modeling results, writes the submission and builds the slides).

Note: `src/train.py` caches the evaluation artifacts in `models/*.csv`; delete those files to
force a full re-run of the model comparison (~2 minutes).

## Demo (Deliverable E)

```bash
streamlit run app/app.py
```

Runs locally - no hosting was available on the day, so the demo is presented live from the
laptop. The user picks region, crop, year, planting month and the plot details; the app looks up
the growing-season weather and the market price from `app/assets/` and returns predicted yield,
revenue for the plot, a comparison against the region x crop average and a fertilizer what-if
curve. (If you later get a URL, add it here.)

## Where each deliverable lives

| Item | Location |
|---|---|
| Prediction file (G4) | `submission/team_adwa_submission.csv` (3,750 rows, original order) |
| A - cleaning log, join map, audit, proof, feature table, checks, master tables | `notebooks/01_cleaning_and_integration.ipynb`, `reports/A_cleaning_and_integration.md`, `reports/join_map_diagram.png`, `data/processed/` |
| B - analysis, all 14 tasks | `notebooks/02_analysis_report.ipynb`, `reports/B_analysis_report.md` |
| C - 12 figures + captions | `figures/fig01..fig12.png`, `figures/figure_captions.md`, `notebooks/03_visualizations.ipynb` |
| D - modeling D1-D9 + saved model | `notebooks/04_modeling_and_evaluation.ipynb`, `reports/D_model_evaluation.md`, `models/final_model.joblib` |
| E - demo app + bundled tables | `app/app.py`, `app/assets/`, `app/requirements.txt` |
| F - 5 slides | `presentation/team_adwa_slides.pptx` (built by notebook 04) |
| G - structure, README, requirements | this file, `requirements.txt`, folder layout |

## Scores

| Metric | Value |
|---|---|
| Final model (tuned `mlp_neural_net`) - held-out RMSE / MAE / R2 | **0.478 / 0.347 / 0.885** |
| Final model - 5-fold CV RMSE (untuned, full train) | 0.481 +/- 0.005 |
| Runner-up (hist gradient boosting) - 5-fold CV RMSE | 0.484 +/- 0.014 |
| Mean-predictor baseline - held-out RMSE | 1.413 |
| Weather ablation: with / without weather features | 0.483 / 0.518 (weather worth 6.6%) |
| Out-of-time check (train 2021-2023, validate 2024) | 0.564 RMSE, 0.845 R2 |
| Submission | 3,750 rows, no blanks, no duplicates, template order - passes all G4 checks |

Split: 80/20 of the train file, `random_state=42`. Nothing was fit on the test file, price is
never a model feature, and the model uses six weather-derived features (Rules 5 and 6).
The leaderboard score against the hidden key is not available to us - the validation numbers
above are the ones we can stand behind.

## Reproducibility (G3)

- `requirements.txt` pins every package used; `app/requirements.txt` pins the demo's subset.
- Relative paths only; raw files in `data/raw/` are never edited.
- Every random seed is fixed at 42 (`RANDOM_SEED` in each notebook, `SEED` in `src/train.py`).
- Every number, table and figure in the reports comes from code in this folder: the reports and
  the slide deck are written by the notebooks themselves, so re-running them regenerates
  everything (including `figures/figure_captions.md`, `reports/*.md` and the `.pptx`).
