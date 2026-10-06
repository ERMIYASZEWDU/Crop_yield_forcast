"""Submission helpers (G4) and prediction used by the demo."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train import FINAL_MODEL_PATH

SUBMISSION_DIR = Path("submission")
TEMPLATE_PATH = Path("data/raw/submission_template.csv")
SUB_COLUMNS = ["plot_id", "predicted_yield_tons_per_ha"]


def load_bundle(path: Path = FINAL_MODEL_PATH) -> dict:
    """Load the saved pipeline + the exact feature list it was trained on."""
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"{path} is missing or empty - run notebook 04 first.")
    return joblib.load(path)


def predict_master(master: pd.DataFrame, bundle: dict | None = None) -> pd.Series:
    bundle = bundle or load_bundle()
    pipe = bundle["pipeline"]
    feats = bundle["features"]
    pred = pipe.predict(master[feats])
    # yields are physically non-negative
    return pd.Series(np.clip(pred, 0.0, None), index=master.index)


def build_submission(master_test: pd.DataFrame, template: pd.DataFrame,
                     bundle: dict | None = None) -> pd.DataFrame:
    """Fill the template in its original row order (never re-sorted)."""
    preds = predict_master(master_test, bundle)
    mapping = dict(zip(master_test["plot_id"], preds))
    out = template.copy()
    out["predicted_yield_tons_per_ha"] = out["plot_id"].map(mapping)
    if out["predicted_yield_tons_per_ha"].isna().any():
        bad = out.loc[out["predicted_yield_tons_per_ha"].isna(), "plot_id"].head(5).tolist()
        raise ValueError(f"no prediction produced for plot_ids: {bad}")
    out["predicted_yield_tons_per_ha"] = out["predicted_yield_tons_per_ha"].astype("float64")
    return out[SUB_COLUMNS]


def write_submission(df: pd.DataFrame, path: str | Path = SUBMISSION_DIR /
                     "team_adwa_submission.csv") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def validate_submission(path: str | Path,
                        template_path: str | Path = TEMPLATE_PATH) -> pd.DataFrame:
    """G4 checks: shape, columns, order, uniqueness, no blanks, numeric."""
    sub = pd.read_csv(path)
    tmpl = pd.read_csv(template_path)
    checks = {
        "exact columns": list(sub.columns) == SUB_COLUMNS,
        "row count == 3750": len(sub) == 3750 == len(tmpl),
        "plot_ids in original order": (sub["plot_id"].values == tmpl["plot_id"].values).all(),
        "plot_ids unique": not sub["plot_id"].duplicated().any(),
        "no blank predictions": sub["predicted_yield_tons_per_ha"].notna().all(),
        "all numeric": pd.to_numeric(
            sub["predicted_yield_tons_per_ha"], errors="coerce").notna().all(),
        "no negative yields": (sub["predicted_yield_tons_per_ha"] >= 0).all(),
        "plausible range (< 20 t/ha)": (sub["predicted_yield_tons_per_ha"] < 20).all(),
    }
    report = pd.DataFrame(
        [{"check": k, "result": "PASS" if v else "FAIL"} for k, v in checks.items()])
    if not all(checks.values()):
        raise AssertionError("submission failed:\n" + report.to_string(index=False))
    return report
