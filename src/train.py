"""Model building and evaluation for Deliverable D.

Everything here learns from ``data/processed/master_train.csv`` only:
a fixed 80/20 split with seed 42, 5-fold cross-validation, an out-of-time
check (2021-2023 -> 2024), a weather ablation and a RandomizedSearchCV
tuning pass.  Results are cached as CSVs in ``models/`` so notebooks 03 and
04 report exactly the same numbers.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                              VotingRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES,
                          WEATHER_FEATURES)

SEED = 42
TEST_SIZE = 0.2
MODEL_DIR = Path("models")
FINAL_MODEL_PATH = MODEL_DIR / "final_model.joblib"

# short names used for the ensemble label (ensemble_mlp_hgb, ...)
SHORT = {"mlp_neural_net": "mlp", "hist_gradient_boosting": "hgb",
         "random_forest": "rf"}


def short_name(name: str) -> str:
    return SHORT.get(name, name)

ARTIFACTS = {
    "eval": MODEL_DIR / "eval_results.csv",
    "cv": MODEL_DIR / "cv_results.csv",
    "ablation": MODEL_DIR / "ablation_results.csv",
    "oot": MODEL_DIR / "oot_results.csv",
    "tuning": MODEL_DIR / "tuning_results.csv",
    "holdout": MODEL_DIR / "holdout_predictions.csv",
    "perm": MODEL_DIR / "permutation_importance.csv",
    "extra": MODEL_DIR / "d8_experiment_results.csv",
    "search": MODEL_DIR / "search_results.csv",
    "ens": MODEL_DIR / "ensemble_results.csv",
    "ens_scan": MODEL_DIR / "ensemble_weight_scan.csv",
}


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def make_pipeline(estimator, features: list[str] | None = None, scale: bool = False) -> Pipeline:
    """Preprocessing (train-fitted imputer + one-hot) + model in one object."""
    features = MODEL_FEATURES if features is None else features
    categorical = [c for c in features if c in CATEGORICAL_FEATURES]
    numeric = [c for c in features if c not in CATEGORICAL_FEATURES]

    num_steps = [("impute", "passthrough")]
    if scale:
        num_steps = [("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps) if scale else "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def default_models(scale_mlp: bool = True) -> dict[str, Pipeline]:
    """Baselines (D1) plus three advanced families (D2)."""
    return {
        "dummy_mean": make_pipeline(DummyRegressor(strategy="mean")),
        "ridge": make_pipeline(Ridge(alpha=1.0, random_state=SEED), scale=True),
        "random_forest": make_pipeline(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                  n_jobs=-1, random_state=SEED)),
        "hist_gradient_boosting": make_pipeline(
            HistGradientBoostingRegressor(random_state=SEED)),
        "mlp_neural_net": make_pipeline(
            MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                         max_iter=400, early_stopping=True, random_state=SEED),
            scale=scale_mlp),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def split_data(master: pd.DataFrame, target: str = "yield_tons_per_ha"):
    """Fixed 80/20 holdout split, seed 42 (stated in report D)."""
    X = master[MODEL_FEATURES]
    y = master[target]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


# --------------------------------------------------------------------------- #
# Cached artifact builder
# --------------------------------------------------------------------------- #
def artifacts_ready() -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in ARTIFACTS.values())


def load_artifacts() -> dict[str, pd.DataFrame]:
    return {k: pd.read_csv(p) for k, p in ARTIFACTS.items()}


def _fit_predict(name: str, pipe: Pipeline, X_tr, y_tr, X_va, y_va):
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0
    pred = pipe.predict(X_va)
    row = {"model": name, **regression_metrics(y_va, pred),
           "train_time_s": round(train_time, 2)}
    return row, pred


def build_artifacts(master: pd.DataFrame, force: bool = False,
                    n_iter_search: int = 16) -> dict[str, pd.DataFrame]:
    """Run the full D1-D8 evaluation once and cache every table it produces."""
    if artifacts_ready() and not force:
        return load_artifacts()

    MODEL_DIR.mkdir(exist_ok=True)
    X_tr, X_va, y_tr, y_va = split_data(master)
    X_all = master[MODEL_FEATURES]
    y_all = master["yield_tons_per_ha"]

    # ---- D1 + D2: baselines and model families on the same split ------------
    eval_rows, preds = [], {}
    for name, pipe in default_models().items():
        row, pred = _fit_predict(name, pipe, X_tr, y_tr, X_va, y_va)
        eval_rows.append(row)
        preds[name] = pred
        print(f"  {name:>24s}  rmse={row['rmse']:.4f}  mae={row['mae']:.4f}  "
              f"r2={row['r2']:.4f}  ({row['train_time_s']:.1f}s)")
    eval_df = pd.DataFrame(eval_rows).sort_values("rmse").reset_index(drop=True)
    eval_df.to_csv(ARTIFACTS["eval"], index=False)

    # ---- D3: 5-fold cross-validation for every advanced family -------------
    # The final model family is chosen by 5-fold CV mean RMSE (most reliable
    # estimate we have); runner-up is the second-best family.
    advanced = ["hist_gradient_boosting", "random_forest", "mlp_neural_net"]
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for name in advanced:
        pipe = default_models()[name]
        scores = cross_val_score(pipe, X_all, y_all, cv=cv, n_jobs=-1,
                                 scoring="neg_root_mean_squared_error")
        rmse = -scores
        cv_rows.append({"model": name, "cv_rmse_mean": float(rmse.mean()),
                        "cv_rmse_std": float(rmse.std(ddof=1)),
                        "folds": ",".join(f"{s:.4f}" for s in rmse)})
        print(f"  cv {name:>24s}  {rmse.mean():.4f} +/- {rmse.std(ddof=1):.4f}")
    cv_df = (pd.DataFrame(cv_rows).sort_values("cv_rmse_mean")
             .reset_index(drop=True))
    cv_df.to_csv(ARTIFACTS["cv"], index=False)
    final_name = str(cv_df.loc[0, "model"])
    runner_up = str(cv_df.loc[1, "model"])
    print(f"  final model family: {final_name} | runner-up: {runner_up}")

    # ---- D5: weather ablation ---------------------------------------------
    no_weather = [f for f in MODEL_FEATURES if f not in WEATHER_FEATURES]
    ablation_rows = []
    for label, feats in [("with_weather", MODEL_FEATURES),
                         ("without_weather", no_weather)]:
        pipe = default_models()[final_name]
        pipe = make_pipeline(pipe.named_steps["model"], features=feats,
                             scale=(final_name == "mlp_neural_net"))
        row, _ = _fit_predict(final_name, pipe, X_tr[feats], y_tr, X_va[feats], y_va)
        ablation_rows.append({"variant": label, "n_features": len(feats),
                              "rmse": row["rmse"], "mae": row["mae"], "r2": row["r2"]})
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(ARTIFACTS["ablation"], index=False)

    # ---- D4: out-of-time check (train 2021-2023, validate 2024) ------------
    year = master["survey_year"].to_numpy()
    tr_mask = year <= 2023
    oot_pipe = default_models()[final_name]
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oot_pipe.fit(X_all[tr_mask], y_all[tr_mask])
    oot_pred = oot_pipe.predict(X_all[~tr_mask])
    oot_df = pd.DataFrame([{
        "model": final_name,
        "train_years": "2021-2023", "valid_year": 2024,
        "n_train": int(tr_mask.sum()), "n_valid": int((~tr_mask).sum()),
        **regression_metrics(y_all[~tr_mask], oot_pred),
        "train_time_s": round(time.perf_counter() - t0, 2),
    }])
    oot_df.to_csv(ARTIFACTS["oot"], index=False)

    # ---- D6: hyperparameter search -----------------------------------------
    space = _search_space(final_name)
    search = RandomizedSearchCV(
        make_pipeline(default_models()[final_name].named_steps["model"],
                      scale=(final_name == "mlp_neural_net")),
        param_distributions=space, n_iter=n_iter_search, cv=3,
        scoring="neg_root_mean_squared_error", random_state=SEED,
        n_jobs=-1, refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_tr, y_tr)
    best_pred = search.predict(X_va)
    before = eval_df.loc[eval_df["model"] == final_name].iloc[0]
    tuning_df = pd.DataFrame([
        {"stage": "before_tuning", "model": final_name,
         "rmse": float(before["rmse"]), "mae": float(before["mae"]),
         "r2": float(before["r2"]),
         "params": str(default_models()[final_name].named_steps["model"].get_params())},
        {"stage": "after_tuning", "model": final_name,
         "rmse": float(regression_metrics(y_va, best_pred)["rmse"]),
         "mae": float(regression_metrics(y_va, best_pred)["mae"]),
         "r2": float(regression_metrics(y_va, best_pred)["r2"]),
         "params": str(search.best_params_)},
    ])
    tuning_df.to_csv(ARTIFACTS["tuning"], index=False)
    cvres = pd.DataFrame(search.cv_results_)
    cvres.insert(0, "model", final_name)
    cvres.to_csv(ARTIFACTS["search"], index=False)

    # ---- ensemble of the two best families (final model selection) ---------
    # Second member gets the same tuning budget as the first (D6); the mixing
    # weight is chosen once, by 3-fold out-of-fold predictions on the TRAIN
    # split only - the holdout is never used to pick it.
    ens_name = f"ensemble_{short_name(final_name)}_{short_name(runner_up)}"
    search2 = RandomizedSearchCV(
        make_pipeline(default_models()[runner_up].named_steps["model"]),
        param_distributions=_search_space(runner_up), n_iter=n_iter_search,
        cv=3, scoring="neg_root_mean_squared_error", random_state=SEED,
        n_jobs=-1, refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search2.fit(X_tr, y_tr)

    kf3 = KFold(n_splits=3, shuffle=True, random_state=SEED)
    oof1 = cross_val_predict(search.best_estimator_, X_tr, y_tr, cv=kf3, n_jobs=-1)
    oof2 = cross_val_predict(search2.best_estimator_, X_tr, y_tr, cv=kf3, n_jobs=-1)
    grid = np.arange(0.0, 1.01, 0.1)
    scan_df = pd.DataFrame({
        "weight_first_member": grid,
        "oof_rmse": [float(np.sqrt(mean_squared_error(
            y_tr, g * oof1 + (1.0 - g) * oof2))) for g in grid]})
    scan_df.to_csv(ARTIFACTS["ens_scan"], index=False)
    w_first = float(grid[int(np.argmin(scan_df.oof_rmse))])
    print(f"  ensemble weight scan -> w({final_name}) = {w_first:.2f}")

    ens_va_pipe = VotingRegressor(
        [(final_name, search.best_estimator_), (runner_up, search2.best_estimator_)],
        weights=[w_first, 1.0 - w_first])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ens_va_pipe.fit(X_tr, y_tr)
    ens_va_pred = ens_va_pipe.predict(X_va)

    single_row = regression_metrics(y_va, best_pred)          # tuned first member
    second_row = regression_metrics(y_va, search2.predict(X_va))  # tuned second member
    ens_row = regression_metrics(y_va, ens_va_pred)
    ens_df = pd.DataFrame([
        {"variant": f"{final_name} (tuned)", **single_row},
        {"variant": f"{runner_up} (tuned)", **second_row},
        {"variant": ens_name, **ens_row},
    ])
    ens_df.to_csv(ARTIFACTS["ens"], index=False)
    print(f"  holdout: {final_name} tuned {single_row['rmse']:.4f} | "
          f"{runner_up} tuned {second_row['rmse']:.4f} | {ens_name} "
          f"{ens_row['rmse']:.4f} (w={w_first:.2f})")

    # 5-fold CV of the equal-weight ensemble (default members, same setting as D3)
    ens_cv_pipe = VotingRegressor(
        [(final_name, default_models()[final_name]),
         (runner_up, default_models()[runner_up])])
    ens_scores = -cross_val_score(ens_cv_pipe, X_all, y_all, cv=cv, n_jobs=-1,
                                  scoring="neg_root_mean_squared_error")
    cv_df.loc[len(cv_df)] = {"model": ens_name,
                             "cv_rmse_mean": float(ens_scores.mean()),
                             "cv_rmse_std": float(ens_scores.std(ddof=1)),
                             "folds": ",".join(f"{s:.4f}" for s in ens_scores)}
    cv_df.to_csv(ARTIFACTS["cv"], index=False)
    print(f"  cv {ens_name:>24s}  {ens_scores.mean():.4f} +/- "
          f"{ens_scores.std(ddof=1):.4f}")

    # both members tuned: append the second member's rows to the D6 table
    second_before = eval_df.loc[eval_df.model == runner_up].iloc[0]
    tuning_df = pd.concat([tuning_df, pd.DataFrame([
        {"stage": "before_tuning", "model": runner_up,
         "rmse": float(second_before["rmse"]), "mae": float(second_before["mae"]),
         "r2": float(second_before["r2"]),
         "params": str(default_models()[runner_up].named_steps["model"].get_params())},
        {"stage": "after_tuning", "model": runner_up, **second_row,
         "params": str(search2.best_params_)},
    ])], ignore_index=True)
    tuning_df.to_csv(ARTIFACTS["tuning"], index=False)

    # out-of-time check for the ensemble as well (same 2021-2023 -> 2024 split)
    oot_ens = VotingRegressor(
        [(final_name, search.best_estimator_), (runner_up, search2.best_estimator_)],
        weights=[w_first, 1.0 - w_first])
    t0e = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oot_ens.fit(X_all[tr_mask], y_all[tr_mask])
    oot_df = pd.concat([oot_df, pd.DataFrame([{
        "model": ens_name,
        "train_years": "2021-2023", "valid_year": 2024,
        "n_train": int(tr_mask.sum()), "n_valid": int((~tr_mask).sum()),
        **regression_metrics(y_all[~tr_mask], oot_ens.predict(X_all[~tr_mask])),
        "train_time_s": round(time.perf_counter() - t0e, 2),
    }])], ignore_index=True)
    oot_df.to_csv(ARTIFACTS["oot"], index=False)

    # ---- final holdout predictions (whichever model wins) ------------------
    tuned_single_rmse = float(tuning_df.loc[
        (tuning_df.stage == "after_tuning") & (tuning_df.model == final_name),
        "rmse"].iloc[0])
    use_ensemble = bool(ens_row["rmse"] < tuned_single_rmse)
    final_tr_model = ens_va_pipe if use_ensemble else search.best_estimator_
    final_pred = ens_va_pred if use_ensemble else best_pred

    holdout = X_va.copy()
    holdout["actual_yield_tons_per_ha"] = y_va.values
    holdout["predicted_yield_tons_per_ha"] = final_pred
    holdout["residual"] = holdout["predicted_yield_tons_per_ha"] - holdout["actual_yield_tons_per_ha"]
    holdout["abs_error"] = holdout["residual"].abs()
    holdout["plot_id"] = master.loc[X_va.index, "plot_id"].values
    holdout.to_csv(ARTIFACTS["holdout"], index=False)

    # ---- D7: permutation importance on the final model ---------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(final_tr_model, X_va, y_va,
                                      n_repeats=10, random_state=SEED,
                                      scoring="neg_root_mean_squared_error",
                                      n_jobs=-1)
    imp = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp["is_weather_derived"] = imp["feature"].isin(WEATHER_FEATURES)
    imp.to_csv(ARTIFACTS["perm"], index=False)

    # ---- D8: feature change suggested by the error analysis ----------------
    extra_feats = sorted(set(MODEL_FEATURES) | {"rain_ratio", "fert_x_soil"})
    exp_pipe = make_pipeline(default_models()[final_name].named_steps["model"],
                             features=extra_feats,
                             scale=(final_name == "mlp_neural_net"))
    X_tr_e = _augment(X_tr)[extra_feats]
    X_va_e = _augment(X_va)[extra_feats]
    row, _ = _fit_predict(final_name, exp_pipe, X_tr_e, y_tr, X_va_e, y_va)
    extra_df = pd.DataFrame([
        {"variant": "final_features", "n_features": len(MODEL_FEATURES),
         "rmse": float(before["rmse"]), "mae": float(before["mae"]),
         "r2": float(before["r2"])},
        {"variant": "plus_extra_interactions", "n_features": len(extra_feats),
         "rmse": row["rmse"], "mae": row["mae"], "r2": row["r2"]},
    ])
    extra_df.to_csv(ARTIFACTS["extra"], index=False)

    # ---- refit the final model on the FULL train file ----------------------
    bundle_meta = {"features": MODEL_FEATURES, "seed": SEED}
    if use_ensemble:
        final_pipe = VotingRegressor(
            [(final_name, search.best_estimator_), (runner_up, search2.best_estimator_)],
            weights=[w_first, 1.0 - w_first])
        bundle_meta.update({
            "model_name": ens_name,
            "best_params": {final_name: search.best_params_,
                            runner_up: search2.best_params_},
            "members": [final_name, runner_up],
            "ensemble_weight": w_first,
        })
    else:
        final_pipe = make_pipeline(search.best_estimator_.named_steps["model"],
                                   scale=(final_name == "mlp_neural_net"))
        bundle_meta.update({"model_name": final_name,
                            "best_params": search.best_params_})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipe.fit(X_all, y_all)
    joblib.dump({"pipeline": final_pipe, **bundle_meta}, FINAL_MODEL_PATH)
    print(f"  final model: {bundle_meta['model_name']}"
          + (f" (holdout RMSE {ens_row['rmse']:.4f})" if use_ensemble else ""))

    return load_artifacts()


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    """Extra features proposed by the D7 error analysis."""
    d = df.copy()
    d["rain_ratio"] = (d["rainfall_mm_season"]
                       / d["wx_season_rain_total_mm"].clip(lower=1.0)).round(4)
    d["fert_x_soil"] = (d["fertilizer_kg_per_ha"] * d["soil_quality_index"]).round(4)
    return d


def _search_space(model_name: str) -> dict:
    if model_name == "random_forest":
        return {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 12, 20, 30],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", 0.5, 0.8],
        }
    if model_name == "mlp_neural_net":
        return {
            "model__hidden_layer_sizes": [(64,), (64, 64), (128, 64), (128, 128)],
            "model__alpha": [1e-4, 1e-3, 1e-2],
            "model__learning_rate_init": [1e-3, 3e-3, 1e-2],
        }
    # histogram gradient boosting (default)
    return {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [10, 20, 30, 50],
        "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "model__max_iter": [200, 300, 400],
    }
