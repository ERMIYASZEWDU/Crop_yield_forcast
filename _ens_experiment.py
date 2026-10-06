"""Temp: (a) extra features on the ensemble, (b) VotingRegressor sanity. Delete after use."""
import warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.inspection import permutation_importance

from src.features import MODEL_FEATURES
from src.train import make_pipeline, SEED, TEST_SIZE, _augment

master = pd.read_csv("data/processed/master_train.csv")
X = master[MODEL_FEATURES]
y = master["yield_tons_per_ha"]
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

def mlp(params=None):
    p = dict(hidden_layer_sizes=(64, 64), activation="relu", max_iter=400,
             early_stopping=True, random_state=SEED)
    if params:
        p.update(params)
    return make_pipeline(MLPRegressor(**p), scale=True)

def hgb(params=None):
    p = dict(random_state=SEED)
    if params:
        p.update(params)
    return make_pipeline(HistGradientBoostingRegressor(**p))

def fit_pred(pipe, Xa, ya, Xb):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(Xa, ya)
    return pipe.predict(Xb)

def rmse(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))

MLP_T = dict(hidden_layer_sizes=(64, 64), learning_rate_init=0.01, alpha=0.01)
HGB_T = dict(min_samples_leaf=10, max_leaf_nodes=31, max_iter=400,
             learning_rate=0.08, l2_regularization=0.1)

m1 = mlp(MLP_T); m2 = hgb(HGB_T)
p1 = fit_pred(m1, X_tr, y_tr, X_va)
p2 = fit_pred(m2, X_tr, y_tr, X_va)
print("holdout: mlp", round(rmse(y_va, p1), 4), "| hgb", round(rmse(y_va, p2), 4),
      "| ens0.5", round(rmse(y_va, 0.5 * p1 + 0.5 * p2), 4), flush=True)

# (b) VotingRegressor sanity: fitted members, weights -> same as manual average
ens = VotingRegressor([("mlp_neural_net", m1), ("hist_gradient_boosting", m2)],
                      weights=[0.5, 0.5])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ens.fit(X_tr, y_tr)
pv = ens.predict(X_va)
manual = 0.5 * p1 + 0.5 * p2
print("VotingRegressor vs manual avg max abs diff:", float(np.max(np.abs(pv - manual))),
      "| rmse", round(rmse(y_va, pv), 4), flush=True)

# permutation importance on the ensemble (n_repeats=3 for speed)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pi = permutation_importance(ens, X_va, y_va, n_repeats=3, random_state=SEED,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
top = pd.DataFrame({"f": MODEL_FEATURES, "imp": pi.importances_mean}) \
        .sort_values("imp", ascending=False).head(5)
print("permutation_importance OK; top5:")
print(top.to_string(index=False), flush=True)

# (a) D8-style extra features on the ENSEMBLE
extra = sorted(set(MODEL_FEATURES) | {"rain_ratio", "fert_x_soil"})
Xe_tr = _augment(X_tr)[extra]
Xe_va = _augment(X_va)[extra]
e1 = mlp(MLP_T); e2 = hgb(HGB_T)
q1 = fit_pred(e1, Xe_tr, y_tr, Xe_va)
q2 = fit_pred(e2, Xe_tr, y_tr, Xe_va)
base = rmse(y_va, 0.5 * p1 + 0.5 * p2)
extra_rmse = rmse(y_va, 0.5 * q1 + 0.5 * q2)
print(f"ensemble base {base:.4f} -> +extra_interactions {extra_rmse:.4f} "
      f"({'better' if extra_rmse < base else 'worse'}, "
      f"{100 * (extra_rmse - base) / base:+.1f}%)", flush=True)
