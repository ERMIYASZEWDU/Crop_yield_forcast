"""Raw -> analysis-ready cleaning for the Qiyas Crop-Yield Challenge.

Two kinds of work live in this module:

1. Deterministic label repair that applies identically to every table
   (case / whitespace / abbreviation aliases, duplicate keys, unit mix-ups,
   -999 sentinels, impossible values).
2. Statistics that are **fit on the training file only** (imputation values
   and outlier caps) and then applied unchanged to the test file, per Rule 6.
   Nothing in this module ever reads the test file while fitting.

Every fix is recorded in a :class:`CleaningLog` row so notebook 01 can print
the A1 cleaning log directly from code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths (always relative to the project root - Rule 6.3)
# --------------------------------------------------------------------------- #
RAW_FILES = {
    "train": "data/raw/crop_yield_train.csv",
    "test": "data/raw/crop_yield_leaderboard_test.csv",
    "prices": "data/raw/market_prices.csv",
    "weather": "data/raw/regional_weather.csv",
    "template": "data/raw/submission_template.csv",
}

TARGET = "yield_tons_per_ha"
PLOT_ID = "plot_id"

# --------------------------------------------------------------------------- #
# Canonical vocabularies (the one set every table must end up with)
# --------------------------------------------------------------------------- #
REGIONS = ["Amhara", "Oromia", "SNNPR", "Somali", "Tigray"]
CROPS = ["barley", "maize", "sorghum", "teff", "wheat"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_REGION_ALIASES = {
    "amhara": "Amhara", "amh": "Amhara",
    "oromia": "Oromia", "oro": "Oromia",
    "snnpr": "SNNPR", "snnp": "SNNPR",
    "somali": "Somali", "som": "Somali",
    "tigray": "Tigray", "tig": "Tigray",
}
_CROP_ALIASES = {c: c for c in CROPS}
_CROP_ALIASES["tef"] = "teff"  # singular spelling found in the raw exports
_MONTH_ALIASES = {m.lower(): m for m in MONTHS}

SENTINEL = -999          # "missing" marker used in the raw exports
CAP_QUANTILE = 0.995     # outlier cap, fitted on train

IMPUTE_MEDIAN = [
    "rainfall_mm_season",
    "fertilizer_kg_per_ha",
    "soil_quality_index",
    "labor_days_per_ha",
]
IMPUTE_MODE = ["pest_disease_flag"]
CAP_COLS = [
    "rainfall_mm_season",
    "farm_size_ha",
    "fertilizer_kg_per_ha",
    "labor_days_per_ha",
    "distance_to_market_km",
]


def read_raw(name: str) -> pd.DataFrame:
    """Read one of the untouched raw exports by key of :data:`RAW_FILES`."""
    return pd.read_csv(RAW_FILES[name])


# --------------------------------------------------------------------------- #
# Label standardisation
# --------------------------------------------------------------------------- #
def standardize_region(s: pd.Series) -> pd.Series:
    """Strip, lowercase and map every alias (incl. AMH/ORO/...) to 5 regions."""
    key = s.astype("string").str.strip().str.lower()
    out = key.map(_REGION_ALIASES)
    if out.isna().any():
        bad = sorted(key[out.isna()].unique().tolist())
        raise ValueError(f"unknown region labels: {bad}")
    return out.astype("string")


def standardize_crop(s: pd.Series) -> pd.Series:
    key = s.astype("string").str.strip().str.lower()
    out = key.map(_CROP_ALIASES)
    if out.isna().any():
        bad = sorted(key[out.isna()].unique().tolist())
        raise ValueError(f"unknown crop labels: {bad}")
    return out.astype("string")


def standardize_month(s: pd.Series) -> pd.Series:
    key = s.astype("string").str.strip().str.lower()
    out = key.map(_MONTH_ALIASES)
    if out.isna().any():
        bad = sorted(key[out.isna()].unique().tolist())
        raise ValueError(f"unknown month labels: {bad}")
    return out.astype("string")


# --------------------------------------------------------------------------- #
# Cleaning log (Deliverable A1)
# --------------------------------------------------------------------------- #
LOG_COLUMNS = [
    "file", "columns", "issue_type", "rows_affected",
    "pct_of_rows", "fix_applied", "why",
]


class CleaningLog:
    """One row per issue: file, columns, issue, count, %, fix, why."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, file: str, columns: str, issue_type: str, n: int,
            total: int, fix: str, why: str) -> None:
        if n == 0:
            return
        self.rows.append({
            "file": file,
            "columns": columns,
            "issue_type": issue_type,
            "rows_affected": int(n),
            "pct_of_rows": round(100.0 * n / total, 2) if total else 0.0,
            "fix_applied": fix,
            "why": why,
        })

    def extend(self, other: "CleaningLog") -> "CleaningLog":
        self.rows.extend(other.rows)
        return self

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=LOG_COLUMNS)

    def __len__(self) -> int:
        return len(self.rows)


# --------------------------------------------------------------------------- #
# Train-fitted imputation values and outlier caps (Rule 6)
# --------------------------------------------------------------------------- #
def fit_plot_stats(df: pd.DataFrame, quantile: float = CAP_QUANTILE) -> dict:
    """Learn medians / modes / outlier caps from a table (call on TRAIN only)."""
    d = df.copy()
    for c in d.select_dtypes(include="number").columns:
        d.loc[d[c] <= SENTINEL, c] = np.nan
    return {
        "median": {c: float(d[c].median()) for c in IMPUTE_MEDIAN},
        "mode": {c: float(d[c].mode().iloc[0]) for c in IMPUTE_MODE},
        "caps": {c: float(d[c].quantile(quantile)) for c in CAP_COLS},
        "quantile": quantile,
        "fit_rows": int(len(df)),
    }


# --------------------------------------------------------------------------- #
# Plot-level table (train and test share this exact code path)
# --------------------------------------------------------------------------- #
def clean_plots(df: pd.DataFrame, source: str, stats: dict | None = None):
    """Clean one plot table.

    Parameters
    ----------
    df : raw plot table
    source : file name used in the cleaning log
    stats : statistics fit on train. ``None`` means "fit on this table" and
        must only ever be passed for the train file (Rule 6).

    Returns
    -------
    (clean_df, stats, log)
    """
    d = df.copy()
    total = len(d)
    log = CleaningLog()
    file = source

    # 1 ---- key standardisation ------------------------------------------------
    for col, fn, issue in [
        ("region", standardize_region,
         "inconsistent region labels (upper/lower case, trailing spaces, abbreviations)"),
        ("crop_type", standardize_crop,
         "inconsistent crop labels (upper/lower case, padded with spaces)"),
        ("planting_month", standardize_month,
         "inconsistent planting-month labels (case / whitespace)"),
    ]:
        canon = fn(d[col])
        n = int((d[col].astype("string") != canon).sum())
        if n:
            log.add(file, col, issue, n, total,
                    "str.strip + case fold + alias map to the canonical label set",
                    "region/crop/month are join keys for weather and price; "
                    "spelling variants would silently drop matches.")
        d[col] = canon

    # 2 ---- missing values already present in the export -----------------------
    missing_counts = {}
    for c in d.columns:
        if c in (PLOT_ID, TARGET):
            continue
        n = int(d[c].isna().sum())
        if n:
            missing_counts[c] = n

    # 3 ---- -999 sentinels -----------------------------------------------------
    sentinel_counts = {}
    for c in d.select_dtypes(include="number").columns:
        if c == TARGET:
            mask = d[c] <= SENTINEL
        else:
            mask = d[c] <= SENTINEL
        n = int(mask.sum())
        if n:
            d[c] = d[c].astype("float64")
            d.loc[mask, c] = np.nan
            sentinel_counts[c] = n
            log.add(file, c, f"missing-value sentinel {SENTINEL} stored as a real number",
                    n, total,
                    f"replaced {SENTINEL} with NaN, then imputed with the train-fitted value",
                    f"as written {SENTINEL} would be read as a measurement of "
                    f"{SENTINEL} and destroy means, correlations and the model.")

    for c, n in missing_counts.items():
        log.add(file, c, "blank / missing values in the raw export", n, total,
                "imputed with the train-fitted median (or mode for binary flags)",
                "rows must be kept - the leaderboard scores every test plot, "
                "so deleting rows is not an option.")

    # 4 ---- impossible values --------------------------------------------------
    if "soil_quality_index" in d.columns:
        n = int(((d["soil_quality_index"] < 0) | (d["soil_quality_index"] > 1)).sum())
        if n:
            d.loc[(d["soil_quality_index"] < 0) | (d["soil_quality_index"] > 1),
                  "soil_quality_index"] = np.nan
            log.add(file, "soil_quality_index", "value outside its documented 0-1 range",
                    n, total, "set to NaN and imputed with the train median",
                    "an out-of-range score is a data-entry error, not a super-soil.")
    for col, lo in [("farm_size_ha", 0.0), ("altitude_m", 0.0),
                    ("distance_to_market_km", 0.0)]:
        if col in d.columns:
            n = int((d[col] <= lo).sum())
            if n:
                d.loc[d[col] <= lo, col] = np.nan
                log.add(file, col, "non-positive physically impossible value", n, total,
                        "set to NaN and imputed with the train median",
                        f"{col} must be > 0 for a real plot.")
    for col in ["improved_seed_used", "pest_disease_flag"]:
        if col in d.columns:
            n = int((~d[col].isin([0, 1]) & d[col].notna()).sum())
            if n:
                d.loc[~d[col].isin([0, 1]) & d[col].notna(), col] = np.nan
                log.add(file, col, "value outside the documented 0/1 domain", n, total,
                        "set to NaN and imputed with the train mode",
                        "binary flags only make sense as 0 or 1.")

    # 5 ---- impute with train-fitted statistics (Rule 6) ------------------------
    if stats is None:
        stats = fit_plot_stats(d)
    for c, value in {**stats["median"], **stats["mode"]}.items():
        if c not in d.columns:
            continue
        n = int(d[c].isna().sum())
        if n:
            d[c] = d[c].fillna(value)
    d["pest_disease_flag"] = d["pest_disease_flag"].astype("int64")
    d["improved_seed_used"] = d["improved_seed_used"].astype("int64")
    d["survey_year"] = d["survey_year"].astype("int64")

    # 6 ---- winsorise extreme but non-impossible values -------------------------
    for c, cap in stats["caps"].items():
        if c not in d.columns:
            continue
        n = int((d[c] > cap).sum())
        if n:
            d.loc[d[c] > cap, c] = cap
            log.add(file, c, "implausible extreme value (heavy tail)",
                    n, total,
                    f"capped at the train {stats['quantile']:.1%} quantile = {cap:.2f}",
                    "a handful of 3,800 mm / 900 kg/ha / 60 ha reports are survey "
                    "errors; capping keeps the row (we must predict every plot) "
                    "while stopping it from dominating splits.")

    return d, stats, log


# --------------------------------------------------------------------------- #
# Regional weather table
# --------------------------------------------------------------------------- #
def clean_weather(df: pd.DataFrame):
    """Clean the weather export: labels, duplicate keys, missing readings.

    Returns (clean_df, log).  The clean table is unique on
    (region, year, month) and has no missing readings.
    """
    d = df.copy()
    total = len(d)
    log = CleaningLog()
    file = "regional_weather.csv"

    canon = standardize_region(d["region"])
    n = int((d["region"].astype("string") != canon).sum())
    if n:
        log.add(file, "region", "inconsistent region labels "
                "(case, trailing spaces, 3-letter abbreviations such as AMH/ORO/SNNP)",
                n, total, "alias map to the canonical 5 region names",
                "the weather join keys on region; unmapped labels would silently "
                "lose a whole region's weather.")
    d["region"] = canon
    d["month"] = standardize_month(d["month"])

    # duplicate (region, year, month) readings
    key = ["region", "year", "month"]
    n_dup = int(d.duplicated(subset=key, keep=False).sum())
    n_keys_lost = int(d.duplicated(subset=key).sum())
    if n_dup:
        before = len(d)
        d = (d.groupby(key, as_index=False, sort=False)
               .agg({"avg_temp_c": "mean",
                     "monthly_rainfall_mm": "mean",
                     "extreme_heat_days": "max"})
               )
        log.add(file, ", ".join(key), "duplicate key rows "
                "(same region-year-month reported twice with slightly different values)",
                n_dup, total,
                f"averaged the duplicate readings, removing {before - len(d)} rows; "
                f"key is now unique",
                "a many-to-one join from plot to weather would have duplicated "
                "plots and inflated the row count if left in.")

    # missing readings -> region-month climatology (weather table only)
    for c, n_miss in [("avg_temp_c", int(d["avg_temp_c"].isna().sum())),
                      ("monthly_rainfall_mm", int(d["monthly_rainfall_mm"].isna().sum()))]:
        if n_miss:
            clim = d.groupby(["region", "month"], as_index=False)[c].mean()
            clim.columns = ["region", "month", "_fill"]
            d = d.merge(clim, on=["region", "month"], how="left")
            d[c] = d[c].fillna(d["_fill"])
            d = d.drop(columns="_fill")
            log.add(file, c, "missing monthly reading", n_miss, total,
                    "filled with that region's month climatology (mean over the "
                    "available years of the same table)",
                    "every region-year-month must exist so growing-season "
                    "aggregates are comparable across plots.")

    # sanity range check (records only if something is out of range)
    n = int(((d["avg_temp_c"] < -10) | (d["avg_temp_c"] > 45)
             | (d["monthly_rainfall_mm"] < 0)).sum())
    log.add(file, "avg_temp_c, monthly_rainfall_mm",
            "readings outside physical range [-10C..45C] / negative rainfall",
            n, total, "none found - values already inside physical range",
            "guards against unit or transcription errors in the raw export.")

    return d.reset_index(drop=True), log


# --------------------------------------------------------------------------- #
# Market price table
# --------------------------------------------------------------------------- #
def clean_prices(df: pd.DataFrame):
    """Clean the price export: labels, unit mix-up, missing prices.

    Returns (clean_df, log).  ``price_birr_per_quintal`` is on the correct
    birr-per-quintal scale for every row afterwards.
    """
    d = df.copy()
    total = len(d)
    log = CleaningLog()
    file = "market_prices.csv"

    canon = standardize_crop(d["crop_type"])
    n = int((d["crop_type"].astype("string") != canon).sum())
    if n:
        log.add(file, "crop_type", "inconsistent crop labels (upper/lower case, padded)",
                n, total, "str.strip + case fold to the canonical 5 crop labels",
                "crop_type is a join key with the plot table; variants would "
                "break the price merge for those rows.")
    d["crop_type"] = canon

    canon_r = standardize_region(d["region"])
    n = int((d["region"].astype("string") != canon_r).sum())
    if n:
        log.add(file, "region", "inconsistent region labels", n, total,
                "alias map to the canonical 5 region names",
                "region is a join key with the plot table.")
    d["region"] = canon_r

    # unit mix-up: rows recorded in birr per *kilogram* instead of per quintal
    price = pd.to_numeric(d["price_birr_per_quintal"], errors="coerce")
    wrong = price.notna() & (price < 100)
    n_wrong = int(wrong.sum())
    if n_wrong:
        d.loc[wrong, "price_birr_per_quintal"] = price[wrong] * 100.0
        log.add(file, "price_birr_per_quintal",
                "unit mix-up: price recorded in birr per kg (~34-49) instead of "
                "birr per quintal (~3,400-4,900)",
                n_wrong, total, "multiplied by 100 (1 quintal = 100 kg)",
                "as recorded these rows would understate revenue 100-fold and "
                "break every price trend in the analysis.")

    # missing prices -> crop x region mean over the available years
    n_miss = int(price.isna().sum() | (d["price_birr_per_quintal"].isna()).sum())
    n_miss = int(d["price_birr_per_quintal"].isna().sum())
    if n_miss:
        by_cr = d.groupby(["crop_type", "region"])["price_birr_per_quintal"].transform("mean")
        by_c = d.groupby("crop_type")["price_birr_per_quintal"].transform("mean")
        d["price_birr_per_quintal"] = d["price_birr_per_quintal"].fillna(by_cr).fillna(by_c)
        log.add(file, "price_birr_per_quintal", "missing price", n_miss, total,
                "filled with the mean of the other years for the same crop x region "
                "(falling back to the crop mean)",
                "revenue per hectare and the demo need a price for every "
                "crop-region-year combination.")
    d["year"] = d["year"].astype("int64")
    return d, log
