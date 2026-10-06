"""Growing-season weather join and feature engineering (Deliverable A6).

Design notes
------------
* Growing-season window = planting month + the following three months
  (the FAQ's suggested starting point, documented in report A3).
* Weather is aggregated to one row per (region, survey_year, planting_month)
  so the plot -> weather join is strictly **many-to-one** and can never change
  the plot row count.
* Months that are absent from the weather table are filled from that region's
  month climatology, and ``wx_n_months_observed`` keeps a record of how many of
  the four window months were actually observed (used in the A4 join audit).
* Price is joined for **analysis and the demo only** - it is never part of
  ``MODEL_FEATURES`` (price does not cause yield; feeding it in would be
  leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.cleaning import CROPS, MONTHS, REGIONS, TARGET

SEASON_MONTHS = 3  # planting month + following three months

# --------------------------------------------------------------------------- #
# Feature contract
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    "altitude_m",
    "rainfall_mm_season",
    "farm_size_ha",
    "fertilizer_kg_per_ha",
    "improved_seed_used",
    "pest_disease_flag",
    "soil_quality_index",
    "labor_days_per_ha",
    "distance_to_market_km",
    "survey_year",
    "planting_month_num",
    "fert_x_seed",
    "rain_gap_mm",
    "wx_season_mean_temp_c",
    "wx_season_rain_total_mm",
    "wx_extreme_heat_days",
    "wx_temp_anomaly_c",
    "wx_n_months_observed",
]
CATEGORICAL_FEATURES = ["region", "crop_type", "planting_month"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
WEATHER_FEATURES = [
    "wx_season_mean_temp_c",
    "wx_season_rain_total_mm",
    "wx_extreme_heat_days",
    "wx_temp_anomaly_c",
    "wx_n_months_observed",
    "rain_gap_mm",
]
ANALYSIS_ONLY = ["price_birr_per_quintal"]


def season_window(year: int, month: str, k: int = SEASON_MONTHS) -> list[tuple[int, str]]:
    """Calendar months covered by the season: planting month + next `k` months."""
    i = MONTHS.index(month)
    out = []
    for j in range(k + 1):
        idx = i + j
        out.append((int(year) + idx // 12, MONTHS[idx % 12]))
    return out


# --------------------------------------------------------------------------- #
# Weather -> one row per (region, year, planting month)
# --------------------------------------------------------------------------- #
def build_season_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly weather into growing-season features.

    Parameters
    ----------
    weather : cleaned monthly table (unique on region, year, month). Cells that
        are absent from the table are *not* inserted, so
        ``wx_n_months_observed`` reports the raw coverage.

    Returns
    -------
    DataFrame keyed by (region, survey_year, planting_month).
    """
    clim = (weather.groupby(["region", "month"], as_index=False)
            .agg(clim_temp=("avg_temp_c", "mean"),
                 clim_rain=("monthly_rainfall_mm", "mean"),
                 clim_heat=("extreme_heat_days", "mean")))
    clim_ix = clim.set_index(["region", "month"])

    cell = weather.set_index(["region", "year", "month"])

    rows = []
    for region in REGIONS:
        for year in sorted(weather["year"].unique()):
            for pm in MONTHS:
                win = season_window(year, pm)
                temps, rains, heats = [], [], []
                n_obs = 0
                clim_t, clim_r = [], []
                for (y, m) in win:
                    ct = float(clim_ix.loc[(region, m), "clim_temp"])
                    cr = float(clim_ix.loc[(region, m), "clim_rain"])
                    clim_t.append(ct)
                    clim_r.append(cr)
                    if (region, y, m) in cell.index:
                        r = cell.loc[(region, y, m)]
                        n_obs += 1
                        temps.append(float(r["avg_temp_c"]))
                        rains.append(float(r["monthly_rainfall_mm"]))
                        heats.append(float(r["extreme_heat_days"]))
                    else:  # month absent from the weather table -> climatology
                        temps.append(ct)
                        rains.append(cr)
                        clim_row = clim_ix.loc[(region, m)]
                        heats.append(float(clim_row["clim_heat"]))
                mean_temp = float(np.mean(temps))
                rows.append({
                    "region": region,
                    "survey_year": int(year),
                    "planting_month": pm,
                    "wx_season_mean_temp_c": round(mean_temp, 3),
                    "wx_season_rain_total_mm": round(float(np.sum(rains)), 1),
                    "wx_extreme_heat_days": int(np.sum(heats)),
                    "wx_n_months_observed": int(n_obs),
                    "wx_temp_anomaly_c": round(mean_temp - float(np.mean(clim_t)), 3),
                    "wx_rain_anomaly_mm": round(float(np.sum(rains) - np.sum(clim_r)), 1),
                })
    out = pd.DataFrame(rows)
    return out


# --------------------------------------------------------------------------- #
# Plot-level engineered features
# --------------------------------------------------------------------------- #
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that need no external table (interactions, month encoding)."""
    d = df.copy()
    d["planting_month_num"] = d["planting_month"].map(
        {m: i + 1 for i, m in enumerate(MONTHS)}).astype("int64")
    d["fert_x_seed"] = (d["fertilizer_kg_per_ha"] * d["improved_seed_used"]).round(4)
    if "wx_season_rain_total_mm" in d.columns:
        d["rain_gap_mm"] = (d["rainfall_mm_season"]
                            - d["wx_season_rain_total_mm"]).round(2)
    return d


def join_weather(plots: pd.DataFrame, season_weather: pd.DataFrame) -> pd.DataFrame:
    """Many-to-one join of plots onto season-level weather features."""
    n_before = len(plots)
    out = plots.merge(
        season_weather,
        on=["region", "survey_year", "planting_month"],
        how="left",
        validate="many_to_one",
    )
    assert len(out) == n_before, "weather join changed the plot row count"
    return out


def join_prices(plots: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Many-to-one join of prices (analysis / demo only, never a model feature)."""
    n_before = len(plots)
    out = plots.merge(
        prices.rename(columns={"year": "survey_year"}),
        on=["region", "crop_type", "survey_year"],
        how="left",
        validate="many_to_one",
    )
    assert len(out) == n_before, "price join changed the plot row count"
    return out


def revenue_per_ha(yield_tons_per_ha: pd.Series, price_birr_per_quintal: pd.Series):
    """Yield (t/ha) x 10 quintals per ton x birr per quintal -> birr per hectare."""
    return yield_tons_per_ha * 10.0 * price_birr_per_quintal


# --------------------------------------------------------------------------- #
# Data dictionary for the master tables (A8)
# --------------------------------------------------------------------------- #
_MASTER_SPEC: list[tuple[str, str, str, str, str]] = [
    # column, source file, description, derivation, in model?
    ("plot_id", "crop_yield_*.csv", "Unique plot ID / join key for scoring",
     "as given", "no"),
    ("region", "crop_yield_*.csv + regional_weather.csv",
     "Ethiopian region of the plot", "label standardisation (alias map)", "yes"),
    ("crop_type", "crop_yield_*.csv + market_prices.csv", "Crop grown",
     "label standardisation (alias map)", "yes"),
    ("survey_year", "crop_yield_*.csv", "Survey / season year 2021-2024",
     "as given, integer", "yes"),
    ("planting_month", "crop_yield_*.csv", "Month the crop was planted",
     "label standardisation", "yes"),
    ("altitude_m", "crop_yield_*.csv", "Plot elevation, metres above sea level",
     "as given", "yes"),
    ("rainfall_mm_season", "crop_yield_*.csv",
     "Plot self-reported growing-season rainfall, mm",
     "median-imputed (train median); capped at train p99.5", "yes"),
    ("farm_size_ha", "crop_yield_*.csv", "Farm size, hectares",
     "capped at train p99.5", "yes"),
    ("fertilizer_kg_per_ha", "crop_yield_*.csv", "Fertilizer applied, kg/ha",
     "median-imputed; capped at train p99.5", "yes"),
    ("improved_seed_used", "crop_yield_*.csv", "Certified improved seed (0/1)",
     "as given, integer", "yes"),
    ("pest_disease_flag", "crop_yield_*.csv", "Pest/disease pressure (0/1)",
     "-999 sentinel -> missing -> train mode", "yes"),
    ("soil_quality_index", "crop_yield_*.csv", "Composite soil quality (0-1)",
     "median-imputed", "yes"),
    ("labor_days_per_ha", "crop_yield_*.csv", "Labor-days per hectare",
     "-999 sentinel -> missing -> train median; capped at train p99.5", "yes"),
    ("distance_to_market_km", "crop_yield_*.csv",
     "Distance to nearest market, km", "capped at train p99.5", "yes"),
    ("yield_tons_per_ha", "crop_yield_train.csv",
     "TRUE yield, tons per hectare (train only)", "as given (target)", "target"),
    ("wx_season_mean_temp_c", "regional_weather.csv",
     "Mean temperature over the growing-season window, degC",
     "mean of planting month + 3 following months for the plot's region/year",
     "yes (weather)"),
    ("wx_season_rain_total_mm", "regional_weather.csv",
     "Station rainfall total over the growing-season window, mm",
     "sum of the same 4 monthly readings", "yes (weather)"),
    ("wx_extreme_heat_days", "regional_weather.csv",
     "Extreme-heat days during the window",
     "sum of extreme_heat_days over the 4 window months", "yes (weather)"),
    ("wx_temp_anomaly_c", "regional_weather.csv",
     "Window temperature minus the region's own typical window temperature, degC",
     "wx_season_mean_temp_c - mean of the same calendar months for that region "
     "across all years", "yes (weather)"),
    ("wx_n_months_observed", "regional_weather.csv",
     "How many of the 4 window months exist in the raw weather table (0-4)",
     "count of observed cells before climatology filling", "yes (weather)"),
    ("wx_rain_anomaly_mm", "regional_weather.csv",
     "Window rainfall minus the region's typical window rainfall, mm",
     "wx_season_rain_total_mm - regional climatology for those months",
     "no (reported in analysis)"),
    ("planting_month_num", "crop_yield_*.csv", "Planting month as 1-12 ordinal",
     "map from planting_month", "yes"),
    ("fert_x_seed", "crop_yield_*.csv",
     "Interaction: fertilizer x improved seed",
     "fertilizer_kg_per_ha * improved_seed_used", "yes"),
    ("rain_gap_mm", "crop_yield_*.csv + regional_weather.csv",
     "Plot-reported rainfall minus station season rainfall, mm",
     "rainfall_mm_season - wx_season_rain_total_mm", "yes (weather)"),
    ("price_birr_per_quintal", "market_prices.csv",
     "Market price, birr per quintal (ANALYSIS ONLY - never a model feature)",
     "unit-corrected (x100 where recorded per kg) + missing filled", "no"),
]


def build_data_dictionary(master: pd.DataFrame) -> pd.DataFrame:
    """One row per master column: type, source, description, derivation."""
    rows = []
    known = {spec[0]: spec for spec in _MASTER_SPEC}
    for col in master.columns:
        if col in known:
            _, source, desc, deriv, model = known[col]
        else:
            source, desc, deriv, model = "derived", "engineered feature", "see notebook 01", "yes"
        rows.append({
            "column": col,
            "dtype": str(master[col].dtype),
            "source_file": source,
            "description": desc,
            "derivation": deriv,
            "used_as_model_feature": model,
        })
    return pd.DataFrame(rows)
