"""Yield & revenue demo - Team Adwa, Qiyas / IADE Crop-Yield Challenge.

Run from the project root:

    streamlit run app/app.py

The user never types a weather or price number: the app looks up the growing-season
weather, the plot rainfall reference and the market price from the cleaned tables
bundled in app/assets/, then applies the trained pipeline (preprocessing + model)
exactly as it was fit on train.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"

REGIONS = ["Amhara", "Oromia", "SNNPR", "Somali", "Tigray"]
CROPS = ["barley", "maize", "sorghum", "teff", "wheat"]
YEARS = [2021, 2022, 2023, 2024]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PLANTING_MONTHS = ["Feb", "Mar", "Jun", "Jul", "Aug"]
WORLDWIDE_RAIN_FALLBACK = 850.0  # train median, used only if a lookup cell is missing

st.set_page_config(page_title="Ethiopian plot yield predictor", page_icon="🌾",
                   layout="wide")


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    path = ASSETS / "final_model.joblib"
    if not path.exists():
        st.error("Model file missing - run notebooks/04_modeling_and_evaluation.ipynb first.")
        st.stop()
    return joblib.load(path)


@st.cache_data
def load_tables():
    season = pd.read_csv(ASSETS / "season_weather.csv")
    prices = pd.read_csv(ASSETS / "prices_clean.csv")
    rain = pd.read_csv(ASSETS / "plot_rain_lookup.csv")
    bench = pd.read_csv(ASSETS / "yield_benchmarks.csv")
    return season, prices, rain, bench


bundle = load_model()
PIPE, FEATURES = bundle["pipeline"], bundle["features"]
season, prices, rain_lookup, bench = load_tables()


def lookup_weather(region, year, month):
    """Season weather features for one (region, year, planting month)."""
    hit = season[(season.region == region) & (season.survey_year == year)
                 & (season.planting_month == month)]
    if hit.empty:
        return None
    return hit.iloc[0]


def lookup_price(region, crop, year):
    hit = prices[(prices.region == region) & (prices.crop_type == crop)
                 & (prices.year == year)]
    if hit.empty or pd.isna(hit.iloc[0].price_birr_per_quintal):
        return None
    return float(hit.iloc[0].price_birr_per_quintal)


def lookup_plot_rain(region, year, month):
    hit = rain_lookup[(rain_lookup.region == region) & (rain_lookup.survey_year == year)
                      & (rain_lookup.planting_month == month)]
    if hit.empty:
        return WORLDWIDE_RAIN_FALLBACK
    return float(hit.iloc[0].plot_rain_median_mm)


def build_row(region, crop, year, month, altitude, farm_size, fertilizer,
              improved_seed, pest_flag, soil, labor, distance, rainfall):
    """One feature row exactly as the pipeline was trained to consume."""
    wx = lookup_weather(region, year, month)
    row = {
        "region": region,
        "crop_type": crop,
        "planting_month": month,
        "survey_year": int(year),
        "altitude_m": float(altitude),
        "farm_size_ha": float(farm_size),
        "fertilizer_kg_per_ha": float(fertilizer),
        "improved_seed_used": int(improved_seed),
        "pest_disease_flag": int(pest_flag),
        "soil_quality_index": float(soil),
        "labor_days_per_ha": float(labor),
        "distance_to_market_km": float(distance),
        "rainfall_mm_season": float(rainfall),
        "planting_month_num": MONTHS.index(month) + 1,
        "fert_x_seed": float(fertilizer) * int(improved_seed),
    }
    if wx is not None:
        row.update({k: float(wx[k]) for k in season.columns
                    if k.startswith("wx_")})
        row["rain_gap_mm"] = float(rainfall) - float(wx["wx_season_rain_total_mm"])
    return pd.DataFrame([row])[FEATURES], wx


def validate(region, crop, year, month, altitude, farm_size, fertilizer,
             soil, labor, distance):
    """Friendly messages for nonsensical inputs; returns (ok, warnings)."""
    notes = []
    if not (0 < altitude <= 4500):
        return False, ["Altitude should be between 0 and 4,500 m above sea level."]
    if not (0 < farm_size <= 200):
        return False, ["Farm size should be between 0 and 200 hectares."]
    if fertilizer < 0 or fertilizer > 2000:
        return False, ["Fertilizer must be between 0 and 2,000 kg/ha."]
    if not (0 <= soil <= 1):
        return False, ["Soil quality index must be between 0 and 1."]
    if labor < 0 or labor > 400:
        return False, ["Labor days per hectare must be between 0 and 400."]
    if distance < 0 or distance > 500:
        return False, ["Distance to market must be between 0 and 500 km."]
    if altitude > 3200:
        notes.append("Altitude above 3,200 m is outside the survey range - treat the "
                     "prediction as an extrapolation.")
    if farm_size > 60:
        notes.append("Farm sizes above 60 ha are rare in this survey.")
    return True, notes


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
st.title("Ethiopian smallholder plot: yield and revenue predictor 🌾")
st.caption("Team Adwa - Qiyas / IADE hackathon. You enter the plot; the app looks up the "
           "growing-season weather and the market price for you.")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1. Where and when")
    region = st.selectbox("Region", REGIONS, index=0)
    crop = st.selectbox("Crop type", CROPS, index=3)
    year = st.selectbox("Survey year", YEARS, index=3)
    month = st.selectbox("Planting month", PLANTING_MONTHS, index=4)

    st.subheader("2. The plot")
    altitude = st.number_input("Altitude [m]", min_value=0.0, max_value=4500.0,
                               value=1800.0, step=50.0)
    farm_size = st.number_input("Farm size [ha]", min_value=0.0, max_value=200.0,
                                value=1.2, step=0.1, format="%.2f")
    fertilizer = st.number_input("Fertilizer [kg/ha]", min_value=0.0, max_value=2000.0,
                                 value=40.0, step=5.0)
    improved_seed = st.checkbox("Improved (certified) seed used", value=False)
    pest_flag = st.checkbox("Pest / disease pressure observed", value=False)
    soil = st.slider("Soil quality index", 0.0, 1.0, 0.55, step=0.01)
    labor = st.number_input("Labor days per hectare", min_value=0.0, max_value=400.0,
                            value=45.0, step=1.0)
    distance = st.number_input("Distance to market [km]", min_value=0.0, max_value=500.0,
                               value=8.0, step=0.5)
    run = st.button("Predict yield and revenue", type="primary")

with right:
    if not run:
        st.subheader("3. Result")
        st.info("Set the plot details on the left and press **Predict yield and revenue**.")
        st.subheader("What the app does for you")
        st.markdown(
            "- reads the **growing-season weather** for this region, year and planting month "
            "(planting month + 3 months)\n"
            "- fills the plot's seasonal rainfall from the survey median for that region-season\n"
            "- reads the **market price** for this crop, region and year\n"
            "- runs the trained pipeline (same preprocessing as training) and converts yield "
            "into revenue")
    else:
        ok, notes = validate(region, crop, year, month, altitude, farm_size,
                             fertilizer, soil, labor, distance)
        wx = lookup_weather(region, year, month)
        price = lookup_price(region, crop, year)

        if not ok:
            st.error(notes[0])
        elif wx is None or price is None:
            st.error("No weather or price record for that combination - "
                     "please pick another region, crop or year.")
        else:
            rainfall = lookup_plot_rain(region, year, month)
            X, _ = build_row(region, crop, year, month, altitude, farm_size,
                             fertilizer, int(improved_seed), int(pest_flag),
                             soil, labor, distance, rainfall)
            pred = float(np.clip(PIPE.predict(X)[0], 0.0, None))
            revenue = pred * farm_size * 10.0 * price

            st.subheader("3. Result")
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted yield", f"{pred:.2f} t/ha")
            m2.metric("Estimated revenue for this plot",
                      f"{revenue:,.0f} birr")
            m3.metric("Revenue per hectare", f"{pred * 10 * price:,.0f} birr/ha")

            st.markdown(
                f"**Looked up for you:** season average "
                f"**{wx['wx_season_mean_temp_c']:.1f} °C**, season rainfall "
                f"**{wx['wx_season_rain_total_mm']:.0f} mm** "
                f"({int(wx['wx_n_months_observed'])}/4 months observed), "
                f"**{int(wx['wx_extreme_heat_days'])}** extreme-heat days, plot rainfall "
                f"reference **{rainfall:.0f} mm**, price **{price:,.0f} birr/quintal**.")
            st.caption(f"Revenue = {pred:.2f} t/ha x {farm_size:.2f} ha x 10 quintals/ton "
                       f"x {price:,.0f} birr/quintal. Price and weather come from the cleaned "
                       f"tables in app/assets - you never type them.")
            for n in notes:
                st.warning(n)

            st.subheader("4. How this plot compares")
            b = bench[(bench.region == region) & (bench.crop_type == crop)]
            if not b.empty:
                base = float(b.mean_yield_t_ha.iloc[0])
                comp = pd.DataFrame({
                    "group": [f"This plot (predicted)",
                              f"{region} x {crop} average"],
                    "yield_t_ha": [pred, base],
                })
                st.bar_chart(comp.set_index("group"))

            st.subheader("5. What-if: yield across fertilizer rates")
            rates = np.linspace(0, 200, 11)
            whatif = []
            for r in rates:
                Xw, _ = build_row(region, crop, year, month, altitude, farm_size,
                                  float(r), int(improved_seed), int(pest_flag),
                                  soil, labor, distance, rainfall)
                whatif.append(float(np.clip(PIPE.predict(Xw)[0], 0.0, None)))
            chart = pd.DataFrame({"fertilizer_kg_per_ha": rates,
                                  "predicted_yield_t_ha": whatif}).set_index(
                "fertilizer_kg_per_ha")
            st.line_chart(chart)
            st.caption(f"Your plot uses {fertilizer:.0f} kg/ha. The curve is the model's "
                       f"answer with everything else held fixed.")

st.divider()
st.caption("Model: "
           f"{bundle['model_name']} | {len(FEATURES)} features "
           f"({sum(1 for f in FEATURES if f.startswith('wx_') or f == 'rain_gap_mm')} "
           "weather-derived) | trained on the full train file with seed 42. "
           "Run `streamlit run app/app.py` from the project root.")
