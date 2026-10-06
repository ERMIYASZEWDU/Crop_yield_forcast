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
st.markdown(
    """
    <style>
    .block-container {max-width: 1440px; padding-top: 2.2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff 0%, #f4f8f2 100%);
        border: 1px solid #e1e9df;
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] {color: #526354;}
    div.stButton > button[kind="primaryFormSubmit"] {
        min-height: 3rem; border-radius: 10px; font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="padding:1.6rem 1.8rem; border-radius:18px;
                background:linear-gradient(115deg,#123b2b,#26734d);
                color:white; margin-bottom:1.4rem;">
      <div style="font-size:.78rem; letter-spacing:.12em; text-transform:uppercase;
                  opacity:.8;">TEAM ADWA · CROP INTELLIGENCE</div>
      <h1 style="color:white; margin:.45rem 0 .35rem;">A clearer view of your harvest</h1>
      <p style="font-size:1.05rem; margin:0; opacity:.9;">
        Estimate plot yield and revenue using your farm details, seasonal weather,
        and local crop prices.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

input_col, result_col = st.columns([0.92, 1.28], gap="large")

with input_col:
    st.subheader("🌱 Describe your plot")
    st.caption("Choose a location and enter the farm details. Weather and prices are looked up automatically.")

    with st.form("plot_prediction_form", border=True):
        st.markdown("#### Location and season")
        location_a, location_b = st.columns(2)
        with location_a:
            region = st.selectbox("Region", REGIONS, index=0)
            year = st.selectbox("Survey year", YEARS, index=3)
        with location_b:
            crop = st.selectbox("Crop", CROPS, index=3)
            month = st.selectbox("Planting month", PLANTING_MONTHS, index=4)

        st.markdown("#### Farm conditions")
        plot_a, plot_b = st.columns(2)
        with plot_a:
            altitude = st.number_input(
                "Altitude (m)", min_value=0.0, max_value=4500.0,
                value=1800.0, step=50.0,
            )
            farm_size = st.number_input(
                "Farm size (ha)", min_value=0.0, max_value=200.0,
                value=1.2, step=0.1, format="%.2f",
            )
            fertilizer = st.number_input(
                "Fertilizer (kg/ha)", min_value=0.0, max_value=2000.0,
                value=40.0, step=5.0,
            )
            soil = st.slider("Soil quality", 0.0, 1.0, 0.55, step=0.01)
        with plot_b:
            labor = st.number_input(
                "Labor (days/ha)", min_value=0.0, max_value=400.0,
                value=45.0, step=1.0,
            )
            distance = st.number_input(
                "Distance to market (km)", min_value=0.0, max_value=500.0,
                value=8.0, step=0.5,
            )
            improved_seed = st.checkbox("Improved seed used")
            pest_flag = st.checkbox("Pest or disease pressure observed")

        run = st.form_submit_button(
            "Estimate yield and revenue", type="primary", use_container_width=True,
        )

with result_col:
    st.subheader("📊 Your estimate")
    if not run:
        with st.container(border=True):
            st.markdown("### Ready when you are")
            st.write("Complete the plot form and select **Estimate yield and revenue** to see your results.")
            st.divider()
            preview = st.columns(3)
            preview[0].markdown("🌦️  \n**Season weather**  \nLooked up for your region and planting month")
            preview[1].markdown("🌾  \n**Yield estimate**  \nPredicted by the trained crop-yield model")
            preview[2].markdown("💰  \n**Revenue estimate**  \nBased on the matching market price")
        st.caption("Estimates are model outputs for planning, not guaranteed harvests or income.")
    else:
        ok, notes = validate(
            region, crop, year, month, altitude, farm_size,
            fertilizer, soil, labor, distance,
        )
        wx = lookup_weather(region, year, month)
        price = lookup_price(region, crop, year)

        if not ok:
            st.error(notes[0])
        elif wx is None or price is None:
            st.error(
                "No weather or price record is available for that selection. "
                "Try another region, crop, or year."
            )
        else:
            rainfall = lookup_plot_rain(region, year, month)
            X, _ = build_row(
                region, crop, year, month, altitude, farm_size,
                fertilizer, int(improved_seed), int(pest_flag),
                soil, labor, distance, rainfall,
            )
            pred = float(np.clip(PIPE.predict(X)[0], 0.0, None))
            revenue = pred * farm_size * 10.0 * price
            comparison = bench[
                (bench.region == region) & (bench.crop_type == crop)
            ]

            with st.container(border=True):
                metric_cols = st.columns(3)
                metric_cols[0].metric("Predicted yield", f"{pred:.2f} t/ha")
                metric_cols[1].metric("Plot revenue", f"{revenue:,.0f} birr")
                metric_cols[2].metric(
                    "Revenue per hectare", f"{pred * 10 * price:,.0f} birr/ha"
                )

                if not comparison.empty:
                    base = float(comparison.mean_yield_t_ha.iloc[0])
                    difference = pred - base
                    direction = "above" if difference >= 0 else "below"
                    st.info(
                        f"Compared with the {region} {crop} average "
                        f"({base:.2f} t/ha), this estimate is "
                        f"{abs(difference):.2f} t/ha {direction}."
                    )

                for note in notes:
                    st.warning(note)

                st.markdown("#### Seasonal conditions and market")
                context = st.columns(4)
                context[0].metric(
                    "Mean temperature", f"{wx['wx_season_mean_temp_c']:.1f} °C"
                )
                context[1].metric(
                    "Season rainfall", f"{wx['wx_season_rain_total_mm']:.0f} mm"
                )
                context[2].metric(
                    "Extreme-heat days", f"{int(wx['wx_extreme_heat_days'])}"
                )
                context[3].metric("Market price", f"{price:,.0f} birr/qtl")
                st.caption(
                    f"Weather coverage: {int(wx['wx_n_months_observed'])}/4 months · "
                    f"Plot rainfall reference: {rainfall:.0f} mm. "
                    f"Revenue uses 10 quintals per ton and the full {farm_size:.2f} ha."
                )

            overview_tab, scenario_tab, method_tab = st.tabs(
                ["Yield comparison", "Fertilizer scenario", "How it works"]
            )
            with overview_tab:
                if not comparison.empty:
                    comp = pd.DataFrame({
                        "Yield (t/ha)": [pred, base],
                    }, index=["This plot · predicted", f"{region} {crop} average"])
                    st.bar_chart(comp, color="#26734d")
                else:
                    st.info("No regional crop benchmark is available for comparison.")
            with scenario_tab:
                rates = np.linspace(0, 200, 11)
                whatif = []
                for rate in rates:
                    Xw, _ = build_row(
                        region, crop, year, month, altitude, farm_size,
                        float(rate), int(improved_seed), int(pest_flag),
                        soil, labor, distance, rainfall,
                    )
                    whatif.append(
                        float(np.clip(PIPE.predict(Xw)[0], 0.0, None))
                    )
                chart = pd.DataFrame(
                    {"Predicted yield (t/ha)": whatif},
                    index=pd.Index(rates, name="Fertilizer (kg/ha)"),
                )
                st.line_chart(chart, color="#26734d")
                st.caption(
                    f"Your current rate is {fertilizer:.0f} kg/ha. "
                    "The scenario varies fertilizer and holds other inputs fixed; "
                    "it is not a fertilizer recommendation."
                )
            with method_tab:
                st.markdown(
                    "- Weather is looked up for the selected region, year, and "
                    "four-month growing-season window.\n"
                    "- The matching crop, region, and year market price is used "
                    "to estimate gross revenue.\n"
                    "- The trained model predicts yield; revenue is calculated "
                    "from predicted yield × farm area × price."
                )
                st.caption(
                    f"Revenue calculation: {pred:.2f} t/ha × {farm_size:.2f} ha "
                    f"× 10 quintals/ton × {price:,.0f} birr/quintal."
                )

            st.caption(
                "Planning estimate only: actual harvest and market prices can vary."
            )

st.divider()
st.caption("Model: "
           f"{bundle['model_name']} | {len(FEATURES)} features "
           f"({sum(1 for f in FEATURES if f.startswith('wx_') or f == 'rain_gap_mm')} "
           "weather-derived) | trained on the full train file with seed 42. "
           "Run `streamlit run app/app.py` from the project root.")
