import streamlit as st
# AQI Predictor v2
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌍",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    saved = pickle.load(open("model.pkl", "rb"))
    if isinstance(saved, dict):
        return saved["model"], saved["features"]
    else:
        return saved, ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

model, FEATURES = load_model()

# ── AQI helpers ───────────────────────────────────────────────────────────────

AQI_CATEGORIES = [
    (50,  "Good",       "🟢", "#2e7d32",
     "Air quality is satisfactory. Enjoy outdoor activities freely."),
    (100, "Moderate",   "🟡", "#f9a825",
     "Acceptable air quality. Unusually sensitive people should limit prolonged outdoor exertion."),
    (200, "Poor",       "🟠", "#e65100",
     "Members of sensitive groups may experience health effects. Wear a mask outdoors if possible."),
    (300, "Very Poor",  "🔴", "#b71c1c",
     "Everyone may begin to experience health effects. Avoid prolonged outdoor activity."),
    (500, "Severe",     "⚫", "#212121",
     "Health warnings of emergency conditions. Stay indoors, keep windows closed."),
]

def get_category(aqi):
    for limit, label, icon, color, advice in AQI_CATEGORIES:
        if aqi <= limit:
            return label, icon, color, advice
    return "Severe", "⚫", "#212121", "Hazardous. Stay indoors."


# Pollutant input constraints  [min, max, default, unit, help_text]
POLLUTANT_CONFIG = {
    "PM2.5": (0.0, 500.0, 60.0,  "µg/m³", "Fine particulate matter (≤2.5 µm). Major source: vehicles, fires."),
    "PM10":  (0.0, 600.0, 100.0, "µg/m³", "Coarse particulate matter (≤10 µm). Major source: dust, construction."),
    "NO2":   (0.0, 400.0, 40.0,  "µg/m³", "Nitrogen dioxide. Major source: traffic, power plants."),
    "SO2":   (0.0, 800.0, 20.0,  "µg/m³", "Sulphur dioxide. Major source: coal burning, industries."),
    "CO":    (0.0,  50.0,  1.0,  "mg/m³", "Carbon monoxide. Major source: incomplete combustion, vehicles."),
    "O3":    (0.0, 300.0, 50.0,  "µg/m³", "Ground-level ozone. Formed from NOx + VOCs in sunlight."),
}

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🌍 Air Quality Index (AQI) Predictor")
st.markdown(
    "Enter pollutant concentrations below to predict the AQI and get a health advisory."
)
st.divider()

# ── Input section ─────────────────────────────────────────────────────────────

st.subheader("Pollutant Inputs")

col1, col2 = st.columns(2)
inputs = {}
validation_errors = []

for i, feature in enumerate(FEATURES):
    if feature not in POLLUTANT_CONFIG:
        continue
    min_val, max_val, default, unit, help_text = POLLUTANT_CONFIG[feature]
    col = col1 if i % 2 == 0 else col2

    with col:
        val = st.number_input(
            label=f"{feature} ({unit})",
            min_value=min_val,
            max_value=max_val,
            value=default,
            step=0.1 if feature == "CO" else 1.0,
            help=help_text,
            key=feature,
        )
        inputs[feature] = val

        if val == min_val and feature != "CO":
            st.caption(f"⚠️ 0 is unusually low for {feature}.")

# ── Cross-field validation ────────────────────────────────────────────────────

if "PM2.5" in inputs and "PM10" in inputs:
    if inputs["PM2.5"] > inputs["PM10"]:
        validation_errors.append("PM2.5 should not exceed PM10 (finer particles are a subset of coarser ones).")

if validation_errors:
    for err in validation_errors:
        st.warning(f"⚠️ {err}")

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────

predict_btn = st.button("🔍 Predict AQI", use_container_width=True, type="primary")

if predict_btn:
    input_array = np.array([[inputs[f] for f in FEATURES]])
    prediction  = model.predict(input_array)[0]
    prediction  = round(float(prediction), 1)

    label, icon, color, advice = get_category(prediction)

    # ── Result card ───────────────────────────────────────────────────────────

    st.markdown("### Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(label="Predicted AQI", value=prediction)

    with res_col2:
        st.markdown(
            f"""
            <div style="
                background-color: {color}18;
                border-left: 4px solid {color};
                border-radius: 6px;
                padding: 0.75rem 1rem;
                margin-top: 0.5rem;
            ">
                <span style="font-size:1.1rem; font-weight:600; color:{color};">
                    {icon} {label}
                </span><br>
                <span style="font-size:0.9rem; color:#555;">{advice}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── AQI scale gauge ───────────────────────────────────────────────────────

    st.markdown("#### AQI Scale")
    scale_cols = st.columns(5)
    categories_display = [
        ("Good",      "#2e7d32", "0–50"),
        ("Moderate",  "#f9a825", "51–100"),
        ("Poor",      "#e65100", "101–200"),
        ("Very Poor", "#b71c1c", "201–300"),
        ("Severe",    "#212121", "301–500"),
    ]
    for idx, (cat_name, cat_color, cat_range) in enumerate(categories_display):
        is_current = cat_name == label
        with scale_cols[idx]:
            st.markdown(
                f"""<div style="
                    text-align:center;
                    padding:0.4rem 0.2rem;
                    border-radius:6px;
                    background:{cat_color}{'33' if is_current else '18'};
                    border: {'2px' if is_current else '1px'} solid {cat_color};
                    font-size:0.75rem;
                    font-weight: {'700' if is_current else '400'};
                    color:{cat_color};
                ">
                    {cat_name}<br><span style="font-size:0.7rem">{cat_range}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Feature importance chart ──────────────────────────────────────────────

    if hasattr(model, "feature_importances_"):
        st.markdown("#### Which Pollutant Drove This Prediction?")
        st.caption(
            "Feature importance shows how much each pollutant influenced the model's decision overall. "
            "Higher = more influence."
        )

        importances = model.feature_importances_
        fi_df = (
            pd.DataFrame({"Pollutant": FEATURES, "Importance": importances})
            .sort_values("Importance", ascending=True)
        )

        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ["#1976d2" if p != fi_df["Pollutant"].iloc[-1] else "#e65100"
                  for p in fi_df["Pollutant"]]
        bars = ax.barh(fi_df["Pollutant"], fi_df["Importance"], color=colors, height=0.55)

        for bar, val in zip(bars, fi_df["Importance"]):
            ax.text(
                bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9, color="#444"
            )

        ax.set_xlabel("Importance Score", fontsize=9)
        ax.set_xlim(0, fi_df["Importance"].max() * 1.25)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        top_feature = fi_df["Pollutant"].iloc[-1]
        st.caption(
            f"💡 **{top_feature}** is the most influential pollutant in this model. "
            f"Your input: **{inputs[top_feature]} {POLLUTANT_CONFIG[top_feature][3]}**"
        )

    else:
        st.markdown("#### Your Input Values")
        fig, ax = plt.subplots(figsize=(6, 3))
        normalized = [inputs[f] / POLLUTANT_CONFIG[f][1] for f in FEATURES if f in POLLUTANT_CONFIG]
        ax.barh(FEATURES, normalized, color="#1976d2", height=0.55)
        ax.set_xlabel("Value as % of max safe range", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Data reference: CPCB AQI standards. "
    "Model trained on 16,010 real measurements from 26 Indian cities (2015–2020). "
    "For informational purposes only."
)
