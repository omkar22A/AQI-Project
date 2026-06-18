import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("🌍 Air Quality Index (AQI) Prediction System")

st.write("Enter pollutant values below:")

# Inputs
pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
no2 = st.number_input("NO2")
so2 = st.number_input("SO2")
co = st.number_input("CO")
o3 = st.number_input("O3")

# Predict
if st.button("Predict AQI"):
    input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
    prediction = model.predict(input_data)[0]

    st.subheader(f"Predicted AQI: {prediction:.2f}")

    # Category
    if prediction <= 50:
        st.success("Good 😊")
    elif prediction <= 100:
        st.info("Moderate 🙂")
    elif prediction <= 200:
        st.warning("Poor 😷")
    elif prediction <= 300:
        st.error("Very Poor 🤢")
    else:
        st.error("Severe ☠️")