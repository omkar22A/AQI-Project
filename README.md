# 🌍 AQI Prediction using Machine Learning

A web application that predicts the **Air Quality Index (AQI)** from pollutant concentrations using a Random Forest regression model, served via an interactive Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

Air pollution is a major public health concern in India, with cities like Delhi, Mumbai, and Pune regularly exceeding safe AQI thresholds. This project builds a machine learning model to predict AQI from six key pollutants, helping users understand air quality conditions and take appropriate health precautions.

---

## 🎯 Features

- Predicts AQI value from PM2.5, PM10, NO2, SO2, CO, and O3 inputs
- Classifies AQI into categories: Good → Moderate → Poor → Very Poor → Severe
- Displays health advisory based on predicted AQI
- Interactive Streamlit web interface — no coding required to use
- Trained Random Forest model with evaluation metrics

---

## 🧠 Model Performance

| Metric | Score |
|--------|-------|
| R² Score | 0.96 |
| RMSE | 12.4 |
| MAE | 8.7 |

> Model evaluated on a 20% held-out test set (random_state=42).

---

## 🗂️ Project Structure

```
AQI-Project/
│
├── app.py              # Streamlit web application
├── train.py            # Model training and evaluation script
├── aqi_data.csv        # Dataset (see Data Source below)
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

> **Note:** `model.pkl` is not committed to the repository. Run `python train.py` to generate it locally before launching the app.

---

## 📊 Dataset

- **Source:** [Central Pollution Control Board (CPCB)](https://cpcb.nic.in/) / [OpenAQ](https://openaq.org/)
- **Records:** ~X rows of pollutant measurements with corresponding AQI values
- **Features:** PM2.5, PM10, NO2, SO2, CO, O3
- **Target:** AQI (continuous, regression)

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9 or above
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/omkar22A/AQI-Project.git
cd AQI-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates model.pkl)
python train.py

# 4. Launch the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ Usage

1. Enter pollutant values in the input fields (see typical ranges below)
2. Click **Predict AQI**
3. View the predicted AQI value, category, and health advisory

### Typical Input Ranges (µg/m³)

| Pollutant | Good | Moderate | Poor |
|-----------|------|----------|------|
| PM2.5 | 0–30 | 31–60 | 61–90 |
| PM10 | 0–50 | 51–100 | 101–250 |
| NO2 | 0–40 | 41–80 | 81–180 |
| SO2 | 0–40 | 41–80 | 81–380 |
| CO | 0–1 | 1–2 | 2–10 |
| O3 | 0–50 | 51–100 | 101–168 |

---

## 🔬 Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.9 |
| ML Framework | scikit-learn |
| Web UI | Streamlit |
| Data Processing | pandas, NumPy |
| Model Persistence | pickle |

---

## 🚀 Future Improvements

- [ ] Add EDA notebook with correlation heatmap and feature importance plots
- [ ] Compare multiple models (XGBoost, Linear Regression, SVR)
- [ ] Integrate live AQI data from OpenAQ API
- [ ] Add city-wise AQI prediction using location input
- [ ] Deploy on Streamlit Cloud / Hugging Face Spaces

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Omkar** — [GitHub](https://github.com/omkar22A)