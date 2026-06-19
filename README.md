# 🌍 Air Quality Index (AQI) Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Project Overview

A machine learning web application that predicts the **Air Quality Index (AQI)** of Indian cities based on real pollutant concentration data. The system compares multiple regression models, selects the best performer, and serves predictions through an interactive **Streamlit** interface with health advisories and feature importance visualizations.

> 🔗 **Live Demo:** [omkar22a-aqi-project-app-gee99u.streamlit.app](https://omkar22a-aqi-project-app-gee99u.streamlit.app/)
> 📁 **Dataset:** [Air Quality Data in India — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

---

## 🧩 Problem Statement

Air pollution is one of India's most pressing public health crises. Cities like Delhi, Mumbai, and Kolkata frequently record AQI values in the "Severe" range, leading to respiratory illnesses, reduced visibility, and long-term health damage. Despite the availability of pollutant sensor data, there is limited accessibility to tools that translate raw measurements into actionable health information for the general public.

This project addresses that gap by building a predictive model that takes pollutant readings as input and outputs an AQI value along with a health advisory — making air quality data interpretable and useful for everyday users.

---

## 🎯 Objectives

- Collect and preprocess real-world air quality data from Indian cities
- Explore and analyze pollutant distributions and their correlation with AQI
- Train and compare multiple machine learning regression models
- Select the best model based on evaluation metrics
- Deploy an interactive web app for real-time AQI prediction
- Provide health advisories based on predicted AQI levels

---

## 📊 Dataset Information

### Source
- **Primary:** [Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) — Kaggle
- **AQI Standards:** Central Pollution Control Board (CPCB), India
- **Coverage:** 26 major Indian cities | 2015 – 2020

### Records
| Split | Rows |
|-------|------|
| Raw dataset | ~29,000 |
| After cleaning | **16,010** |
| Train set (80%) | 12,808 |
| Test set (20%) | 3,202 |

### Features
| Feature | Description | Unit |
|---------|-------------|------|
| PM2.5 | Fine particulate matter ≤ 2.5 µm | µg/m³ |
| PM10 | Coarse particulate matter ≤ 10 µm | µg/m³ |
| NO2 | Nitrogen Dioxide | µg/m³ |
| SO2 | Sulphur Dioxide | µg/m³ |
| CO | Carbon Monoxide | mg/m³ |
| O3 | Ground-level Ozone | µg/m³ |

### Target Variable
- **AQI** (Air Quality Index) — continuous numerical value ranging from 0 to 500+
- Computed using CPCB sub-index breakpoint formula

---

## 🔧 Data Preprocessing

### Missing Values
- Columns with more than 50% null values were dropped automatically
- Remaining nulls were filled using **column-wise median imputation** — chosen over mean to avoid sensitivity to outliers
- Final dataset had **0 null values** across all 6 feature columns

### Outlier Handling
- AQI values above 500 were retained as they represent genuine "Severe" pollution events in Indian cities (e.g., Delhi winters)
- Negative pollutant values, if any, were treated as data errors and removed

### Feature Engineering
- No polynomial or interaction features were added — tree-based models capture non-linear relationships natively
- Features were kept in their original units (no normalization needed for Random Forest / Gradient Boosting)
- For Linear Regression comparison, raw features were used to demonstrate its limitations on this dataset

### Train/Test Split
- **80/20 split** with `random_state=42` for reproducibility
- Stratification not applied (regression task, not classification)

---

## 📈 Exploratory Data Analysis (EDA)

> 📓 See [`notebook.ipynb`](notebook.ipynb) for full EDA with visualizations.

Key analyses performed:
- **Distribution plots** — AQI and each pollutant's frequency distribution
- **Correlation heatmap** — PM2.5 and PM10 show strongest correlation with AQI
- **City-wise AQI comparison** — Delhi and Lucknow record highest average AQI
- **Seasonal trends** — AQI spikes in winter months (Oct–Jan) due to crop burning and low wind
- **Scatter plots** — PM2.5 vs AQI shows near-linear relationship above AQI 200
- **Outlier boxplots** — CO and O3 have significant right-skewed distributions

---

## 🔬 Features Used

Final features selected for model training (all 6 available after preprocessing):

```
PM2.5 | PM10 | NO2 | SO2 | CO | O3
```

Feature selection was data-driven — columns with >50% missing values were excluded, and the remaining 6 passed the threshold with 0 nulls post-imputation.

---

## 🤖 Machine Learning Models

Four regression models were trained and compared:

| Model | Why Included |
|-------|-------------|
| Linear Regression | Baseline — assumes linear relationships |
| Decision Tree | Captures non-linearity, interpretable |
| Random Forest | Ensemble of trees, robust to noise |
| Gradient Boosting | Sequential boosting, typically highest accuracy |

All models trained on the same 80/20 split with `random_state=42` for fair comparison.

---

## 📉 Model Evaluation

### Metrics Used

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | Mean Absolute Error | Average prediction error in AQI units |
| MSE | Mean Squared Error | Penalises large errors more heavily |
| RMSE | √MSE | Error in same unit as AQI |
| R² Score | 1 - SS_res/SS_tot | % of variance explained (higher = better) |

### Results

| Model | R² | RMSE | MAE |
|-------|----|------|-----|
| Linear Regression | 0.899 | 33.10 | 21.14 |
| Decision Tree | 0.853 | 40.04 | 23.21 |
| Random Forest | 0.924 | 28.82 | 16.59 |
| **Gradient Boosting** | **0.925** | **28.52** | **17.50** |

> ✅ **Gradient Boosting** selected as the final model with **R² = 0.925**

---

## 🏆 Results

- The model explains **92.5% of the variance** in AQI values
- Average prediction error of **±28.5 AQI points** (RMSE)
- Performs well across the full AQI range (14 – 1389)
- Gradient Boosting outperformed Linear Regression by ~2.6% R² — demonstrating the value of ensemble methods on this data

---

## 📊 Feature Importance

| Pollutant | Importance | Contribution |
|-----------|-----------|--------------|
| **PM2.5** | 0.478 | 47.8% |
| **PM10** | 0.255 | 25.5% |
| **CO** | 0.255 | 25.5% |
| O3 | 0.006 | 0.6% |
| SO2 | 0.003 | 0.3% |
| NO2 | 0.002 | 0.2% |

> PM2.5 alone drives nearly **half** of the model's predictions — consistent with CPCB's own AQI formula which weights fine particulates most heavily.

---

## 💡 Key Insights

- **PM2.5 is the dominant predictor** — improving PM2.5 sensor coverage directly improves AQI prediction accuracy
- **Delhi's AQI spikes in winter** due to crop residue burning in neighbouring states, which is captured in the seasonal trend analysis
- **Linear Regression still achieves R² = 0.899** — suggesting the AQI formula itself is largely linear with respect to PM2.5, which makes sense given how CPCB computes it
- **O3, SO2, and NO2 contribute less than 1% each** — these pollutants matter more for health classification than AQI magnitude in the Indian context
- **Gradient Boosting vs Random Forest gap is tiny (0.001 R²)** — both are viable; Random Forest trains faster and may be preferable for deployment

---

## 🛠️ Tech Stack

| Layer | Tool | Version |
|-------|------|---------|
| Language | Python | 3.9+ |
| ML Framework | scikit-learn | 1.x |
| Web UI | Streamlit | 1.x |
| Data Processing | pandas, NumPy | 2.x / 1.x |
| Visualization | matplotlib | 3.x |
| Model Persistence | pickle | built-in |
| Data Source | Kaggle API / CSV | — |

---

## 🏗️ Project Architecture / Workflow

```
Raw CSV (city_day.csv)
        │
        ▼
  Data Cleaning
  (drop nulls, median fill)
        │
        ▼
  train.py
  ├── Model Comparison (4 models)
  ├── Evaluation (R², RMSE, MAE)
  ├── Feature Importance
  └── Save best model → model.pkl
        │
        ▼
  app.py (Streamlit)
  ├── User inputs 6 pollutant values
  ├── Input validation (range checks)
  ├── model.pkl → predict AQI
  ├── AQI category + health advisory
  └── Feature importance chart
```

---

## 📁 Project Structure

```
AQI-Project/
│
├── app.py                  # Streamlit web application
├── train.py                # Model training, comparison & evaluation
├── fetch_aqi_data.py       # OpenAQ API data fetcher (optional)
├── notebook.ipynb          # EDA notebook with visualizations
│
├── aqi_data.csv            # Cleaned dataset (generated from city_day.csv)
├── city_day.csv            # Raw Kaggle dataset (do not commit — add to .gitignore)
├── model.pkl               # Trained Gradient Boosting model
│
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/omkar22A/AQI-Project.git
cd AQI-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# Go to: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
# Download city_day.csv and place it in the project root

# 4. Prepare the dataset
python -c "import pandas as pd; df = pd.read_csv('city_day.csv'); df = df[['PM2.5','PM10','NO2','SO2','CO','O3','AQI']].dropna(); df.to_csv('aqi_data.csv', index=False); print(f'Saved {len(df)} rows')"

# 5. Train the model
python train.py

# 6. Launch the app
streamlit run app.py
```

---

## 🖥️ Usage

1. Open the app at `http://localhost:8501`
2. Enter pollutant concentrations using the sliders/inputs
3. Click **Predict AQI**
4. View:
   - Predicted AQI value
   - AQI category (Good / Moderate / Poor / Very Poor / Severe)
   - Health advisory
   - Feature importance chart

### AQI Category Reference

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0 – 50 | 🟢 Good | Minimal impact |
| 51 – 100 | 🟡 Moderate | Minor breathing discomfort for sensitive people |
| 101 – 200 | 🟠 Poor | Breathing discomfort for general population |
| 201 – 300 | 🔴 Very Poor | Respiratory illness on prolonged exposure |
| 301 – 500 | ⚫ Severe | Affects healthy people, serious impact on sensitive groups |

---

## 📸 Screenshots

> *(Add screenshots of your Streamlit app here)*

---

## 🚀 Future Improvements

- [x] Add EDA notebook with full visualizations (`notebook.ipynb`)
- [x] Deploy on Streamlit Community Cloud with public URL
- [ ] Integrate live AQI data from OpenAQ v3 API for real-time predictions
- [ ] Add city-wise AQI prediction using location dropdown
- [ ] Hyperparameter tuning with GridSearchCV for better model performance
- [ ] Add time-series forecasting (LSTM) to predict AQI for next 24 hours
- [ ] Add map visualization showing AQI levels across Indian cities
- [ ] Support multi-language UI (Hindi, Marathi, Tamil)

---

## ⚡ Challenges Faced

- **Missing data:** OpenAQ API returned sparse data with entire pollutant columns missing for some cities — solved by switching to the Kaggle CPCB dataset
- **Column nulls:** PM10 was 100% null in the initial API fetch — required dynamic feature selection logic in `train.py`
- **AQI calculation:** OpenAQ doesn't provide AQI directly — had to implement CPCB sub-index breakpoint formula manually
- **Model selection:** Gradient Boosting and Random Forest performed nearly identically — chose Gradient Boosting by R² but documented both

---

## 📚 Learnings

- Real-world datasets are messy — data cleaning took more time than model training
- API data quality varies significantly; always have a fallback dataset source
- Feature importance from tree models gives directly interpretable, domain-relevant insights
- Building an end-to-end pipeline (data → model → UI) is very different from just training a model in a notebook
- CPCB's AQI formula is sub-index based — understanding domain context improves both preprocessing and model interpretation

---

## ✅ Conclusion

This project demonstrates a complete machine learning pipeline — from raw data ingestion and preprocessing, through model training and evaluation, to a deployed interactive web application. The Gradient Boosting model achieves **R² = 0.925** on 16,010 real CPCB measurements, with PM2.5 emerging as the dominant predictor at 47.8% feature importance — consistent with established air quality science.

The project is structured, documented, and reproducible, making it a strong demonstration of practical ML and Python skills for data science and software engineering roles.

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
requests
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 👤 Author

**Omkar Avasarkar**
B.Tech Student | Aspiring Data Scientist

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

## 📬 Contact Information

| Platform | Link |
|----------|------|
| 💼 LinkedIn | [linkedin.com/in/omkar-avasarkar-3460b9402](https://www.linkedin.com/in/omkar-avasarkar-3460b9402/) |
| 🐙 GitHub | [github.com/omkar22A](https://github.com/omkar22A) |
| 📧 Email | avasarkaromkar22@gmail.com |

---

<p align="center">
  Made with ❤️ using Python & Streamlit &nbsp;|&nbsp; Data: CPCB via Kaggle
</p>