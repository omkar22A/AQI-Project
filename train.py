import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ── 1. Load data ──────────────────────────────────────────────────────────────

df = pd.read_csv("aqi_data.csv")

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"  Rows       : {len(df)}")
print(f"  AQI range  : {df['AQI'].min():.1f} – {df['AQI'].max():.1f}")
print(f"  AQI mean   : {df['AQI'].mean():.1f}  |  std: {df['AQI'].std():.1f}")

print("\n  Null counts per column:")
print(df.isnull().sum().to_string())


# ── 2. Drop columns that are mostly empty (>50% null) ────────────────────────

ALL_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
threshold = 0.5 * len(df)

good_features = [
    col for col in ALL_FEATURES
    if col in df.columns and df[col].isnull().sum() < threshold
]
dropped_cols = [col for col in ALL_FEATURES if col not in good_features]

if dropped_cols:
    print(f"\n  ⚠  Dropping columns with >50% nulls: {dropped_cols}")
print(f"  ✓  Using features: {good_features}")

# Fill remaining nulls in good columns with their median
for col in good_features:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)


# ── 3. Features & target ──────────────────────────────────────────────────────

FEATURES = good_features
TARGET   = "AQI"

X = df[FEATURES]
y = df[TARGET]

print(f"\n  Rows for training : {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train size : {len(X_train)}   Test size: {len(X_test)}")


# ── 4. Model comparison ───────────────────────────────────────────────────────

MODELS = {
    "Linear Regression"  : LinearRegression(),
    "Decision Tree"      : DecisionTreeRegressor(random_state=42),
    "Random Forest"      : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = []

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(f"  {'Model':<22} {'R²':>6}  {'RMSE':>7}  {'MAE':>7}")
print("  " + "-" * 47)

for name, model in MODELS.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    results.append({"name": name, "model": model, "r2": r2, "rmse": rmse, "mae": mae})
    print(f"  {name:<22} {r2:>6.3f}  {rmse:>7.2f}  {mae:>7.2f}")


# ── 5. Pick best model ────────────────────────────────────────────────────────

best = max(results, key=lambda x: x["r2"])

print("\n" + "=" * 55)
print("BEST MODEL")
print("=" * 55)
print(f"  Name  : {best['name']}")
print(f"  R²    : {best['r2']:.4f}  (explains {best['r2']*100:.1f}% of variance)")
print(f"  RMSE  : {best['rmse']:.2f}")
print(f"  MAE   : {best['mae']:.2f}")


# ── 6. Feature importance ─────────────────────────────────────────────────────

if hasattr(best["model"], "feature_importances_"):
    importances = best["model"].feature_importances_
    ranked = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 55)
    print("FEATURE IMPORTANCE")
    print("=" * 55)
    for feat, imp in ranked:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<8}  {bar:<40}  {imp:.3f}")


# ── 7. Save model + feature list ─────────────────────────────────────────────

pickle.dump({"model": best["model"], "features": FEATURES}, open("model.pkl", "wb"))

print("\n" + "=" * 55)
print(f"  ✓  Saved '{best['name']}' as model.pkl")
print(f"  ✓  Features saved: {FEATURES}")
print("=" * 55)