import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ── 1. Load & validate data ───────────────────────────────────────────────────

df = pd.read_csv("aqi_data.csv")

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"  Rows       : {len(df)}")
print(f"  Columns    : {list(df.columns)}")
print(f"  Missing    : {df.isnull().sum().sum()} total null values")
print(f"  AQI range  : {df['AQI'].min():.1f} – {df['AQI'].max():.1f}")
print(f"  AQI mean   : {df['AQI'].mean():.1f}  |  std: {df['AQI'].std():.1f}")

# Drop rows with missing values (simple strategy — log how many)
before = len(df)
df = df.dropna()
dropped = before - len(df)
if dropped:
    print(f"\n  ⚠  Dropped {dropped} rows with missing values.")


# ── 2. Features & target ──────────────────────────────────────────────────────

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET   = "AQI"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  Train size : {len(X_train)}   Test size: {len(X_test)}")


# ── 3. Model comparison ───────────────────────────────────────────────────────

MODELS = {
    "Linear Regression"       : LinearRegression(),
    "Decision Tree"           : DecisionTreeRegressor(random_state=42),
    "Random Forest"           : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"       : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = []

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(f"  {'Model':<26} {'R²':>6}  {'RMSE':>7}  {'MAE':>7}")
print("  " + "-" * 51)

for name, model in MODELS.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    results.append({"name": name, "model": model, "r2": r2, "rmse": rmse, "mae": mae})
    print(f"  {name:<26} {r2:>6.3f}  {rmse:>7.2f}  {mae:>7.2f}")


# ── 4. Pick the best model (by R²) ───────────────────────────────────────────

best = max(results, key=lambda x: x["r2"])

print("\n" + "=" * 55)
print("BEST MODEL")
print("=" * 55)
print(f"  Name  : {best['name']}")
print(f"  R²    : {best['r2']:.4f}  (explains {best['r2']*100:.1f}% of variance)")
print(f"  RMSE  : {best['rmse']:.2f}  (avg error ±{best['rmse']:.1f} AQI points)")
print(f"  MAE   : {best['mae']:.2f}")


# ── 5. Feature importance (if tree-based) ────────────────────────────────────

if hasattr(best["model"], "feature_importances_"):
    importances = best["model"].feature_importances_
    ranked = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 55)
    print("FEATURE IMPORTANCE")
    print("=" * 55)
    for feat, imp in ranked:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<8}  {bar:<40}  {imp:.3f}")


# ── 6. Save the best model ────────────────────────────────────────────────────

pickle.dump(best["model"], open("model.pkl", "wb"))

print("\n" + "=" * 55)
print(f"  ✓  Saved '{best['name']}' as model.pkl")
print("=" * 55)