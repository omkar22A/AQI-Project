"""
fetch_aqi_data.py
-----------------
Fetches real pollutant data from OpenAQ v3 API for major Indian cities,
calculates AQI using CPCB breakpoints, and saves to aqi_data.csv.

Usage:
    python fetch_aqi_data.py --api-key YOUR_OPENAQ_API_KEY

Get a free key at: https://explore.openaq.org/register
"""

import argparse
import requests
import pandas as pd
import math
import time


# ── CPCB AQI breakpoints ──────────────────────────────────────────────────────
# Source: Central Pollution Control Board, India
# Format: (C_low, C_high, I_low, I_high) per pollutant

AQI_BREAKPOINTS = {
    "PM2.5": [
        (0,    30,    0,   50),
        (30,   60,   51,  100),
        (60,   90,  101,  200),
        (90,  120,  201,  300),
        (120, 250,  301,  400),
        (250, 500,  401,  500),
    ],
    "PM10": [
        (0,    50,    0,   50),
        (50,  100,   51,  100),
        (100, 250,  101,  200),
        (250, 350,  201,  300),
        (350, 430,  301,  400),
        (430, 600,  401,  500),
    ],
    "NO2": [
        (0,    40,    0,   50),
        (40,   80,   51,  100),
        (80,  180,  101,  200),
        (180, 280,  201,  300),
        (280, 400,  301,  400),
        (400, 800,  401,  500),
    ],
    "SO2": [
        (0,    40,    0,   50),
        (40,   80,   51,  100),
        (80,  380,  101,  200),
        (380, 800,  201,  300),
        (800,1600,  301,  400),
        (1600,2100, 401,  500),
    ],
    "CO": [
        (0,    1,     0,   50),
        (1,    2,    51,  100),
        (2,   10,   101,  200),
        (10,  17,   201,  300),
        (17,  34,   301,  400),
        (34,  50,   401,  500),
    ],
    "O3": [
        (0,    50,    0,   50),
        (50,  100,   51,  100),
        (100, 168,  101,  200),
        (168, 208,  201,  300),
        (208, 748,  301,  400),
        (748,1000,  401,  500),
    ],
}


def calc_sub_index(concentration, breakpoints):
    """Calculate sub-index for one pollutant using linear interpolation."""
    for (C_lo, C_hi, I_lo, I_hi) in breakpoints:
        if C_lo <= concentration <= C_hi:
            return ((I_hi - I_lo) / (C_hi - C_lo)) * (concentration - C_lo) + I_lo
    return None


def calc_aqi(row):
    """AQI = max of all sub-indices (CPCB method)."""
    sub_indices = []
    for pollutant, breakpoints in AQI_BREAKPOINTS.items():
        val = row.get(pollutant)
        if val is not None and not math.isnan(val) and val >= 0:
            si = calc_sub_index(val, breakpoints)
            if si is not None:
                sub_indices.append(si)
    return round(max(sub_indices), 1) if sub_indices else None


# ── OpenAQ v3 station IDs for major Indian cities ────────────────────────────
# These are verified government monitor IDs from OpenAQ Explorer
# Find more at: https://explore.openaq.org (filter by India)

INDIA_LOCATIONS = {
    "Delhi":        8118,
    "Mumbai":       8119,
    "Kolkata":      8120,
    "Chennai":      8121,
    "Bangalore":    8122,
    "Hyderabad":    8123,
    "Pune":         8124,
    "Ahmedabad":    8125,
}

BASE_URL = "https://api.openaq.org/v3"

PARAM_MAP = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2":  "NO2",
    "so2":  "SO2",
    "co":   "CO",
    "o3":   "O3",
}


def fetch_latest(location_id, api_key):
    """Fetch latest measurements for one location."""
    url = f"{BASE_URL}/locations/{location_id}/latest"
    headers = {"X-API-Key": api_key}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 404:
            print(f"  Location {location_id} not found — skipping")
            return None
        resp.raise_for_status()
        return resp.json().get("results", [])
    except requests.RequestException as e:
        print(f"  Request failed for location {location_id}: {e}")
        return None


def fetch_daily_measurements(location_id, api_key, days=90):
    """
    Fetch daily aggregated measurements for one location.
    Returns a list of {date, PM2.5, PM10, NO2, SO2, CO, O3} dicts.
    """
    records = []
    sensors_url = f"{BASE_URL}/locations/{location_id}/sensors"
    headers = {"X-API-Key": api_key}

    try:
        resp = requests.get(sensors_url, headers=headers, timeout=10)
        resp.raise_for_status()
        sensors = resp.json().get("results", [])
    except requests.RequestException as e:
        print(f"  Could not fetch sensors for {location_id}: {e}")
        return []

    # Collect daily averages per sensor/parameter
    param_daily = {}  # {date: {param: value}}

    for sensor in sensors:
        param_name = sensor.get("parameter", {}).get("name", "").lower()
        col_name = PARAM_MAP.get(param_name)
        if not col_name:
            continue

        sensor_id = sensor["id"]
        meas_url = f"{BASE_URL}/sensors/{sensor_id}/days"
        params = {"limit": days}

        try:
            resp = requests.get(meas_url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            measurements = resp.json().get("results", [])
        except requests.RequestException as e:
            print(f"    Sensor {sensor_id} ({col_name}) failed: {e}")
            continue

        for m in measurements:
            date = m.get("period", {}).get("datetimeFrom", {}).get("utc", "")[:10]
            val  = m.get("value")
            if date and val is not None:
                param_daily.setdefault(date, {})[col_name] = round(float(val), 2)

        time.sleep(0.1)  # be polite to the API

    # Build one row per date
    for date, params_dict in sorted(param_daily.items()):
        row = {"date": date}
        row.update(params_dict)
        records.append(row)

    return records


def main():
    parser = argparse.ArgumentParser(description="Fetch AQI data from OpenAQ v3")
    parser.add_argument("--api-key", required=True, help="Your OpenAQ API key")
    parser.add_argument("--days",    type=int, default=90, help="Days of history (default 90)")
    parser.add_argument("--output",  default="aqi_data.csv", help="Output CSV path")
    args = parser.parse_args()

    all_rows = []

    for city, loc_id in INDIA_LOCATIONS.items():
        print(f"Fetching {city} (location {loc_id})...")
        records = fetch_daily_measurements(loc_id, args.api_key, args.days)

        for rec in records:
            rec["city"] = city
            rec["AQI"]  = calc_aqi(rec)
            all_rows.append(rec)

        print(f"  Got {len(records)} daily records")
        time.sleep(0.5)  # rate limit buffer

    if not all_rows:
        print("\nNo data fetched. Check your API key and location IDs.")
        print("Tip: Verify location IDs at https://explore.openaq.org")
        return

    df = pd.DataFrame(all_rows)

    # Ensure all pollutant columns exist (fill missing with NaN)
    for col in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]:
        if col not in df.columns:
            df[col] = float("nan")

    # Drop rows where we couldn't compute AQI (too many missing pollutants)
    before = len(df)
    df = df.dropna(subset=["AQI"])
    dropped = before - len(df)

    # Reorder columns nicely
    col_order = ["date", "city", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
    df = df[[c for c in col_order if c in df.columns]]

    df.to_csv(args.output, index=False)

    print(f"\n{'='*50}")
    print(f"Saved {len(df)} rows to {args.output}  ({dropped} dropped, missing AQI)")
    print(f"Cities: {df['city'].nunique()}  |  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"AQI range: {df['AQI'].min():.0f} – {df['AQI'].max():.0f}  |  Mean: {df['AQI'].mean():.1f}")
    print(f"{'='*50}")
    print("\nNext step: run  python train.py  to retrain with real data")


if __name__ == "__main__":
    main()
