"""
Explainable Network Inference Framework for
Handling Missing Data & Outliers in Smart Energy Demand Forecasting

Author  : Siddhant Singh
Roll No : 25SCS1003000996
Guide   : Mr. Ramavath Ganesh
Dept    : Computer Science & Engineering, IILM University
Year    : 2025
"""

# standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error


# generate hourly smart meter data with daily + weekly patterns
def generate_synthetic_energy(n_days=60, seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="H")

    hours = idx.hour
    days  = idx.dayofweek

    # consumption tends to peak in evenings, dip at night
    daily_pattern  = 1.5 + 1.2 * np.sin(2 * np.pi * (hours - 6) / 24)

    # weekdays slightly higher than weekends
    weekly_pattern = 0.3 + 0.2 * (days < 5).astype(float)

    noise = rng.normal(0, 0.15, size=len(idx))
    load  = daily_pattern + weekly_pattern + noise
    load  = np.clip(load, 0, None)

    df = pd.DataFrame({"timestamp": idx, "load_kwh": load})
    df.set_index("timestamp", inplace=True)
    return df


# simulate the kind of mess you get in real smart meter data
def corrupt_data(df, missing_rate=0.08, outlier_rate=0.02, seed=42):
    rng = np.random.RandomState(seed)
    df_corrupt = df.copy()
    n = len(df_corrupt)

    # random gaps, like a meter losing connectivity
    miss_mask = rng.rand(n) < missing_rate
    df_corrupt.loc[miss_mask, "load_kwh"] = np.nan

    # spikes - sensor glitch or sudden load surge
    out_idx = rng.choice(n, size=int(outlier_rate * n), replace=False)
    factors = rng.uniform(3, 6, size=len(out_idx))
    df_corrupt.iloc[out_idx, 0] *= factors

    # a few negatives - happens with firmware bugs or bad resets
    neg_idx = rng.choice(n, size=max(3, int(0.005 * n)), replace=False)
    df_corrupt.iloc[neg_idx, 0] = -rng.uniform(1, 3, size=len(neg_idx))

    return df_corrupt


def add_time_features(df):
    out = df.copy()
    out["hour"]      = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    return out


# main cleaning function - KNN fills the gaps, isolation forest catches outliers
def clean_with_network_inference(df_corrupt):
    df_feat = add_time_features(df_corrupt)

    # KNN imputation using hour + day context, not just raw values
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    arr     = imputer.fit_transform(df_feat[["load_kwh", "hour", "dayofweek"]])

    df_imputed = df_feat.copy()
    df_imputed["load_kwh"] = arr[:, 0]

    # isolation forest to find the remaining anomalies
    iso = IsolationForest(
        contamination=0.03,
        n_estimators=200,
        random_state=0,
    )
    iso.fit(df_imputed[["load_kwh", "hour", "dayofweek"]])
    is_outlier = iso.predict(df_imputed[["load_kwh", "hour", "dayofweek"]]) == -1

    # replace each outlier with the median of its neighbours
    df_clean = df_imputed.copy()
    series   = df_clean["load_kwh"].values

    for i, flag in enumerate(is_outlier):
        if flag:
            left   = max(0, i - 3)
            right  = min(len(series), i + 4)
            window = np.delete(series[left:right], np.where(np.arange(left, right) == i))
            series[i] = np.median(window)

    df_clean["load_kwh"] = series
    return df_clean, is_outlier


# sliding window to build X, y for the forecasting model
def create_supervised(df, lag=24, horizon=24):
    values = df["load_kwh"].values
    X, y   = [], []

    for i in range(lag, len(values) - horizon):
        X.append(values[i - lag : i])
        y.append(values[i : i + horizon].sum())

    return np.array(X), np.array(y)


# compare forecast error on dirty data vs cleaned data
def evaluate_forecasting(df_raw, df_clean, lag=24, horizon=24):
    # naive preprocessing for the baseline - just forward fill
    df_raw_ffill = df_raw.copy()
    df_raw_ffill["load_kwh"] = df_raw_ffill["load_kwh"].ffill().bfill()

    X_raw,   y_raw   = create_supervised(df_raw_ffill, lag, horizon)
    X_clean, y_clean = create_supervised(df_clean,     lag, horizon)

    split_raw   = int(0.8 * len(X_raw))
    split_clean = int(0.8 * len(X_clean))

    model_raw   = RandomForestRegressor(n_estimators=200, random_state=0)
    model_clean = RandomForestRegressor(n_estimators=200, random_state=0)

    model_raw.fit(X_raw[:split_raw],     y_raw[:split_raw])
    model_clean.fit(X_clean[:split_clean], y_clean[:split_clean])

    y_pred_raw   = model_raw.predict(X_raw[split_raw:])
    y_pred_clean = model_clean.predict(X_clean[split_clean:])

    mape_raw   = mean_absolute_percentage_error(y_raw[split_raw:],   y_pred_raw)   * 100
    mape_clean = mean_absolute_percentage_error(y_clean[split_clean:], y_pred_clean) * 100

    return mape_raw, mape_clean


def plot_before_after(df_original, df_corrupt, df_clean):
    plt.figure(figsize=(14, 5))
    plt.plot(df_original.index, df_original["load_kwh"],
             label="Original (ideal)", linewidth=1.2, color="steelblue")
    plt.plot(df_corrupt.index, df_corrupt["load_kwh"],
             label="With Missing/Outliers", alpha=0.6, color="orange")
    plt.plot(df_clean.index, df_clean["load_kwh"],
             label="After Network Inference", linewidth=1.5, color="green")
    plt.xlabel("Time")
    plt.ylabel("Load (kWh)")
    plt.title("Energy Time Series - Before vs After Network Inference")
    plt.legend()
    plt.tight_layout()
    plt.savefig("energy_before_after.png", dpi=300)
    plt.close()


def plot_forecast_bar(mape_raw, mape_clean):
    labels = ["Before Cleaning", "After Inference"]
    values = [mape_raw, mape_clean]
    colors = ["#e74c3c", "#2ecc71"]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=colors, width=0.4)
    plt.ylabel("MAPE (%)")
    plt.title("24-Hour Demand Forecast Error")
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}%",
            ha="center",
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig("forecast_before_after.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    print("generating data...")
    df_original = generate_synthetic_energy(n_days=60)

    print("corrupting data...")
    df_corrupt = corrupt_data(df_original)

    print("cleaning with network inference...")
    df_clean, is_outlier = clean_with_network_inference(df_corrupt)

    print("evaluating forecasting...")
    mape_raw, mape_clean = evaluate_forecasting(df_corrupt, df_clean)

    print("saving plots...")
    plot_before_after(df_original, df_corrupt, df_clean)
    plot_forecast_bar(mape_raw, mape_clean)

    print("\n===== DATA QUALITY SUMMARY =====")
    print(f"Total points             : {len(df_corrupt)}")
    print(f"Missing values (before)  : {df_corrupt['load_kwh'].isna().sum()}")
    print(f"Missing values (after)   : {df_clean['load_kwh'].isna().sum()}")
    print(f"Outliers detected        : {is_outlier.sum()}")

    print("\n===== 24-HOUR FORECASTING (Random Forest) =====")
    print(f"MAPE before cleaning : {mape_raw:.2f}%")
    print(f"MAPE after inference : {mape_clean:.2f}%")
    print(f"Improvement          : {mape_raw - mape_clean:.2f} percentage points")
