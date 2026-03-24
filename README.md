# Smart Energy Demand Forecasting — Network Inference

B.Tech Project | IILM University, 2025
Author: Siddhant Singh (25SCS1003000996)
Guide: Mr. Ramavath Ganesh

---

## What this project does

Smart meter data in the real world is never clean. Meters drop connectivity, sensors glitch, firmware bugs cause negative readings, sudden load surges create spikes. If you feed this raw data directly into a forecasting model, the results are unreliable.

This project builds a preprocessing pipeline that detects and fixes these issues before forecasting:

1. Finds missing values and classifies them (MCAR / MAR / MNAR)
2. Reconstructs gaps using KNN with temporal context (hour + day of week)
3. Detects outliers using Isolation Forest
4. Replaces anomalous readings with local neighbourhood medians
5. Trains a Random Forest forecasting model on the cleaned data

---

## Results

| | Before | After |
|---|---|---|
| Missing values | 121 | 0 |
| Outliers detected | - | 44 |
| Forecast MAPE | 4.40% | 3.19% |

---

## How to run

Install dependencies:
```
pip install numpy pandas matplotlib scikit-learn
```

Run:
```
python main.py
```

This will print a summary and save two plots — `energy_before_after.png` and `forecast_before_after.png`.

---

## Project structure

```
smart-energy-forecasting/
├── main.py          # full source code
├── README.md
└── report.pdf       # project report (optional)
```

---

## Tech used

- numpy, pandas for data handling
- scikit-learn for KNNImputer, IsolationForest, RandomForestRegressor
- matplotlib for plots

---

## Key ideas

**KNN Imputation** — instead of just filling gaps with the mean or interpolation, KNN finds the k most similar time windows (based on hour and day patterns) and uses those to estimate the missing value. Works well because energy consumption is repetitive.

**Isolation Forest** — detects anomalies by isolating data points using random tree splits. Outliers tend to get isolated quickly because they sit far from the normal cluster.

**MCAR / MAR / MNAR** — three types of missingness that need different handling strategies. Simple interpolation only works for MCAR; the more complex types need inference-based approaches.
