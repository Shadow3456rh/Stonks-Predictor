"""
train_models.py
===============
Machine Learning Model Training Engine
Trains, evaluates, and persists candidate regression models:
1. Linear Regression
2. K-Nearest Neighbors (KNN Regressor)
3. Decision Tree Regressor
4. Random Forest Regressor
5. XGBoost Regressor (XGBRegressor)

Computes RMSE, MAE, R², and inference time (ms) on an out-of-sample test split.
Persists models to models/<model_name>.pkl and logs evaluation metrics to models/metrics.json.
"""

import os
import time
import json
import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Optional import for XGBoost with fallback
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Features to use for forecasting next-day closing price
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA_5", "MA_10", "MA_20", "MA_50",
    "Daily_Return", "Volatility_10", "Spread"
]
TARGET_COL = "Target_Next_Close"


def load_dataset():
    """
    Loads cleaned dataset and strictly EXCLUDES the present-day (most recent trading day)
    from the training set. The present day is held out completely as unseen out-of-sample data
    so live forward predictions for tomorrow can be demonstrated to teachers and evaluators.
    """
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset missing at {CLEANED_DATA_PATH}. Run clean_data.py first.")

    df = pd.read_csv(CLEANED_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Date"]).reset_index(drop=True)

    # Filter out present day (where Target_Next_Close is NaN because tomorrow has not occurred yet)
    df_historical = df.dropna(subset=[TARGET_COL]).copy()
    present_day_count = len(df) - len(df_historical)
    print(f"[*] Excluded {present_day_count} present-day record(s) from model training.")
    print(f"[*] Present day is held out as purely unseen live inference test data.")
    print(f"[*] Training strictly on historical completed periods ({len(df_historical)} records).")

    X = df_historical[FEATURE_COLS].copy()
    y = df_historical[TARGET_COL].copy()

    # Time-series chronological 80/20 train-test split on completed historical data
    split_idx = int(len(df_historical) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, len(df_historical)


def initialize_candidate_models():
    """Defines candidate models with standard scaling where needed."""
    models = {
        "Linear_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),
        "KNN_Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", KNeighborsRegressor(n_neighbors=5, weights="distance"))
        ]),
        "Decision_Tree": Pipeline([
            ("regressor", DecisionTreeRegressor(max_depth=6, min_samples_split=5, random_state=42))
        ]),
        "Random_Forest": Pipeline([
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1))
        ])
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("regressor", XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                n_jobs=-1
            ))
        ])
    else:
        print("[!] XGBoost is not installed in the environment. Training other 4 candidate models.")

    return models


def train_and_evaluate():
    """Trains all models, computes metrics, and persists artifacts."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[*] Loading and splitting time-series dataset...")
    X_train, X_test, y_train, y_test, total_samples = load_dataset()
    print(f"    Train size: {len(X_train)} samples | Test size: {len(X_test)} samples")

    candidate_models = initialize_candidate_models()
    metrics_summary = {}

    for model_name, model_pipe in candidate_models.items():
        print(f"\n[*] Training {model_name}...")
        start_train = time.time()
        model_pipe.fit(X_train, y_train)
        train_duration = time.time() - start_train

        # Measure inference latency on test set
        start_inf = time.time()
        y_pred = model_pipe.predict(X_test)
        inf_duration_ms = (time.time() - start_inf) * 1000.0

        # Compute evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics_summary[model_name] = {
            "model_name": model_name,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "inference_time_ms": round(inf_duration_ms, 2),
            "train_time_sec": round(train_duration, 3)
        }

        print(f"    -> {model_name} Results:")
        print(f"       R² Score:          {metrics_summary[model_name]['r2']}")
        print(f"       RMSE:              {metrics_summary[model_name]['rmse']}")
        print(f"       MAE:               {metrics_summary[model_name]['mae']}")
        print(f"       Inference Latency: {metrics_summary[model_name]['inference_time_ms']} ms")

        # Persist model pipeline artifact
        model_file_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        joblib.dump(model_pipe, model_file_path)
        print(f"    [+] Model saved to {model_file_path}")

    # Prepare complete metrics output file
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    output_payload = {
        "timestamp": timestamp,
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "models": metrics_summary
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[+] All metrics successfully logged to {METRICS_PATH}")
    return output_payload


if __name__ == "__main__":
    train_and_evaluate()
