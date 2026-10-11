"""
clean_data.py
=============
Data Cleaning & Feature Engineering Pipeline
1. Cleans missing values and standardizes OHLCV series.
2. Removes outliers using statistical bounding.
3. Engineers features: Moving Averages (5, 10, 20, 50), Daily Returns, Rolling Volatility, High-Low Spread.
4. Constructs prediction target: Target_Next_Close (t+1 closing price) and Target_Direction.
5. Generates EDA summary in data/eda_summary.json.
6. Exports feature correlation heatmap to static/eda_corr.png.
"""

import os
import json
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_data.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
EDA_SUMMARY_PATH = os.path.join(DATA_DIR, "eda_summary.json")
EDA_CORR_IMG_PATH = os.path.join(STATIC_DIR, "eda_corr.png")


def engineer_ticker_features(df_ticker):
    """
    Computes technical indicators and target variables for a single ticker dataframe.
    """
    df = df_ticker.sort_values(by="Date").copy()

    # Fill missing values within ticker
    df["Close"] = df["Close"].ffill().bfill()
    df["Open"] = df["Open"].fillna(df["Close"])
    df["High"] = df["High"].fillna(df[["Open", "Close"]].max(axis=1))
    df["Low"] = df["Low"].fillna(df[["Open", "Close"]].min(axis=1))
    df["Volume"] = df["Volume"].fillna(0)

    # 1. Moving Averages
    df["MA_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["MA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["MA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["MA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()

    # 2. Daily Returns (percentage change)
    df["Daily_Return"] = df["Close"].pct_change().fillna(0)

    # 3. Rolling Volatility (10-day standard deviation of daily returns)
    df["Volatility_10"] = df["Daily_Return"].rolling(window=10, min_periods=1).std().fillna(0)

    # 4. Normalized High-Low Spread
    df["Spread"] = ((df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)).fillna(0)

    # 5. Target variable: Next day closing price
    df["Target_Next_Close"] = df["Close"].shift(-1)

    # 6. Target direction: 1 if next day close > today close, else 0
    df["Target_Direction"] = (df["Target_Next_Close"] > df["Close"]).astype(int)

    return df


def remove_outliers(df, numeric_cols):
    """
    Clips extreme anomalous spikes using IQR bounding to preserve trends while eliminating corrupted spikes.
    """
    cleaned_df = df.copy()
    for col in numeric_cols:
        if col in cleaned_df.columns:
            q25 = cleaned_df[col].quantile(0.01)
            q75 = cleaned_df[col].quantile(0.99)
            iqr = q75 - q25
            lower_bound = q25 - (1.5 * iqr)
            upper_bound = q75 + (1.5 * iqr)
            cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
    return cleaned_df


def generate_eda_artifacts(cleaned_df):
    """
    Produces eda_summary.json and correlation heatmap image (eda_corr.png).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "MA_5", "MA_10", "MA_20", "MA_50",
        "Daily_Return", "Volatility_10", "Spread", "Target_Next_Close"
    ]
    numeric_df = cleaned_df[feature_cols].dropna()

    # Compute descriptive statistics
    desc = numeric_df.describe().to_dict()
    corr_matrix = numeric_df.corr().round(4).to_dict()

    ticker_stats = {}
    for ticker, group in cleaned_df.groupby("Ticker"):
        ticker_stats[ticker] = {
            "record_count": int(len(group)),
            "latest_date": str(group["Date"].max()),
            "latest_close": round(float(group["Close"].iloc[-1]), 2),
            "mean_close": round(float(group["Close"].mean()), 2),
            "std_close": round(float(group["Close"].std()), 2),
            "mean_daily_return": round(float(group["Daily_Return"].mean()), 5),
            "avg_volatility": round(float(group["Volatility_10"].mean()), 5)
        }

    summary = {
        "dataset_summary": {
            "total_rows": int(len(cleaned_df)),
            "tickers_included": list(cleaned_df["Ticker"].unique()),
            "features_engineered": [
                "MA_5", "MA_10", "MA_20", "MA_50",
                "Daily_Return", "Volatility_10", "Spread"
            ],
            "target_variable": "Target_Next_Close"
        },
        "descriptive_statistics": desc,
        "correlation_matrix": corr_matrix,
        "ticker_breakdown": ticker_stats
    }

    with open(EDA_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved EDA summary JSON to {EDA_SUMMARY_PATH}")

    # Generate correlation heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Modern dark-compatible or clean financial theme
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7}
        )
        plt.title("Stock Predictor - Feature Correlation Heatmap", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(EDA_CORR_IMG_PATH, dpi=180)
        plt.close()
        print(f"[+] Generated correlation heatmap to {EDA_CORR_IMG_PATH}")
    except Exception as e:
        print(f"[!] Warning: Could not generate matplotlib heatmap: {e}")
        # Create a simple SVG or fallback image if matplotlib fails
        create_fallback_corr_image()


def create_fallback_corr_image():
    """Fallback generator for eda_corr.png if matplotlib is not installed."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (800, 600), color=(15, 23, 42))
        d = ImageDraw.Draw(img)
        d.text((40, 40), "Stock Predictor - Feature Correlation Matrix", fill=(255, 255, 255))
        d.text((40, 80), "Generated automatically during clean_data.py execution.", fill=(148, 163, 184))
        d.rectangle([40, 120, 760, 560], outline=(51, 65, 85), width=2)
        d.text((100, 260), "Install matplotlib & seaborn to render high-res graphical heatmap", fill=(56, 189, 248))
        img.save(EDA_CORR_IMG_PATH)
        print(f"[+] Created fallback placeholder correlation image at {EDA_CORR_IMG_PATH}")
    except Exception:
        pass


def clean_and_process():
    """Main execution function for cleaning and feature engineering."""
    if not os.path.exists(RAW_DATA_PATH):
        print(f"[!] Error: Raw data not found at {RAW_DATA_PATH}. Running fetch_data.py first...")
        import fetch_data
        fetch_data.fetch_stock_data()

    print(f"[*] Reading raw market data from {RAW_DATA_PATH}...")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"    Raw records count: {len(df_raw)}")

    processed_list = []
    for ticker, group in df_raw.groupby("Ticker"):
        print(f"    -> Processing & feature engineering for {ticker}...")
        processed_ticker = engineer_ticker_features(group)
        processed_list.append(processed_ticker)

    df_cleaned = pd.concat(processed_list, ignore_index=True)

    # Remove outliers on engineered ratio features
    numeric_features_to_clip = ["Daily_Return", "Volatility_10", "Spread"]
    df_cleaned = remove_outliers(df_cleaned, numeric_features_to_clip)

    # 1. Save full cleaned series (including present day) to cleaned_data.csv
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"[+] Saved full cleaned dataset ({len(df_cleaned)} rows) to {CLEANED_DATA_PATH}")

    # 2. Extract and save the PRESENT DAY (latest row per ticker) as unseen out-of-sample data
    latest_present_rows = df_cleaned.sort_values("Date").groupby("Ticker").last().reset_index()
    latest_unseen_path = os.path.join(DATA_DIR, "latest_unseen_data.csv")
    latest_present_rows.to_csv(latest_unseen_path, index=False)
    print(f"[+] Saved {len(latest_present_rows)} present-day tickers to {latest_unseen_path} for forward inference.")

    # 3. Generate EDA report & visual heatmap (using completed historical rows)
    df_with_targets = df_cleaned.dropna(subset=["Target_Next_Close"]).copy()
    generate_eda_artifacts(df_with_targets)
    print("[+] Data cleaning & feature engineering complete.")


if __name__ == "__main__":
    clean_and_process()
