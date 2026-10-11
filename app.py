"""
app.py
======
Flask Application Server for Stock Predictor
Handles:
- Public landing page (/)
- Separate regular user and admin authentication flows (/login, /admin)
- Consumer finance user dashboard (/dashboard) with Chart.js and prediction cards
- High-density admin ops dashboard (/admin/dashboard) with leaderboard, EDA summary,
  deployment history, and real-time GitHub Actions / local pipeline execution.
- REST APIs for AJAX charts and asynchronous pipeline execution polling.
"""

import os
import json
import time
import threading
import datetime
import requests
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify

from auth import (
    init_db,
    register_user,
    authenticate_user,
    authenticate_admin,
    login_required,
    admin_required
)

# Application Initialization
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "stock-predictor-academic-secret-key-2025")

# Paths to storage artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_data.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
TIMESTAMP_PATH = os.path.join(DATA_DIR, "last_updated.txt")
EDA_SUMMARY_PATH = os.path.join(DATA_DIR, "eda_summary.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
DEPLOYED_MODEL_PATH = os.path.join(MODELS_DIR, "deployed_model.pkl")
DEPLOYMENT_LOG_PATH = os.path.join(MODELS_DIR, "deployment_log.json")
MODEL_RUN_LOG_PATH = os.path.join(DATA_DIR, "model_run_log.txt")

# In-memory execution state for real-time pipeline monitoring
pipeline_state = {
    "is_running": False,
    "source": None,          # "github_actions" or "local"
    "progress": 0,           # 0 to 100 percentage
    "current_step": "Idle",
    "logs": [],
    "last_run_timestamp": None,
    "status": "idle"         # "idle", "running", "success", "failed"
}


# ==============================================================================
# Helper Functions: Data & Model Ingestion
# ==============================================================================

def get_last_updated_time():
    """Reads the timestamp recorded by the pipeline."""
    if os.path.exists(TIMESTAMP_PATH):
        try:
            with open(TIMESTAMP_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return "Never"


def get_eda_summary():
    """Loads eda_summary.json if available."""
    if os.path.exists(EDA_SUMMARY_PATH):
        try:
            with open(EDA_SUMMARY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_models_metrics():
    """Loads models/metrics.json if available."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_deployment_log():
    """Loads models/deployment_log.json history."""
    if os.path.exists(DEPLOYMENT_LOG_PATH):
        try:
            with open(DEPLOYMENT_LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
def get_model_run_logs(max_entries=10):
    """Loads recent 'model ran at YYYY-MM-DD : HH:MM:SS UTC' log entries."""
    if os.path.exists(MODEL_RUN_LOG_PATH):
        try:
            with open(MODEL_RUN_LOG_PATH, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                return lines[-max_entries:]
        except Exception:
            pass
    return []


def get_available_tickers():
    """Extracts unique tickers from cleaned or raw data."""
    if os.path.exists(CLEANED_DATA_PATH):
        try:
            df = pd.read_csv(CLEANED_DATA_PATH, usecols=["Ticker"])
            tickers = sorted(df["Ticker"].unique().tolist())
            if tickers:
                return tickers
        except Exception:
            pass
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


def generate_ticker_predictions(ticker):
    """
    Computes actual vs predicted series and next-day forecast for the given ticker.
    Uses the currently deployed model artifact (deployed_model.pkl).
    """
    if not os.path.exists(CLEANED_DATA_PATH) or not os.path.exists(DEPLOYED_MODEL_PATH):
        return None

    try:
        try:
            model = joblib.load(DEPLOYED_MODEL_PATH)
        except Exception:
            # Fallback to pure scikit-learn models if deployed model requires missing serverless module (e.g. xgboost)
            model = None
            for alt_name in ["Linear_Regression.pkl", "Random_Forest.pkl", "Decision_Tree.pkl", "KNN_Regressor.pkl"]:
                alt_path = os.path.join(MODELS_DIR, alt_name)
                if os.path.exists(alt_path):
                    try:
                        model = joblib.load(alt_path)
                        break
                    except Exception:
                        continue
            if model is None:
                return None

        df = pd.read_csv(CLEANED_DATA_PATH)
        df_ticker = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)

        if df_ticker.empty:
            return None

        feature_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "MA_5", "MA_10", "MA_20", "MA_50",
            "Daily_Return", "Volatility_10", "Spread"
        ]

        # Historical subset for charting (last 60 trading days for clarity)
        chart_window = 60
        df_chart = df_ticker.tail(chart_window).copy()

        X_chart = df_chart[feature_cols]
        predicted_targets = model.predict(X_chart)

        # Build chart series
        dates = df_chart["Date"].tolist()
        actual_prices = [round(float(p), 2) for p in df_chart["Close"]]
        # In this dataset, prediction at day t was for day t+1
        predicted_prices = [round(float(p), 2) for p in predicted_targets]

        # Next-day forecast using the most recent data point
        latest_row = df_ticker.iloc[-1:]
        X_latest = latest_row[feature_cols]
        next_day_forecast = float(model.predict(X_latest)[0])

        current_price = float(latest_row["Close"].values[0])
        price_diff = next_day_forecast - current_price
        pct_change = (price_diff / current_price) * 100 if current_price else 0.0
        direction = "UP" if price_diff >= 0 else "DOWN"

        return {
            "ticker": ticker,
            "dates": dates,
            "actual_prices": actual_prices,
            "predicted_prices": predicted_prices,
            "current_price": round(current_price, 2),
            "next_day_predicted": round(next_day_forecast, 2),
            "price_diff": round(price_diff, 2),
            "pct_change": round(pct_change, 2),
            "direction": direction,
            "latest_date": dates[-1] if dates else "N/A"
        }
    except Exception as e:
        print(f"[!] Error in generate_ticker_predictions for {ticker}: {e}")
        return None


# ==============================================================================
# Public & Authentication Routes
# ==============================================================================

@app.route("/")
def index():
    """Landing page introducing Stock Predictor and its architecture."""
    last_updated = get_last_updated_time()
    metrics = get_models_metrics()
    deployment_log = get_deployment_log()
    current_model = deployment_log[-1]["deployed_model"] if deployment_log else "None Deployed"

    return render_template(
        "index.html",
        last_updated=last_updated,
        metrics=metrics,
        current_model=current_model
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    """Regular user login and registration."""
    if session.get("user_id"):
        return redirect(url_for("user_dashboard"))

    if request.method == "POST":
        action = request.form.get("action", "login")
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if action == "register":
            success, msg = register_user(email, password)
            flash(msg, "success" if success else "danger")
            return redirect(url_for("login"))
        else:
            user = authenticate_user(email, password)
            if user:
                session["user_id"] = user["id"]
                session["user_email"] = user["email"]
                flash(f"Welcome back, {user['email']}!", "success")
                return redirect(url_for("user_dashboard"))
            else:
                flash("Invalid email or password. Please try again.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    """Clears regular user session."""
    session.pop("user_id", None)
    session.pop("user_email", None)
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    """Separate dedicated login route for administrators."""
    if session.get("admin_id"):
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        admin = authenticate_admin(username, password)
        if admin:
            session["admin_id"] = admin["username"]
            session["admin_username"] = admin["username"]
            flash("Administrator session authenticated.", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid administrator credentials.", "danger")

    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    """Clears administrator session."""
    session.pop("admin_id", None)
    session.pop("admin_username", None)
    flash("Administrator session closed.", "info")
    return redirect(url_for("admin_login"))


# ==============================================================================
# User Dashboard Routes
# ==============================================================================

@app.route("/dashboard")
@login_required
def user_dashboard():
    """User finance dashboard: Chart.js actual vs predicted prices & forecast card."""
    tickers = get_available_tickers()
    selected_ticker = request.args.get("ticker", tickers[0] if tickers else "AAPL").upper()
    if selected_ticker not in tickers and tickers:
        selected_ticker = tickers[0]

    last_updated = get_last_updated_time()
    deployment_log = get_deployment_log()
    current_model = deployment_log[-1]["deployed_model"] if deployment_log else "Linear_Regression"
    current_metrics = deployment_log[-1].get("metrics_at_deployment", {}) if deployment_log else {}

    prediction_data = generate_ticker_predictions(selected_ticker)

    return render_template(
        "dashboard.html",
        user_email=session.get("user_email"),
        tickers=tickers,
        selected_ticker=selected_ticker,
        last_updated=last_updated,
        current_model=current_model,
        current_metrics=current_metrics,
        prediction_data=prediction_data
    )


@app.route("/api/predict/<ticker>")
@login_required
def api_predict(ticker):
    """JSON API endpoint for asynchronous ticker chart switching."""
    data = generate_ticker_predictions(ticker.upper())
    if not data:
        return jsonify({"error": f"No forecast data available for {ticker}"}), 404
    return jsonify(data)


# ==============================================================================
# Admin Dashboard & Pipeline Orchestration Routes
# ==============================================================================

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """
    Admin ops and telemetry dashboard:
    - Pipeline status indicator
    - Model leaderboard table (all 4-5 models with deployed highlighted)
    - Deployment history timeline
    - EDA summary statistics and correlation heatmap
    - Real-time pipeline trigger button
    """
    last_updated = get_last_updated_time()
    metrics_data = get_models_metrics()
    deployment_history = get_deployment_log()
    eda_data = get_eda_summary()

    # Determine currently deployed model name
    deployed_model_name = deployment_history[-1]["deployed_model"] if deployment_history else None

    # Calculate model history chart frequencies
    history_counts = {}
    for entry in deployment_history:
        m = entry.get("deployed_model", "Unknown")
        history_counts[m] = history_counts.get(m, 0) + 1

    return render_template(
        "admin_dashboard.html",
        admin_username=session.get("admin_username"),
        last_updated=last_updated,
        metrics_data=metrics_data,
        deployment_history=deployment_history,
        deployed_model_name=deployed_model_name,
        history_counts=history_counts,
        eda_data=eda_data,
        pipeline_state=pipeline_state,
        model_run_logs=get_model_run_logs()
    )


def run_local_pipeline_worker():
    """Background worker to execute the ML pipeline locally for demo/grading purposes."""
    global pipeline_state
    pipeline_state["is_running"] = True
    pipeline_state["status"] = "running"
    pipeline_state["progress"] = 5
    pipeline_state["current_step"] = "Initializing Pipeline Environment"
    pipeline_state["logs"] = ["[INIT] Starting automated pipeline runner..."]

    steps = [
        ("Step 1/4: Ingesting Market Data via fetch_data.py", "fetch_data", 25),
        ("Step 2/4: Cleaning & Generating EDA via clean_data.py", "clean_data", 50),
        ("Step 3/4: Training Models via train_models.py", "train_models", 80),
        ("Step 4/4: Promoting Best Model via deploy_best_model.py", "deploy_best_model", 100)
    ]

    try:
        import fetch_data
        pipeline_state["current_step"] = steps[0][0]
        pipeline_state["progress"] = steps[0][2]
        pipeline_state["logs"].append(f"[EXEC] Running fetch_data...")
        fetch_data.fetch_stock_data()
        pipeline_state["logs"].append(f"[DONE] Raw market data saved.")

        import clean_data
        pipeline_state["current_step"] = steps[1][0]
        pipeline_state["progress"] = steps[1][2]
        pipeline_state["logs"].append(f"[EXEC] Running clean_data and generating EDA artifacts...")
        clean_data.clean_and_process()
        pipeline_state["logs"].append(f"[DONE] Cleaned data & eda_corr.png updated.")

        import train_models
        pipeline_state["current_step"] = steps[2][0]
        pipeline_state["progress"] = steps[2][2]
        pipeline_state["logs"].append(f"[EXEC] Training candidate regression models...")
        train_models.train_and_evaluate()
        pipeline_state["logs"].append(f"[DONE] Model training and benchmarking completed.")

        import deploy_best_model
        pipeline_state["current_step"] = steps[3][0]
        pipeline_state["progress"] = steps[3][2]
        pipeline_state["logs"].append(f"[EXEC] Selecting best model and deploying...")
        log_entry = deploy_best_model.select_and_deploy()
        pipeline_state["logs"].append(f"[DEPLOY] Best model deployed: {log_entry['deployed_model']}.")

        pipeline_state["current_step"] = "Pipeline Run Complete"
        pipeline_state["status"] = "success"
        pipeline_state["progress"] = 100
        pipeline_state["last_run_timestamp"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        pipeline_state["logs"].append("[COMPLETE] Automated pipeline finished successfully.")
    except Exception as e:
        pipeline_state["status"] = "failed"
        pipeline_state["current_step"] = f"Error: {str(e)}"
        pipeline_state["logs"].append(f"[ERROR] Pipeline execution failed: {str(e)}")
    finally:
        pipeline_state["is_running"] = False


@app.route("/api/admin/trigger-pipeline", methods=["POST"])
@admin_required
def trigger_pipeline():
    """
    Triggers GitHub Actions workflow_dispatch via REST API if configured;
    falls back seamlessly to local asynchronous execution for live offline demos.
    """
    global pipeline_state
    if pipeline_state["is_running"]:
        return jsonify({"status": "already_running", "message": "Pipeline is already actively executing."}), 409

    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "auto")  # 'github', 'local', or 'auto'

    gh_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_PAT")
    repo_owner = os.environ.get("GITHUB_REPO_OWNER")
    repo_name = os.environ.get("GITHUB_REPO_NAME", "stock-predictor")

    # If GitHub credentials provided and mode is github or auto, dispatch to GitHub Actions
    if mode in ("github", "auto") and gh_token and repo_owner:
        try:
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/pipeline.yml/dispatches"
            headers = {
                "Authorization": f"Bearer {gh_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            payload = {
                "ref": "main",
                "inputs": {
                    "deployment_criterion": "r2"
                }
            }
            res = requests.post(url, headers=headers, json=payload, timeout=10)
            if res.status_code in (204, 201, 200):
                pipeline_state["is_running"] = True
                pipeline_state["source"] = "github_actions"
                pipeline_state["status"] = "running"
                pipeline_state["current_step"] = "GitHub Actions Workflow Dispatched"
                pipeline_state["logs"] = [f"[GITHUB] Workflow dispatched to https://github.com/{repo_owner}/{repo_name}/actions"]
                return jsonify({
                    "status": "dispatched",
                    "source": "github_actions",
                    "message": f"GitHub Actions workflow triggered on repo {repo_owner}/{repo_name}."
                })
            else:
                print(f"[!] GitHub Actions dispatch failed with HTTP {res.status_code}: {res.text}. Falling back to local execution.")
        except Exception as gh_err:
            print(f"[!] GitHub API error: {gh_err}. Falling back to local runner.")

    # Local fallback execution in background thread
    pipeline_state["source"] = "local"
    thread = threading.Thread(target=run_local_pipeline_worker, daemon=True)
    thread.start()

    return jsonify({
        "status": "started",
        "source": "local",
        "message": "Local background pipeline runner initiated successfully."
    })


@app.route("/api/admin/pipeline-status")
@admin_required
def get_pipeline_status():
    """Returns current pipeline state and execution logs for admin UI polling."""
    return jsonify({
        "is_running": pipeline_state["is_running"],
        "status": pipeline_state["status"],
        "progress": pipeline_state["progress"],
        "current_step": pipeline_state["current_step"],
        "logs": pipeline_state["logs"],
        "last_updated": get_last_updated_time(),
        "source": pipeline_state["source"]
    })


# ==============================================================================
# Initialization Hook
# ==============================================================================

with app.app_context():
    init_db()


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
