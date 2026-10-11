# 👥 Project Presentation & Task Separation Guide
## Project: Stock Predictor — Automated ML Pipeline & Telemetry Dashboard

This document divides the presentation and code walkthrough of the **Stock Predictor** project among **3 team members**. Each section provides assigned files, key talking points, code highlights to demonstrate, and likely viva/teacher questions.

---

```
                       ┌──────────────────────────────────────────────┐
                       │           Stock Predictor Project            │
                       └──────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
   ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
   │    PERSON 1     │       │    PERSON 2     │       │    PERSON 3     │
   │  Data Pipeline  │       │    ML & MLOps   │       │  Web App & UI   │
   │   & Feature Eng │       │   Architecture  │       │   Experience    │
   └─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

## 🧑‍💻 PERSON 1: Data Ingestion, Cleaning & Feature Engineering

### 📌 Assigned Files
- `fetch_data.py`
- `clean_data.py`
- `data/raw_data.csv`
- `data/cleaned_data.csv`
- `data/latest_unseen_data.csv`
- `data/eda_summary.json`
- `static/eda_corr.png`

### 🎯 Primary Responsibilities & Talking Points
1. **Automated Market Ingestion (`fetch_data.py`)**:
   - Explain how `yfinance` fetches historical OHLCV (Open, High, Low, Close, Volume) data for configurable tickers (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`).
   - Mention the built-in **resilient fallback generator** that prevents pipeline crashes if network timeouts or market holidays occur.
   - Explain that data is stored in a clean tabular schema in `data/raw_data.csv`.

2. **Data Preprocessing & Outlier Handling (`clean_data.py`)**:
   - Forward-fill and backward-fill missing data across weekends and holidays.
   - Outlier filtering/clipping using **Interquartile Range (IQR)** bounding on ratio features (`Daily_Return`, `Volatility_10`, `Spread`) to prevent extreme anomalous market spikes from distorting model weights.

3. **Feature Engineering (12 Technical Indicators)**:
   - **Trend Moving Averages**: Short-term `MA_5`, `MA_10` and medium-term `MA_20`, `MA_50` to smooth out noise.
   - **Momentum**: `Daily_Return = (Close_t - Close_{t-1}) / Close_{t-1}`.
   - **Risk Metric**: `Volatility_10` (10-day rolling standard deviation of returns).
   - **Intraday Volatility Ratio**: `Spread = (High - Low) / Close`.
   - **Prediction Target**: `Target_Next_Close` (the next trading day's closing price, `Close.shift(-1)`).

4. **Critical Academic Concept — Present Day Holdout (`latest_unseen_data.csv`)**:
   - **Crucial point for teachers**: Explain that the **present day** (the latest market close) has an unknown future target (`Target_Next_Close` is NaN).
   - We explicitly extract and save this day to `data/latest_unseen_data.csv` and **exclude it from model training**.
   - This guarantees that tomorrow's live prediction is evaluated on **100% unseen, out-of-sample data with zero lookahead bias**.

5. **Exploratory Data Analysis (EDA)**:
   - Point out `static/eda_corr.png` (the Seaborn correlation heatmap) showing strong correlation between moving averages and target price, and the distribution statistics in `data/eda_summary.json`.

### 🎓 Likely Viva / Teacher Questions for Person 1
* **Q: Why did you use moving averages (5, 10, 20, 50)?**  
  *Answer:* Moving averages capture price momentum across different horizons. Short-term (5, 10) reflects recent price momentum, while long-term (20, 50) captures macro baseline trends.
* **Q: How did you prevent data leakage / lookahead bias?**  
  *Answer:* We strictly computed indicators chronologically using past rolling windows. More importantly, the present day was completely removed from the training split and stored separately in `latest_unseen_data.csv`.

---

## 🤖 PERSON 2: Machine Learning Models & MLOps Automation

### 📌 Assigned Files
- `train_models.py`
- `deploy_best_model.py`
- `models/metrics.json`
- `models/deployment_log.json`
- `models/*.pkl` and `models/deployed_model.pkl`
- `.github/workflows/pipeline.yml`
- `data/model_run_log.txt`
- `vercel.json`

### 🎯 Primary Responsibilities & Talking Points
1. **Multi-Model Candidate Training (`train_models.py`)**:
   - Explain the 5 candidate regression algorithms evaluated in each cycle:
     1. **Linear Regression** (Parametric baseline with `StandardScaler`).
     2. **K-Nearest Neighbors (`KNN_Regressor`)** (Instance-based pattern matching with distance weighting, $k=5$).
     3. **Decision Tree Regressor** (Non-linear orthogonal boundaries, `max_depth=6`).
     4. **Random Forest Regressor** (Bagging ensemble of 100 decorrelated trees reducing variance).
     5. **XGBoost Regressor (`XGBRegressor`)** (Gradient boosted trees with regularization).
   - **Time-Series Split**: Explain why standard random `train_test_split(shuffle=True)` is invalid for stock data. We use a chronological 80% train / 20% test split to honor temporal causality.

2. **Model Evaluation & Benchmarking**:
   - Explain the 4 metrics computed for each model in `models/metrics.json`:
     - **$R^2$ Score**: Proportion of variance explained (e.g., Linear Reg: 0.9857, RF: 0.9816).
     - **RMSE ($)**: Root Mean Squared Error penalizing large dollar errors.
     - **MAE ($)**: Mean Absolute Error in dollar units.
     - **Inference Latency (ms)**: Millisecond benchmark per prediction batch.

3. **Automated Promotion Engine (`deploy_best_model.py`)**:
   - Reads `models/metrics.json`, compares candidate models using a configurable criterion ($R^2$ or lowest RMSE), promotes the champion to `models/deployed_model.pkl`, and appends an audit trail to `models/deployment_log.json`.

4. **CI/CD Pipeline Automation (`.github/workflows/pipeline.yml`)**:
   - **GitHub Actions Scheduled Cron**: Configured with `0 13-21 * * 1-5` to run **hourly during US stock market trading hours (Mon–Fri)** on headless Ubuntu runners.
   - **Automatic Git Push-Back**: Automatically commits new `.pkl` models, metrics, and logs back to the repository.
   - **Execution Logging**: Logs every execution timestamp to `data/model_run_log.txt` (`model ran at YYYY-MM-DD : HH:MM:SS UTC`).
   - **Vercel Cloud CD**: Whenever GitHub Actions pushes fresh models, Vercel detects the commit and redeploys the web application automatically.

### 🎓 Likely Viva / Teacher Questions for Person 2
* **Q: Why did you benchmark 5 different algorithms instead of just training one?**  
  *Answer:* Financial markets experience regime changes between trending and volatile states. Linear Regression excels in steady trends, while Random Forest and XGBoost handle non-linear volatility regimes. Continuous benchmarking guarantees that whichever model performs best on out-of-sample data is automatically deployed.
* **Q: Why does the pipeline not require a laptop to be running 24/7?**  
  *Answer:* Because the orchestration is completely offloaded to GitHub Actions' cloud VMs via cron schedules. The repository acts as the single source of truth for both code and storage.

---

## 🎨 PERSON 3: Web Application Architecture, Security & User Interfaces

### 📌 Assigned Files
- `app.py`
- `auth.py`
- `users.db`
- `templates/base.html`
- `templates/index.html`
- `templates/login.html` & `templates/admin_login.html`
- `templates/dashboard.html` & `templates/admin_dashboard.html`
- `static/style.css` & `static/admin.css`

### 🎯 Primary Responsibilities & Talking Points
1. **Flask Architecture & REST APIs (`app.py`)**:
   - Explain how Flask serves dynamic Jinja2 templates without heavy frontend framework bloat.
   - Walk through the key API endpoints:
     - `/api/predict/<ticker>`: Asynchronously feeds the deployed model with features and returns predictions for interactive chart rendering.
     - `/api/admin/trigger-pipeline`: Dispatches on-demand GitHub Actions workflows or executes local background runner threads.
     - `/api/admin/pipeline-status`: Polling endpoint for real-time progress bars and log streams.

2. **Security & Session Isolation (`auth.py`)**:
   - **Strict Session Separation**:
     - Regular User: `session['user_id']`, protected by `@login_required`.
     - Administrator: `session['admin_id']`, protected by `@admin_required`.
     - A logged-in user can never access admin routes, and vice versa.
   - **Password Security**: Passwords stored as salted SHA-256 hashes in `users.db` using `werkzeug.security`.
   - **Zero Plaintext Admin Passwords**: Admin password is read dynamically from the `ADMIN_PASSWORD` environment variable.
   - **Serverless Compatibility**: On Vercel, `auth.py` automatically initializes SQLite in `/tmp/users.db` to prevent read-only filesystem errors.

3. **Consumer Trader Dashboard (`templates/dashboard.html` & `style.css`)**:
   - Clean consumer finance aesthetic with card elevation and soft typography (Inter font).
   - **Centerpiece Interactive Chart (Chart.js)**: Displays actual historical prices vs out-of-sample model predictions.
   - **Tomorrow's Prediction Card**: Shows tomorrow's predicted price, dollar delta, and color-coded direction badges (`▲ UP` or `▼ DOWN`).
   - **AJAX Ticker Switcher**: Changing the dropdown (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`) updates the charts and cards instantly without refreshing the page.

4. **Administrator & MLOps Console (`templates/admin_dashboard.html` & `admin.css`)**:
   - Dark cybersecurity / ops theme (Slate/Zinc palette with telemetry indicators).
   - **Model Leaderboard**: Highlights the deployed champion model with side-by-side metric comparisons ($R^2$, RMSE, MAE, Latency).
   - **Deployment History Chart**: Visualizes model promotion frequency.
   - **EDA Heatmap Preview**: Embedded inspection of `static/eda_corr.png`.
   - **"Run Pipeline Now" Demo Trigger**: Live demonstration button showing an active progress bar and terminal log stream.
   - **Automated Model Execution Log**: Displays the live history from `data/model_run_log.txt`.

### 🎓 Likely Viva / Teacher Questions for Person 3
* **Q: Why did you implement two completely separate login systems instead of one role column in the database?**  
  *Answer:* For strict privilege separation and defense-in-depth. Regular users authenticate against a database, while admin access requires an infrastructure-level secret (`ADMIN_PASSWORD` environment variable), preventing privilege escalation even if the database is compromised.
* **Q: How does the chart update without reloading the entire page?**  
  *Answer:* The dropdown triggers an `onchange` JavaScript event that fetches data from `/api/predict/<ticker>`. Chart.js updates its datasets and re-renders smoothly via the virtual DOM canvas.

---

## ⏱️ Recommended 10-Minute Presentation Flow

| Time | Presenter | Section | What to Show on Screen |
| :--- | :--- | :--- | :--- |
| **0:00 - 1:00** | **All / Person 1** | System Overview & Problem Statement | Public Landing Page (`/`) |
| **1:00 - 3:30** | **Person 1** | Data Ingestion, IQR Cleaning, 12 Features, Present-Day Holdout | `fetch_data.py`, `clean_data.py`, `static/eda_corr.png` |
| **3:30 - 6:00** | **Person 2** | 5 Algorithms, $R^2$/RMSE Metrics, Promotion Engine, GitHub Actions Cron | `train_models.py`, `models/metrics.json`, `pipeline.yml` |
| **6:00 - 8:30** | **Person 3** | Dual Auth, Trader Dashboard, Chart.js, Admin MLOps Console | Trader Dashboard (`/dashboard`), Admin Console (`/admin/dashboard`) |
| **8:30 - 10:00**| **Person 2 & 3** | Live Pipeline Run Demo & Q&A | Click "Run Pipeline Now" live in the Admin Dashboard |
