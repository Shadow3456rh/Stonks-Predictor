# 📈 Stock Predictor — Automated ML Pipeline & Telemetry Dashboard

An automated, local-first Machine Learning engineering project built for academic demonstration and grading. 

**Stock Predictor** demonstrates a production-architected MLOps lifecycle: scheduled data ingestion, outlier rejection, automated feature engineering, multi-model benchmarking (Linear Regression, KNN, Decision Tree, Random Forest, and XGBoost), automated model promotion, and dual-portal serving (Consumer Trader Dashboard + MLOps Ops Console) via Flask.

---

## 🏛️ System Architecture & Workflow

The entire lifecycle executes **autonomously on GitHub Actions runners** (or via a local background engine for live offline presentations), committing validated model artifacts back to the repository without requiring a personal laptop or local cron daemon to remain powered on.

```
                      +---------------------------------------+
                      |       GitHub Actions Runner (Cron)     |
                      |     Schedule: 0 0 * * * / On-Demand    |
                      +---------------------------------------+
                                          |
                                          v
                              [1] fetch_data.py
                       (yfinance ticker ingestion / OHLCV)
                                          |
                                          v  --> data/raw_data.csv & last_updated.txt
                              [2] clean_data.py
                    (Outliers, MAs, Volatility, Spread, Target)
                                          |
                                          v  --> data/cleaned_data.csv & eda_summary.json
                                             --> static/eda_corr.png
                             [3] train_models.py
                  (Linear Reg, KNN, Decision Tree, RF, XGBoost)
                                          |
                                          v  --> models/<model_name>.pkl & metrics.json
                           [4] deploy_best_model.py
                      (R² / RMSE Evaluation & Selection)
                                          |
                                          v  --> models/deployed_model.pkl
                                             --> models/deployment_log.json
                                          |
                      +---------------------------------------+
                      |    Git Auto-Commit & Push Back to Repo |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |          Local / Hosted Flask App     |
                      +---------------------------------------+
                         /                    \
                        v                      v
             [Trader Portal: /dashboard]   [Admin Console: /admin/dashboard]
             - Chart.js Actual vs Pred    - Telemetry & Retrain Trigger
             - Tomorrow's Price & Delta   - Model Benchmark Leaderboard
             - Direction Pill (UP/DOWN)   - Deployment History & EDA
```

---

## 💻 Technology Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Backend** | Python 3.11+, Flask | Micro-framework serving routes, APIs, and templating |
| **Data & ML** | Pandas, NumPy, Scikit-Learn, XGBoost, Joblib | Pipeline processing, model training, evaluation |
| **Data Ingestion** | `yfinance` | Automated financial market time-series ingestion |
| **Visualization** | Matplotlib, Seaborn, Chart.js 4.4 | Static correlation heatmaps & interactive frontend charts |
| **Automation** | GitHub Actions (`pipeline.yml`) | Headless scheduled execution & artifact committing |
| **Frontend** | Jinja2, Semantic HTML5, Vanilla CSS | Dual consumer/admin visual themes (zero heavy JS framework) |
| **Database & Auth**| SQLite3, `werkzeug.security` | Salted SHA-256 password hashing & isolated sessions |

---

## 🧠 Machine Learning Algorithms Evaluated

Each pipeline run benchmarks 5 candidate regressors against an out-of-sample chronological test set:

1. **Linear Regression (`LinearRegression`)**: Scaled baseline evaluating linear combinations of technical moving averages and spreads.
2. **K-Nearest Neighbors (`KNeighborsRegressor`)**: Instance-based model identifying $k=5$ historical market days with similar volatility regimes using distance-weighted voting.
3. **Decision Tree Regressor (`DecisionTreeRegressor`)**: Non-linear axis-aligned partitioning with bounded tree depth (`max_depth=6`).
4. **Random Forest Regressor (`RandomForestRegressor`)**: Bagging ensemble of 100 decorrelated trees reducing variance in volatile time-series.
5. **XGBoost Regressor (`XGBRegressor`)**: Gradient boosted trees minimizing loss gradients with regularized trees for high predictive precision.

### Evaluation Metrics Computed
- **$R^2$ (Coefficient of Determination)**: Proportion of variance explained by the model (primary promotion criterion).
- **RMSE (Root Mean Squared Error)**: Penalizes larger prediction residuals in currency units ($).
- **MAE (Mean Absolute Error)**: Average absolute magnitude of dollar forecast error.
- **Inference Latency (ms)**: Millisecond benchmark for test set predictions, ensuring production responsiveness.

---

## 🔒 Security & Session Isolation Architecture

To adhere to enterprise security practices:
- **Zero Plaintext Admin Credentials**: The administrator password is **never stored in the repository**. It is loaded dynamically from the `ADMIN_PASSWORD` environment variable.
- **Strictly Separated Sessions**:
  - Regular users sign in at `/login` and receive a `session['user_id']` token, protected by the `@login_required` decorator.
  - Administrators sign in at `/admin` and receive a `session['admin_id']` token, protected by the `@admin_required` decorator.
  - A regular user session **cannot access** `/admin/dashboard`, and an administrator session cannot view user views without logging in separately.
- **Local Credential Storage**: Regular users are stored in `users.db` with salted password hashes generated via `generate_password_hash`.

---

## 📁 Repository Structure

```
stock-predictor/
├── .github/
│   └── workflows/
│       └── pipeline.yml          # GitHub Actions scheduled cron & workflow_dispatch
├── data/
│   ├── raw_data.csv              # Raw market OHLCV feed from yfinance
│   ├── cleaned_data.csv          # Feature-engineered dataset with technical indicators
│   ├── eda_summary.json          # Statistical distributions & correlations
│   └── last_updated.txt          # Timestamp of the latest pipeline run
├── models/
│   ├── Linear_Regression.pkl     # Serialized candidate model
│   ├── KNN_Regressor.pkl         # Serialized candidate model
│   ├── Decision_Tree.pkl         # Serialized candidate model
│   ├── Random_Forest.pkl         # Serialized candidate model
│   ├── XGBoost.pkl               # Serialized candidate model
│   ├── deployed_model.pkl        # Active champion model promoted for inference
│   ├── metrics.json              # Full evaluation benchmark scores for all 5 models
│   └── deployment_log.json       # Append-only deployment audit trail
├── static/
│   ├── style.css                 # Consumer finance theme & design system
│   ├── admin.css                 # Dark MLOps telemetry & ops theme
│   └── eda_corr.png              # Feature correlation matrix heatmap
├── templates/
│   ├── base.html                 # Common layout with adaptive nav headers
│   ├── index.html                # Project landing page & architecture explanation
│   ├── login.html                # Regular user login & registration modal
│   ├── admin_login.html          # Distinct administrative authentication portal
│   ├── dashboard.html            # Trader dashboard (Chart.js live charts & forecast card)
│   └── admin_dashboard.html      # Ops dashboard (Leaderboard, EDA, live run trigger)
├── app.py                        # Flask web application & REST APIs
├── auth.py                       # User DB helper, hashing, and session decorators
├── fetch_data.py                 # Step 1: Historical market ingestion
├── clean_data.py                 # Step 2: Feature engineering & EDA generation
├── train_models.py               # Step 3: Candidate models training & benchmarking
├── deploy_best_model.py          # Step 4: Metric-based promotion & logging
├── users.db                      # Local SQLite user storage
├── requirements.txt              # Production dependency specifications
└── README.md                     # Comprehensive documentation
```

---

## 🚀 Quickstart: Running Locally for Demo & Grading

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/stock-predictor.git
cd stock-predictor

# Create and activate Python virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Scripts Manually (Optional / Initial Run)

You can run the 4-step pipeline locally at any time:

```bash
# Step 1: Ingest stock data
python fetch_data.py

# Step 2: Clean and engineer features
python clean_data.py

# Step 3: Train and benchmark 5 candidate models
python train_models.py

# Step 4: Promote and deploy the best model
python deploy_best_model.py
```

### 4. Configure Environment Variables (Optional)

You can set an administrative password and Flask secret key:

```bash
# On Windows PowerShell:
$env:ADMIN_PASSWORD="adminpass123"
$env:ADMIN_USERNAME="admin"

# On Linux/macOS:
export ADMIN_PASSWORD="adminpass123"
export ADMIN_USERNAME="admin"
```
*(If omitted, sensible local defaults like `adminpass123` are provided so the application runs immediately).*

### 5. Start the Flask Web Application

```bash
python app.py
```

Open your browser at: **`http://localhost:5000`**

---

## 🔑 Demo Login Credentials

For quick evaluation during project grading:

### Trader Account (Consumer Dashboard)
- **URL**: `http://localhost:5000/login`
- **Email**: `user@gmail.com`
- **Password**: `user123`
*(Or click the demo chip button on the login screen to autofill)*

### Administrator Account (Ops Telemetry)
- **URL**: `http://localhost:5000/admin`
- **Username**: `admin`
- **Password**: `adminpass123`
*(Or click the "Insert Admin Credentials" button on the screen)*

---

## ⚙️ Configuring GitHub Actions Automation

To enable full automation on GitHub:

1. **Push the project to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Stock Predictor Automated ML Pipeline"
   git remote add origin https://github.com/<your-username>/stock-predictor.git
   git push -u origin main
   ```

2. **Enable Workflow Write Permissions**:
   - Go to your repository **Settings** -> **Actions** -> **General**.
   - Under **Workflow permissions**, select **"Read and write permissions"** and check **"Allow GitHub Actions to create and approve pull requests"**.
   - Click **Save**.

3. **(Optional) Set Up Secrets for Admin Remote Dispatch**:
   - Under **Settings** -> **Secrets and variables** -> **Actions**, you can define:
     - `ADMIN_PASSWORD`: Your secret production admin password.
     - `GH_PAT`: A GitHub Personal Access Token (with `repo` scope) if you want the Flask app's "Run Pipeline Now" button to dispatch GitHub Actions workflows remotely.

4. **Verify Automated Runs**:
   - Click the **Actions** tab in GitHub.
   - You will see the **Stock Predictor Automated ML Pipeline** workflow.
   - It will trigger automatically on schedule every midnight UTC.
   - You can also click **"Run workflow"** manually to trigger it on-demand for live demonstrations!

---

## 📊 Live Interactive Features

- **Interactive Chart.js Time-Series**: Smooth rendering of actual stock prices alongside the model's out-of-sample predictions.
- **Instant Asynchronous Ticker Switching**: Changing the ticker selector updates predictions and charts in real-time via `/api/predict/<ticker>`.
- **Directional Trend Pills**: Automatically calculates percentage delta and highlights expected movement (`UP` / `DOWN`).
- **Live Terminal Log Streaming**: The Admin Console includes a "Run Pipeline Now" trigger with real-time step progress and log streaming.
- **Model Leaderboard**: Inspect $R^2$, RMSE, MAE, inference latency, and training duration for all 5 algorithms side by side.
