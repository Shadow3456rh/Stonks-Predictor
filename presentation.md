# 📊 Stock Predictor — Academic Presentation Deck
## Automated Time-Series Machine Learning Pipeline & Continuous MLOps Infrastructure

**Course**: Machine Learning Laboratory (5th Semester)  
**Project Repository**: `https://github.com/Shadow3456rh/Stonks-Predictor`  
**Live Cloud Deployment**: Vercel Serverless & GitHub Actions CI/CD  
**Presenters**: 
* **Person 1**: Data Pipeline, Cleaning, & Feature Engineering
* **Person 2**: Machine Learning Models, Evaluation, & MLOps CI/CD
* **Person 3**: Web Architecture, Security, UI/UX, & Cloud Deployment

---

## Slide 1: Title & System Overview
### Stock Predictor: Continuous MLOps Pipeline for Financial Time-Series Forecasting

```
+-----------------------------------------------------------------------------------+
|                        STOCK PREDICTOR MLOPS INFRASTRUCTURE                       |
|   Live Ingestion -> IQR Outlier Bounding -> 12 Features -> 5 Model Benchmarking   |
|         -> Automated Promotion -> Git Auto-Commit -> Dual-Role Web Serving        |
+-----------------------------------------------------------------------------------+
```

* **Core Innovation**: Moving beyond static Jupyter notebooks to an autonomous, cloud-orchestrated machine learning pipeline.
* **Key Capabilities**:
  - Live API ingestion across major US tech equities (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`).
  - Zero lookahead bias via strict chronological holdout of the present trading day.
  - 5-model concurrent evaluation (Linear Regression, KNN, Decision Tree, Random Forest, XGBoost).
  - Autonomous promotion of champion model based on $R^2$ / RMSE optimization.
  - Headless hourly scheduled retraining via GitHub Actions (`0 13-21 * * 1-5`).
  - Dual-role web portal on Flask, deployed serverless to Vercel.

> **🗣️ Speaker (All / Person 1)**:  
> *"Good morning respected evaluators. Today our group presents 'Stock Predictor'—an automated, production-architected time-series machine learning system. Unlike traditional student projects that train a single model on a static CSV file, our project builds a fully autonomous MLOps lifecycle that runs scheduled retraining in the cloud, benchmarks five distinct algorithms, and deploys the champion model directly to a live web platform."*

---

## Slide 2: Problem Statement & Literature Review
### Bridging the Gap: Academic ML vs. Production Financial MLOps

* **Presenter**: Person 1 (Data & Theory)
* **The Problem**:
  1. **Concept Drift**: Financial markets are non-stationary; static models quickly deteriorate.
  2. **Lookahead Bias**: Standard cross-validation with random shuffling leaks future prices into past training data.
  3. **High Infrastructure Cost**: Enterprise MLOps (SageMaker, Kubeflow) requires costly, always-on cloud servers.

#### Summary of 5 Contemporary Literature Papers Analyzed:
| Author & Year | Focus & Approach | Finding / Bottleneck | Solution in Our Project |
| :--- | :--- | :--- | :--- |
| **Nabipour et al. (2020)** | Tree ensembles vs. Deep Learning | Trees (RF, GBDT) outperform LSTMs on daily technical indicator features. | Integrated Decision Trees, Random Forest, and XGBoost. |
| **Mehtab & Sen (2022)** | Feature Engineering on S&P 500 | Multi-period moving averages (5, 10, 20, 50) and momentum yield $R^2 > 0.95$. | Engineered 12 features across trend, momentum, and volatility. |
| **Zhang et al. (2023)** | Regularized XGBoost on NASDAQ | Boosted trees minimize non-linear loss, but require $L_1/L_2$ leaf regularization. | Tuned XGBoost with `subsample=0.8` and bounded tree depth. |
| **Althelaya et al. (2021)**| Lookahead bias in financial ML | Random cross-validation inflates accuracy by 25%; chronological splits are mandatory. | Enforced strict chronological 80/20 train/test split. |
| **Gouws & Pretorius (2024)**| CI/CD in Quantitative Finance | Models require retraining at least monthly to mitigate concept drift. | Implemented hourly headless GitHub Actions retraining. |

> **🗣️ Speaker (Person 1)**:  
> *"In our literature survey of 5 recent papers, two major themes emerged: First, Nabipour and Mehtab proved that classical models and tree ensembles, when provided with rich moving average features, achieve accuracy on par with deep learning without extreme computational overhead. Second, Althelaya proved that random cross-validation causes fatal temporal leakage. We designed our architecture around these findings."*

---

## Slide 3: Data Ingestion & Fallback Architecture
### Automated Ingestion via `fetch_data.py`

* **Presenter**: Person 1
* **Source**: Real-time financial feeds ingested via `yfinance` API.
* **Coverage**: 2 years of daily OHLCV trading data (2,505 records across `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`).
* **Resilient Fallback Generator**:
  - If financial APIs or internet connectivity drop during evaluation demos, `fetch_data.py` activates an internal stochastic geometric Brownian motion generator.
  - Ensures zero pipeline crashes during live presentations and network interruptions.
* **Artifacts Generated**:
  - `data/raw_data.csv`: Normalized raw historical OHLCV data.
  - `data/last_updated.txt`: ISO UTC execution timestamp.

```
+------------------+       Live API Request        +-----------------------+
|   fetch_data.py  |  -------------------------->  |  yfinance (Yahoo API) |
+------------------+                               +-----------------------+
         |                                                     |
         | (On Timeout / Offline)                              v
         +-------------------------------------->  [Stochastic Fallback]
         |
         v
  data/raw_data.csv  (2,505 rows: Date, Ticker, Open, High, Low, Close, Volume)
```

> **🗣️ Speaker (Person 1)**:  
> *"Our first pipeline stage is `fetch_data.py`. It pulls two years of historical daily market data for five technology leaders. To ensure this project is resilient, we also implemented an automated stochastic baseline generator. If the university network or the external API ever experiences downtime, the system automatically falls back to simulated geometric drift so the pipeline never crashes during grading."*

---

## Slide 4: Data Cleaning & Statistical Outlier Bounding
### Robust Preprocessing via `clean_data.py`

* **Presenter**: Person 1
* **Missing Value Imputation**:
  - Grouped by ticker and chronologically sorted.
  - Forward-fill ($P_t = P_{t-1}$) across market holidays and localized trading halts.
  - Back-fill for edge initial cases.
  - Intraday consistency check: $P_{\text{High}} \ge \max(P_{\text{Open}}, P_{\text{Close}})$.
* **Interquartile Range (IQR) Winsorization**:
  - Rather than dropping rows (which breaks time-series continuity), derived ratio features are statistically bounded:
    $$\text{IQR} = Q_3 - Q_1$$
    $$\text{Bounds} = [Q_1 - 1.5 \times \text{IQR}, \; Q_3 + 1.5 \times \text{IQR}]$$
  - Applied to `Daily_Return`, `Volatility_10`, and `Spread` to neutralize anomalous reporting spikes.

> **🗣️ Speaker (Person 1)**:  
> *"Financial time-series data cannot simply have rows deleted when outliers occur because that breaks the continuous rolling window required for moving averages. Instead, in `clean_data.py`, we implemented IQR Winsorization. If a flash crash or split anomaly exceeds 1.5 times the interquartile range, the feature value is clipped to the statistical boundary, protecting the model from gradient explosion while preserving sequential continuity."*

---

## Slide 5: Feature Engineering & The Present-Day Holdout
### 12 Technical Indicators & Zero-Leakage Holdout

* **Presenter**: Person 1

```
+------------------------------------------------------------------------------------+
|                         12 ENGINEERED PREDICTIVE FEATURES                          |
+---------------------------+----------------------------+---------------------------+
|    Moving Averages (Trend)|     Momentum & Spread      |     Volatility & Volume   |
|    - MA_5, MA_10          |     - Daily_Return         |     - Volatility_10       |
|    - MA_20, MA_50         |     - Normalized Spread    |     - Normalized Volume   |
|                           |     - Open, High, Low      |     - Close               |
+---------------------------+----------------------------+---------------------------+
```

* **Mathematical Formulations**:
  - **Moving Averages**: $\text{MA}_k(t) = \frac{1}{k} \sum_{i=0}^{k-1} P_{\text{Close}}(t-i), \; k \in \{5, 10, 20, 50\}$
  - **Daily Return**: $R(t) = \frac{P_t - P_{t-1}}{P_{t-1}}$
  - **Rolling Volatility**: $\sigma_{10}(t) = \text{StdDev}(R(t), \dots, R(t-9))$
  - **Spread**: $\frac{P_{\text{High}} - P_{\text{Low}}}{P_{\text{Close}}}$
  - **Target Variable**: $y(t) = P_{\text{Close}}(t+1)$

#### 🌟 Critical Academic Differentiator: Present-Day Holdout
* At day $t_{\text{present}}$, tomorrow's price $P(t+1)$ **does not exist** ($\text{Target} = \text{NaN}$).
* `clean_data.py` separates this terminal day into `data/latest_unseen_data.csv`.
* `train_models.py` trains **strictly on historical days where the target is realized**.
* **Zero Lookahead Contamination**: Tomorrow's forecast on the web UI is evaluated on pure, unseen out-of-sample data!

> **🗣️ Speaker (Person 1)**:  
> *"Here is the most critical methodological point of our project. Notice that for the present trading day, tomorrow's price hasn't happened yet—its target is null. In many student projects, this row is either mishandled or causes errors. In our pipeline, we explicitly isolate today's row into `latest_unseen_data.csv` and drop it from the training set. When our web dashboard displays tomorrow's price forecast, the model is seeing today's data for the very first time. There is 100% zero lookahead leakage."*

---

## Slide 6: Multi-Model Candidate Architectures
### Competitive Benchmarking in `train_models.py`

* **Presenter**: Person 2 (ML & MLOps)
* **Chronological 80/20 Split**: 2,000 historical training samples, 500 out-of-sample testing samples.

#### 5 Candidate Architectures:
1. **Linear Regression (StandardScaler)**:
   $$\hat{y} = \mathbf{w}^T \mathbf{x} + b, \quad \mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
   Parametric baseline with Z-score standardization to normalize feature magnitudes.
2. **K-Nearest Neighbors (KNN Regressor)**:
   $$\hat{y}(\mathbf{x}) = \frac{\sum_{i \in \mathcal{N}_k} w_i y_i}{\sum w_i}, \quad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}$$
   Non-parametric memory model ($k=5$) identifying similar historical volatility regimes.
3. **Decision Tree Regressor**:
   Recursive orthogonal binary splitting with bounded tree depth ($\text{max\_depth}=6$) to avoid overfitting noise.
4. **Random Forest Regressor**:
   Bagging ensemble of $B=100$ decorrelated decision trees, minimizing variance across volatile market regimes:
   $$\text{Var}(\hat{y}_{\text{RF}}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$
5. **XGBoost Regressor**:
   Second-order gradient boosted trees minimizing regularized loss:
   $$\mathcal{L} \approx \sum [g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i)] + \gamma T + \frac{1}{2}\lambda \sum w_j^2$$

> **🗣️ Speaker (Person 2)**:  
> *"Moving to the machine learning core in `train_models.py`, we implement five distinct algorithms representing parametric, instance-based, tree, bagging, and gradient boosting paradigms. We strictly partition data chronologically—the earliest 2,000 records train the models, and the most recent 500 records evaluate them. We do not use randomized cross-validation because shuffling would destroy the temporal arrow of time."*

---

## Slide 7: Empirical Results & Performance Evaluation
### Official Out-of-Sample Leaderboard Metrics

* **Presenter**: Person 2
* **Data Source**: Extracted directly from production `models/metrics.json`.

| Algorithm | Model Paradigm | $R^2$ Score | RMSE ($) | MAE ($) | Latency (ms) | Train Time (s) | Promotion Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | Parametric (Z-Scaled) | **0.9857** | **$8.5680** | **$5.8642** | **1.06 ms** | **0.008 s** | **PROMOTED CHAMPION** |
| **Random Forest** | Bagging Ensemble (100 Trees) | 0.9816 | $9.7289 | $6.9969 | 15.15 ms | 0.155 s | Candidate |
| **XGBoost** | Gradient Boosted Trees | 0.9809 | $9.9086 | $6.9890 | 3.74 ms | 1.846 s | Candidate |
| **Decision Tree** | Single Orthogonal Tree | 0.9769 | $10.9069 | $7.8538 | 0.75 ms | 0.009 s | Candidate |
| **KNN Regressor** | Instance-Based Memory ($k=5$) | 0.9466 | $16.5797 | $11.7400 | 7.11 ms | 0.004 s | Candidate |

```
R² Goodness-of-Fit Comparison on 500 Test Samples:
Linear Reg   : [==================================================] 0.9857  (Best RMSE: $8.57)
Random Forest: [================================================]   0.9816  (RMSE: $9.73)
XGBoost      : [===============================================]    0.9809  (RMSE: $9.91)
Decision Tree: [=============================================]      0.9769  (RMSE: $10.91)
KNN Regressor: [=======================================]            0.9466  (RMSE: $16.58)
```

> **🗣️ Speaker (Person 2)**:  
> *"Here are our official empirical results evaluated across 500 test instances. Standardized Linear Regression emerged as the champion model with an $R^2$ of 0.9857 and the lowest RMSE of $8.5680. Random Forest and XGBoost followed very closely above 0.98. KNN struggled at 0.9466 with an RMSE of $16.58. Next, let us explain the theoretical justification for why Linear Regression outperformed the complex ensembles."*

---

## Slide 8: Theoretical Justification of Results
### Why Did Linear Regression Outperform Complex Ensembles?

* **Presenter**: Person 2

#### 1. Autoregressive Collinearity in Short-Horizon Forecasting ($t+1$)
* Next-day closing prices ($P_{t+1}$) are strongly collinear with recent moving averages ($r = 0.996$ with $\text{MA}_5$).
* Linear Regression solves for a smooth, optimal hyperplane without introducing variance:
  $$P_{t+1} \approx w_1 \text{MA}_5 + w_2 \text{Close}_t + \dots + b$$
* Because Z-score standardization was applied, OLS achieved high generalization stability without overfitting.

#### 2. The Extrapolation Problem of Decision Trees
* Decision trees predict piecewise constant step functions:
  $$\hat{y}(\mathbf{x}) = \sum_{m=1}^M c_m \mathbb{I}(\mathbf{x} \in R_m)$$
* If a stock enters an all-time high beyond the historical training interval, tree models **cannot extrapolate linearly**; they clip their projection to the highest leaf average $c_m$.
* Linear Regression extrapolates trend directions continuously along its fitted slope.

#### 3. Why Did KNN Degrade ($R^2 = 0.9466$)?
* In non-stationary markets with ongoing equity growth, points from two years ago occupy distant coordinates in 12D space. Euclidean distance becomes distorted by secular market shifts.

> **🗣️ Speaker (Person 2)**:  
> *"A frequent viva question is: 'Why did simple Linear Regression beat XGBoost and Random Forest?' The answer lies in the mathematics of time-series: First, short-term price movements are strongly linear relative to moving averages. Second, decision trees cannot extrapolate beyond the maximum value seen in training—they predict flat step functions at all-time highs. Linear Regression can extrapolate trend vectors cleanly. This proves why rigorous empirical benchmarking is superior to simply assuming that complex deep models are always best."*

---

## Slide 9: Automated Model Promotion & Audit Logging
### Governance Engine in `deploy_best_model.py`

* **Presenter**: Person 2
* **Objective Gatekeeper**: Automated selection replaces human intervention.
  - Evaluates `models/metrics.json` against `DEPLOYMENT_CRITERION` (default: $\max R^2$).
  - Identifies champion model ($M^*$).
  - Atomically copies binary: `shutil.copyfile(M*.pkl, "models/deployed_model.pkl")`.
* **Immutable Audit Trail (`models/deployment_log.json`)**:
  - Assigns unique UUID `run_id`.
  - Records deployment timestamp, winning model, metrics, and summary scores of all competing candidates.

```json
{
  "run_id": "c1f72a90",
  "timestamp": "2026-10-11 16:17:14 UTC",
  "deployed_model": "Linear_Regression",
  "selection_criterion": "r2",
  "metrics_at_deployment": {
    "rmse": 8.568,
    "mae": 5.8642,
    "r2": 0.9857,
    "inference_time_ms": 1.06
  }
}
```

> **🗣️ Speaker (Person 2)**:  
> *"In enterprise MLOps, a developer does not manually copy models into production. In `deploy_best_model.py`, we created an automated gatekeeper. It inspects the evaluation metrics, verifies which candidate scored the highest $R^2$, copies that file to `deployed_model.pkl`, and appends an immutable JSON audit log. The web serving layer simply loads `deployed_model.pkl` without ever needing hardcoded algorithm references."*

---

## Slide 10: Headless CI/CD Automation
### Scheduled Retraining via GitHub Actions (`pipeline.yml`)

* **Presenter**: Person 2 & Person 3
* **Market Hours Cron Schedule**:
  ```yaml
  schedule:
    # Runs hourly during US stock market trading hours (13:00 to 21:00 UTC, Mon-Fri)
    - cron: '0 13-21 * * 1-5'
  ```
  - Triggers at minute 0 of every hour from 9:00 AM to 5:00 PM Eastern Time.
  - 9 automated retrains per trading day; paused on weekends.
* **On-Demand Manual Trigger (`workflow_dispatch`)**:
  - Allows instant manual pipeline dispatch for evaluation demos via GitHub UI or the Flask Admin dashboard.
* **Autonomous Git Commit Bot**:
  - Signs commits as `github-actions[bot]`.
  - Commits updated `data/`, `models/`, `static/eda_corr.png`, and `model_run_log.txt` back to the repository.
  - **Zero Cost**: Eliminates expensive cloud database subscriptions.

> **🗣️ Speaker (Person 2)**:  
> *"To ensure the system never requires a local laptop to stay powered on, we orchestrated the entire lifecycle inside GitHub Actions. Using a cron schedule of `0 13-21 * * 1-5`, GitHub cloud runners automatically spin up hourly during US stock market hours, execute the pipeline, and commit the newly trained weights and execution logs straight back to the repository."*

---

## Slide 11: Production Cloud Architecture & Vercel Optimization
### Solving the 500 MB Serverless Limit via Two-Tier Decoupling

* **Presenter**: Person 3 (Web & Deployment)

```
+------------------------------------------------------------------------------------+
|                         TWO-TIER DEPENDENCY ARCHITECTURE                           |
+-----------------------------------------+------------------------------------------+
|  Training Tier (GitHub Actions Runner)  |    Serving Tier (Vercel Serverless)      |
|  File: requirements-pipeline.txt        |    File: requirements.txt                |
|  - xgboost, yfinance, matplotlib        |    - Flask, Werkzeug, scikit-learn       |
|  - seaborn, python-dotenv               |    - pandas, numpy, joblib, requests     |
|  - Size: ~800 MB (Unrestricted VM)      |    - Size: ~150 MB (< 500 MB limit)      |
+-----------------------------------------+------------------------------------------+
```

* **The Problem**: Initial Vercel deployment failed with:
  `Error: Total bundle size (798.86 MB) exceeds the maximum function size (500 MB)`
  caused by heavy C++ binaries in `xgboost` and graphical libraries (`matplotlib`, `seaborn`).
* **The Engineering Solution**:
  - Vercel only serves model inferences; it never trains models at runtime.
  - Decoupled into lightweight serving dependencies (`requirements.txt` $\approx 150\text{ MB}$) for Vercel, and full pipeline dependencies (`requirements-pipeline.txt`) for GitHub Actions.
  - **Read-Only Filesystem Fix**: In serverless environments, `auth.py` automatically relocates SQLite storage to `/tmp/users.db` to prevent read-only filesystem exceptions.

> **🗣️ Speaker (Person 3)**:  
> *"When deploying our web platform to Vercel, we encountered a real-world cloud engineering challenge: Vercel rejected the deployment because `xgboost`, `matplotlib`, and `yfinance` pushed the bundle size to nearly 800 MB against Vercel's 500 MB ceiling. We solved this by decoupling the architecture into two tiers: GitHub Actions handles the heavy training dependencies, while Vercel runs a lean 150 MB serving package. When GitHub Actions commits new models, Vercel detects the push and redeploys automatically."*

---

## Slide 12: Application Architecture & Dual-Session Security
### Decoupled Security Model in `auth.py`

* **Presenter**: Person 3

```
+------------------------------------------------------------------------------------+
|                         STRICT DUAL-SESSION ISOLATION                              |
+-----------------------------------------+------------------------------------------+
|          Trader Access Portal           |        Administrator Ops Console         |
|  - Routes: /login, /dashboard           |  - Routes: /admin, /admin/dashboard      |
|  - Session Token: session['user_id']    |  - Session Token: session['admin_id']    |
|  - Route Guard: @login_required         |  - Route Guard: @admin_required          |
|  - Auth: SQLite users.db (Scrypt Hash)  |  - Auth: Environment Variable Secret     |
+-----------------------------------------+------------------------------------------+
```

* **Zero Plaintext Admin Passwords**: Admin credentials are read dynamically from `ADMIN_PASSWORD` environment variables.
* **Cryptographic Session Isolation**: A regular user session token can never access administrative monitoring endpoints, and vice versa.
* **REST API Endpoints**:
  - `/api/predict/<ticker>`: Asynchronously streams predictions and actual series to Chart.js.
  - `/api/admin/trigger-pipeline`: Dispatches on-demand GitHub Actions workflows.
  - `/api/admin/pipeline-status`: Polling endpoint for real-time progress bars.

> **🗣️ Speaker (Person 3)**:  
> *"For system security, we enforced strict defense-in-depth in `auth.py`. Instead of using a simple role column in a database, user and admin sessions use entirely different session keys and independent route decorators. The admin password is never stored in the database or source code; it is injected via an environment variable. Even if the user database were breached, administrative pipeline access remains completely secure."*

---

## Slide 13: Live User Interfaces Demonstration
### Trader Dashboard vs. Operations Console

* **Presenter**: Person 3

#### 1. Trader Dashboard (`/dashboard`)
* **Minimalist Consumer Aesthetic**: Clean typography (Inter font), card elevation, high-contrast readability.
* **Centerpiece Chart (Chart.js)**: Displays 60 trading days of actual closing prices alongside out-of-sample model predictions.
* **Tomorrow's Prediction Card**: Displays projected price, dollar difference ($\Delta$), and directional movement badge (`+X.XX% Gain` or `-X.XX% Loss`).
* **Asynchronous Ticker Switcher**: Switching between `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA` updates charts dynamically via AJAX without full page reloads.

#### 2. Admin Operations Console (`/admin/dashboard`)
* **Dark Slate Telemetry**: Health badges displaying cron status (`0 13-21 * * 1-5`), last retrain timestamp, and active champion.
* **Live Model Leaderboard**: All 5 models compared side by side with the deployed champion highlighted in green.
* **Deployment Frequency Bar Chart**: Distribution of promoted models across pipeline history.
* **Real-time Pipeline Runner**: Live "Run Pipeline Now" trigger with an active progress bar and terminal log stream.
* **Model Execution Log**: Displays the live timestamp audit trail from `data/model_run_log.txt`.

> **🗣️ Speaker (Person 3)**:  
> *"Our user interface delivers two distinct experiences: The Trader Dashboard provides consumers with clean Chart.js actual vs. predicted price comparisons and next-day price forecasts. The Admin Console provides MLOps engineers with complete telemetry, the 5-model leaderboard, deployment histories, and an interactive trigger button that lets evaluators dispatch the pipeline live."*

---

## Slide 14: Comprehensive Project Defense (Viva Q&A)
### Anticipated Evaluator Questions & Technical Answers

* **Presenter**: All Presenters

#### Q1: "Why did you use 5 models instead of just training one?"
> **Answer**: *"Market volatility alternates unpredictably between steady trending states and volatile shocks. Linear Regression excels in linear autoregressive trends, while Random Forest and XGBoost handle non-linear volatility regimes. Continuous benchmarking guarantees that whichever model performs best on out-of-sample data is automatically deployed."*

#### Q2: "What is your defense against lookahead bias / data leakage?"
> **Answer**: *"Two strict guarantees: First, we use a chronological 80/20 temporal split with zero random shuffling. Second, the present trading day is completely isolated into `latest_unseen_data.csv` and dropped from training because its target is unobserved. Tomorrow's prediction is evaluated on pure unseen data."*

#### Q3: "How is this project different from existing academic projects?"
> **Answer**: *"Most student projects are static Jupyter notebooks trained once on a historical CSV file. Our project implements a continuous, production-grade MLOps system that automatically ingests data, retrains hourly on GitHub cloud runners, versions model weights in git, solves serverless bundle constraints, and serves real-time predictions on Vercel."*

#### Q4: "What does the model run log prove?"
> **Answer**: *"`data/model_run_log.txt` logs exact UTC timestamps for every execution (`model ran at YYYY-MM-DD : HH:MM:SS UTC`). It proves that the pipeline is actively running on schedule in the cloud and version-controlling artifacts back to the repository."*

---

## Slide 15: Conclusion & Live Demonstration
### Project Summary & Live System Walkthrough

* **Presenter**: All Presenters
* **Summary of Technical Achievements**:
  1. Automated 5-equity market ingestion with stochastic fallback resiliency.
  2. Statistical IQR outlier Winsorization and 12-feature technical engineering.
  3. Methodological zero-leakage guarantee via present-day holdout isolation.
  4. 5-model benchmark proving Linear Regression champion status ($R^2 = 0.9857$, $\text{RMSE} = \$8.57$).
  5. Continuous MLOps execution via GitHub Actions hourly cron during market hours.
  6. Two-tier dependency resolution overcoming Vercel's 500 MB serverless ceiling.
  7. High-contrast, minimal dual-session web application on Flask.

```
                    +------------------------------------------+
                    |           LIVE SYSTEM DEMO               |
                    |   1. Public Landing Page (/)             |
                    |   2. Trader Dashboard (/dashboard)       |
                    |   3. AJAX Ticker Switching (AAPL/TSLA)   |
                    |   4. Admin Console (/admin/dashboard)    |
                    |   5. Live Pipeline Trigger Execution     |
                    +------------------------------------------+
```

> **🗣️ Speaker (All)**:  
> *"In conclusion, Stock Predictor demonstrates that modern machine learning in finance is fundamentally an engineering and MLOps discipline. We now welcome questions from the evaluation committee and invite you to observe our live system demonstration. Thank you."*
