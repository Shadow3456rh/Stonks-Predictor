"""
deploy_best_model.py
====================
Automated Model Deployment & Promotion Engine
Reads models/metrics.json, selects the superior candidate based on a configurable
selection metric (R² or RMSE), promotes it to models/deployed_model.pkl, and
appends the deployment record to models/deployment_log.json for audit and visualization.
"""

import os
import json
import shutil
import datetime
import uuid

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
DEPLOYED_MODEL_PATH = os.path.join(MODELS_DIR, "deployed_model.pkl")
DEPLOYMENT_LOG_PATH = os.path.join(MODELS_DIR, "deployment_log.json")

# Configurable selection criterion: 'r2' (maximize) or 'rmse' (minimize)
CRITERION = os.environ.get("DEPLOYMENT_CRITERION", "r2").lower()


def select_and_deploy():
    """Reads metrics, determines best model, copies artifact, and updates history."""
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"Metrics file missing at {METRICS_PATH}. Run train_models.py first.")

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    models_dict = metrics_data.get("models", {})
    if not models_dict:
        raise ValueError("No models found in metrics.json")

    print(f"[*] Evaluating candidate models using criterion: '{CRITERION.upper()}'")

    best_model_name = None
    best_score = None

    for name, metrics in models_dict.items():
        score = metrics.get(CRITERION)
        if score is None:
            continue

        if CRITERION == "r2":
            # Higher is better
            if best_score is None or score > best_score:
                best_score = score
                best_model_name = name
        elif CRITERION in ("rmse", "mae"):
            # Lower is better
            if best_score is None or score < best_score:
                best_score = score
                best_model_name = name
        else:
            # Default to r2
            if best_score is None or metrics.get("r2", -999) > best_score:
                best_score = metrics.get("r2", -999)
                best_model_name = name

    if not best_model_name:
        # Fallback to first model
        best_model_name = list(models_dict.keys())[0]

    source_model_path = os.path.join(MODELS_DIR, f"{best_model_name}.pkl")
    if not os.path.exists(source_model_path):
        raise FileNotFoundError(f"Target model file not found: {source_model_path}")

    # Copy to deployed_model.pkl
    shutil.copyfile(source_model_path, DEPLOYED_MODEL_PATH)
    print(f"[+] Promoted and deployed model '{best_model_name}' to {DEPLOYED_MODEL_PATH}")

    # Load or initialize deployment log
    deployment_history = []
    if os.path.exists(DEPLOYMENT_LOG_PATH):
        try:
            with open(DEPLOYMENT_LOG_PATH, "r", encoding="utf-8") as f:
                deployment_history = json.load(f)
                if not isinstance(deployment_history, list):
                    deployment_history = []
        except Exception:
            deployment_history = []

    timestamp_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    winning_metrics = models_dict[best_model_name]

    log_entry = {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": timestamp_iso,
        "deployed_model": best_model_name,
        "selection_criterion": CRITERION,
        "metrics_at_deployment": winning_metrics,
        "all_candidates_summary": {m: {"r2": d.get("r2"), "rmse": d.get("rmse")} for m, d in models_dict.items()}
    }

    deployment_history.append(log_entry)

    with open(DEPLOYMENT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(deployment_history, f, indent=2)

    print(f"[+] Appended deployment event to {DEPLOYMENT_LOG_PATH}")
    print(f"    Selected: {best_model_name} | {CRITERION.upper()}: {best_score}")

    return log_entry


if __name__ == "__main__":
    select_and_deploy()
