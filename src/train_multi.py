# src/train_multi.py
import argparse, json, uuid
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
from src.data import load_data, split_stratified
from src.models import build_pipeline
from src.metrics import base_metrics, best_threshold_max_f1
from src.config import (
    ARTIFACTS_DIR, LOGS_DIR, RANDOM_STATE, DATA_SCHEMA_VERSION
)
from src.config import MODEL_VERSION  # reutilizamos el mismo esquema de versión
from src.config import __version__ as code_version

def _row_id_frame(X):
    # Si existe CustomerID úsalo, si no, usa el índice como row_id
    if "CustomerID" in X.columns:
        return pd.Series(X["CustomerID"].astype(str), name="row_id")
    return pd.Series(X.index.astype(str), index=X.index, name="row_id")

def _save_experiment_row(path_csv, row_dict):
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row_dict])
    if path_csv.exists():
        df_row.to_csv(path_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path_csv, index=False)

def train_and_log(model_name, df, target):
    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_stratified(
        df, target=target, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE
    )

    # Pipeline
    pipe = build_pipeline(model_name, df, target)
    print(f"\nEntrenando {model_name} ...")
    pipe.fit(X_train, y_train)

    # Probabilidades
    val_prob = pipe.predict_proba(X_val)[:, 1]
    test_prob = pipe.predict_proba(X_test)[:, 1]

    # Umbral óptimo (F1 en validation)
    thr, best_f1_val = best_threshold_max_f1(y_val, val_prob)

    # Métricas
    val_metrics = base_metrics(y_val, val_prob, threshold=thr)
    test_metrics = base_metrics(y_test, test_prob, threshold=thr)

    # Guardar modelo + metadata + threshold
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    model_token = f"{model_name}_{MODEL_VERSION}_{stamp}"
    model_path = ARTIFACTS_DIR / f"{model_token}.joblib"
    joblib.dump(pipe, model_path)

    meta = {
        "model_name": model_name,
        "model_version": MODEL_VERSION,
        "trained_at_utc": stamp,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "code_version": code_version,
        "random_state": RANDOM_STATE,
        "target": target,
        "threshold_val_f1": thr,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }
    with open(ARTIFACTS_DIR / f"{model_token}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Guardado: {model_path.name}")
    print(f"   Umbral(F1 val): {thr:.3f} | F1(val)={best_f1_val:.3f}")
    print(f"   ROC_AUC(test)={test_metrics['roc_auc']:.3f} | PR_AUC(test)={test_metrics['pr_auc']:.3f} | F1(test)={test_metrics['f1']:.3f}")

    # Logging estructurado de predicciones (TEST)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_name = f"preds_{model_token}_TEST.csv"
    row_id = _row_id_frame(X_test)
    y_pred = (test_prob >= thr).astype(int)
    log_df = pd.DataFrame({
        "row_id": row_id,
        "y_true": y_test.values,
        "y_prob": test_prob,
        "y_pred": y_pred,
        "model_name": model_name,
        "model_version": MODEL_VERSION,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "logged_at_utc": stamp
    })
    log_df.to_csv(LOGS_DIR / log_name, index=False)
    print(f"🗒️  Log de predicciones -> logs/{log_name} ({len(log_df)} filas)")

    # Registrar fila del experimento
    exp_row = {
        "experiment_id": str(uuid.uuid4()),
        "trained_at_utc": stamp,
        "model_token": model_token,
        "model_name": model_name,
        "model_version": MODEL_VERSION,
        "threshold": thr,
        "roc_auc_val": val_metrics["roc_auc"],
        "pr_auc_val": val_metrics["pr_auc"],
        "f1_val": val_metrics["f1"],
        "roc_auc_test": test_metrics["roc_auc"],
        "pr_auc_test": test_metrics["pr_auc"],
        "f1_test": test_metrics["f1"],
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0])
    }
    _save_experiment_row(ARTIFACTS_DIR / "experiments.csv", exp_row)

    return meta, model_path

def main(args):
    df = load_data(args.data)
    target = args.target

    results = []
    for name in ["logreg", "xgb", "mlp"]:
        meta, path = train_and_log(name, df, target)
        results.append((name, meta, path))

    print("\nResumen rápido:")
    for name, meta, path in results:
        tm = meta["test_metrics"]
        print(f" - {name}: ROC_AUC={tm['roc_auc']:.3f} | PR_AUC={tm['pr_auc']:.3f} | F1@thr(test)={tm['f1']:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/retencion.csv")
    parser.add_argument("--target", type=str, default="usuarioPerdido")
    args = parser.parse_args()
    main(args)
