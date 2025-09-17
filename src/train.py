# src/train.py
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
from src.config import DROP_COLS

from src.config import ARTIFACTS_DIR, RANDOM_STATE, MODEL_NAME, MODEL_VERSION, DATA_SCHEMA_VERSION
from src.data import load_data, split_stratified
from src.features import build_preprocessor

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1@0.5": f1_score(y_true, y_pred)
    }
    return metrics

def main(args):
    df = load_data(args.data)
    
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    target = args.target

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_stratified(
        df, target=target, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE
    )

    # Preprocesamiento
    pre = build_preprocessor(df, target=target)

    # Modelo baseline (Logistic Regression)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    print("Entrenando modelo baseline (LogReg)...")
    pipe.fit(X_train, y_train)

    # Eval
    val_prob = pipe.predict_proba(X_val)[:, 1]
    test_prob = pipe.predict_proba(X_test)[:, 1]
    val_metrics = evaluate(y_val, val_prob, threshold=0.5)
    test_metrics = evaluate(y_test, test_prob, threshold=0.5)

    # Mostrar resultados
    print("\nMétricas (VALIDATION):")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nMétricas (TEST):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Guardar artefacto + metadata
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / f"{MODEL_NAME}_{MODEL_VERSION}.joblib"
    joblib.dump(pipe, model_path)

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "random_state": RANDOM_STATE,
        "target": target,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Modelo guardado en: {model_path}")
    print("✅ Metadata en: artifacts/metadata.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/retencion.csv")
    parser.add_argument("--target", type=str, default="usuarioPerdido")
    args = parser.parse_args()
    main(args)
