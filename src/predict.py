# src/predict.py
import argparse, json
from pathlib import Path
from datetime import datetime
import pandas as pd, numpy as np, joblib
from src.config import LOGS_DIR, DATA_SCHEMA_VERSION
from src.features import infer_column_types

def _row_id_frame(X):
    if "CustomerID" in X.columns:
        return pd.Series(X["CustomerID"].astype(str), name="row_id")
    return pd.Series(X.index.astype(str), index=X.index, name="row_id")

def main(args):
    from src.config import RANDOM_STATE
    model = joblib.load("artifacts/champion_model.joblib")
    with open("artifacts/champion_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    thr = float(info["threshold"]); token = info["model_token"]

    df = pd.read_csv(args.data)
    target = args.target if args.target in df.columns else None

    # Muestra opcional (simula lote diario)
    if args.sample is not None and 0 < args.sample < 1:
        df = df.sample(frac=args.sample, random_state=RANDOM_STATE).reset_index(drop=True)

    X = df.copy()
    if target:
        X = X.drop(columns=[target])

    prob = model.predict_proba(X)[:, 1]
    y_pred = (prob >= thr).astype(int)

    row_id = _row_id_frame(X)
    stamp = (args.date or datetime.utcnow().strftime("%Y-%m-%d"))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "row_id": row_id,
        "y_prob": prob,
        "y_pred": y_pred,
        "model_token": token,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "logged_date": stamp
    })
    if target:
        out["y_true"] = df[target].values

    fname = f"preds_{token}_{stamp}.csv"
    out.to_csv(LOGS_DIR / fname, index=False)
    print(f"🗒️  Log guardado -> logs/{fname}  (n={len(out)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/retencion.csv")
    ap.add_argument("--target", type=str, default="usuarioPerdido")
    ap.add_argument("--sample", type=float, default=0.2)  # 20% por defecto
    ap.add_argument("--date", type=str, default=None)     # YYYY-MM-DD (opcional)
    main(ap.parse_args())
