# src/eval_champion.py
import json
from pathlib import Path
import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             confusion_matrix, classification_report)
from src.config import ARTIFACTS_DIR, MONITOR_DIR, RANDOM_STATE
from src.data import load_data, split_stratified

def save_fig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

def precision_at_k(y_true, y_prob, ks=[0.05, 0.10, 0.20]):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False).reset_index(drop=True)
    out = {}; n = len(df)
    for k in ks:
        top = int(np.ceil(n*k)); sel = df.iloc[:top]
        out[f"p@{int(k*100)}"] = float(sel["y"].mean()) if top>0 else np.nan
    return out

def main():
    model = joblib.load(ARTIFACTS_DIR / "champion_model.joblib")
    with open(ARTIFACTS_DIR / "champion_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    thr, token = float(info["threshold"]), info["model_token"]

    df = load_data("data/retencion.csv")
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_stratified(df, target="usuarioPerdido", test_size=0.2, val_size=0.2, random_state=RANDOM_STATE)

    prob = model.predict_proba(X_te)[:,1]
    y_pred = (prob >= thr).astype(int)

    roc_auc = roc_auc_score(y_te, prob)
    pr_auc  = average_precision_score(y_te, prob)
    cm      = confusion_matrix(y_te, y_pred).tolist()
    clsrep  = classification_report(y_te, y_pred, output_dict=True)

    # Curvas
    prec, rec, _ = precision_recall_curve(y_te, prob)
    fpr, tpr, _  = roc_curve(y_te, prob)
    plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve - champion")
    save_fig(MONITOR_DIR / f"eval_{token}_pr_curve.png")
    plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve - champion")
    save_fig(MONITOR_DIR / f"eval_{token}_roc_curve.png")

    p_at_k = precision_at_k(y_te.values, prob, ks=[0.05, 0.10, 0.20])

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    with open(MONITOR_DIR / f"eval_{token}.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_token": token, "threshold": thr,
            "roc_auc_test": float(roc_auc), "pr_auc_test": float(pr_auc),
            "confusion_matrix": cm, "classification_report": clsrep,
            "precision_at_k": p_at_k
        }, f, ensure_ascii=False, indent=2)

    print("✅ Evaluación guardada en monitoring/:")
    print(f"   eval_{token}.json, eval_{token}_pr_curve.png, eval_{token}_roc_curve.png")
    print(f"   Precision@5/10/20%: {p_at_k}")

if __name__ == "__main__":
    main()
