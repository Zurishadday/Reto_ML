# src/select_champion.py
import json, shutil
from pathlib import Path
import pandas as pd
from src.config import ARTIFACTS_DIR

def main():
    exp_path = ARTIFACTS_DIR / "experiments.csv"
    assert exp_path.exists(), "No existe artifacts/experiments.csv"
    df = pd.read_csv(exp_path)

    # Regla: mayor PR_AUC(test), desempate por F1(test)
    df_sorted = df.sort_values(["pr_auc_test", "f1_test"], ascending=[False, False])
    row = df_sorted.iloc[0]
    token = row["model_token"]

    joblib_src = ARTIFACTS_DIR / f"{token}.joblib"
    meta_src   = ARTIFACTS_DIR / f"{token}.json"
    assert joblib_src.exists() and meta_src.exists(), "Faltan artefactos del modelo ganador"

    # Copias "congeladas" del campeón
    shutil.copy2(joblib_src, ARTIFACTS_DIR / "champion_model.joblib")
    shutil.copy2(meta_src,   ARTIFACTS_DIR / "champion_meta.json")

    info = {
        "selected_from": str(exp_path),
        "selection_rule": "max pr_auc_test then max f1_test",
        "model_token": token,
        "metrics_test": {
            "roc_auc": row["roc_auc_test"],
            "pr_auc": row["pr_auc_test"],
            "f1": row["f1_test"],
        },
        "threshold": row["threshold"]
    }
    with open(ARTIFACTS_DIR / "champion_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("✅ Campeón seleccionado:")
    print(f"   token: {token}")
    print(f"   PR_AUC(test): {row['pr_auc_test']:.3f} | F1(test): {row['f1_test']:.3f}")
    print(f"   Umbral guardado: {row['threshold']:.3f}")
    print("   Archivos: artifacts/champion_model.joblib, champion_meta.json, champion_info.json")

if __name__ == "__main__":
    main()
