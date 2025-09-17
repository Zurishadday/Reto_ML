# src/build_reference.py
import json, re
from pathlib import Path
import pandas as pd, numpy as np

REF_PATH = Path("monitoring/reference_scores.json")
LOGS_DIR = Path("logs")

def main():
    # Usa el token del campeón para buscar un log de TEST o el más reciente
    with open("artifacts/champion_info.json", "r", encoding="utf-8") as f:
        token = json.load(f)["model_token"]

    # Preferimos un log TEST si existe; si no, cualquier log del token
    cand = sorted([p for p in LOGS_DIR.glob(f"*{token}*.csv")], key=lambda x: x.stat().st_mtime, reverse=True)
    assert cand, "No hay logs para construir referencia. Corre predict o usa los TEST."
    df = pd.read_csv(cand[0])
    scores = df["y_prob"].to_numpy()

    hist, edges = np.histogram(scores, bins=20, range=(0.0, 1.0), density=True)
    with open(REF_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_token": token,
            "source_log": str(cand[0]),
            "bins": edges.tolist(),
            "pdf": hist.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std())
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ Referencia creada en {REF_PATH} usando {cand[0].name}")

if __name__ == "__main__":
    main()
