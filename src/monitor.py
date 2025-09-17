# src/monitor.py
import argparse, json
from pathlib import Path
from datetime import datetime
import pandas as pd, numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

MON_DIR = Path("monitoring")
LOGS_DIR = Path("logs")

def _load_reference():
    ref = json.load(open(MON_DIR / "reference_scores.json", "r", encoding="utf-8"))
    return np.array(ref["bins"]), np.array(ref["pdf"]), ref

def _ks_against_reference(scores, ref_scores):
    # KS against empirical reference scores (approx via resampling from PDF bins)
    bins, pdf, _ = ref_scores
    # Monte Carlo sample from reference pdf
    bin_probs = pdf / pdf.sum()
    # pick bin then uniform inside bin
    rnd_bins = np.random.choice(len(bin_probs), size=len(scores), p=bin_probs)
    left = bins[rnd_bins]; right = bins[rnd_bins+1]
    ref_samples = left + (right - left) * np.random.rand(len(scores))
    stat, p = ks_2samp(scores, ref_samples)
    return float(p)

def main(args):
    MON_DIR.mkdir(parents=True, exist_ok=True)
    bins, pdf, ref_meta = _load_reference()

    # Agrega resultados diarios
    summary_path = MON_DIR / "daily_metrics.csv"
    alerts_path  = MON_DIR / "alerts.json"
    alerts = []

    # Lee todos los logs del campeón
    token = ref_meta["model_token"]
    files = sorted([p for p in LOGS_DIR.glob(f"preds_{token}_*.csv")])
    if not files:
        print("No hay logs para monitorear. Corre src.predict primero.")
        return

    rows = []
    for fp in files:
        df = pd.read_csv(fp)
        date = str(df["logged_date"].iloc[0]) if "logged_date" in df.columns else "unknown"
        scores = df["y_prob"].to_numpy()
        p_ks = _ks_against_reference(scores, (bins, pdf, ref_meta))
        pos_rate = float((df["y_pred"]==1).mean())

        # Si hay verdad terreno, calcula métricas
        roc = pr = f1 = np.nan
        if "y_true" in df.columns and not df["y_true"].isna().all():
            y_true = df["y_true"].to_numpy().astype(int)
            roc = float(roc_auc_score(y_true, scores))
            pr  = float(average_precision_score(y_true, scores))
            thr = df["y_pred"].median()  # dummy para f1 con y_pred ya binaria
            f1  = float(f1_score(y_true, df["y_pred"]))
        rows.append({"date": date, "n": len(df), "pos_rate_pred": pos_rate, "ks_p_score": p_ks,
                     "roc_auc": roc, "pr_auc": pr, "f1": f1, "log_file": fp.name})

        # Reglas de alerta simples
        if p_ks < args.ks_alpha:
            alerts.append({"date": date, "type": "DRIFT_SCORES", "p_value": p_ks, "file": fp.name})
        if pos_rate < args.pos_rate_min or pos_rate > args.pos_rate_max:
            alerts.append({"date": date, "type": "POS_RATE_OUT_OF_RANGE", "pos_rate": pos_rate, "file": fp.name})

    out = pd.DataFrame(rows).sort_values("date")
    if summary_path.exists():
        # merge incremental por fecha+archivo para evitar duplicados
        old = pd.read_csv(summary_path)
        out = pd.concat([old, out]).drop_duplicates(subset=["date","log_file"], keep="last")
    out.to_csv(summary_path, index=False)

    with open(alerts_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at_utc": datetime.utcnow().isoformat(timespec="seconds")+"Z",
                   "rules": {"ks_alpha": args.ks_alpha, "pos_rate_min": args.pos_rate_min, "pos_rate_max": args.pos_rate_max},
                   "alerts": alerts}, f, ensure_ascii=False, indent=2)

    print(f"✅ Métricas diarias -> {summary_path}")
    print(f"🔔 Alertas -> {alerts_path}  (total={len(alerts)})")
    if alerts:
        for a in alerts[-5:]:
            print("  -", a)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks_alpha", type=float, default=0.01)      # p<0.01 => drift
    ap.add_argument("--pos_rate_min", type=float, default=0.05)  # rango sano (ajústalo con tus datos)
    ap.add_argument("--pos_rate_max", type=float, default=0.60)
    main(ap.parse_args())
