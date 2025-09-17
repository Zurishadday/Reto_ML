# src/metrics.py
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve
)

def base_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred))
    }

def best_threshold_max_f1(y_true, y_prob):
    # Usamos la curva P-R para evaluar F1 en muchos puntos
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Evitar división por cero
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.argmax(f1s))
    # precision_recall_curve regresa N umbrales para N-1 puntos
    thr = float(thresholds[max(idx-1, 0)]) if len(thresholds) else 0.5
    return thr, float(f1s[idx])
