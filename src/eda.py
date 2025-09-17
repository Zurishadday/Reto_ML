# src/eda.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import mutual_info_score
from src.config import MONITOR_DIR, RANDOM_STATE
from src.data import load_data, split_stratified
from src.features import infer_column_types

def save_fig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

def factorize_series(s: pd.Series):
    # Si es categórica, primero agrega explícitamente la categoría para missing
    if is_categorical_dtype(s):
        s = s.cat.add_categories(["__MISSING__"]).fillna("__MISSING__")
    else:
        s = s.fillna("__MISSING__")

    # factorizar en enteros (mutual_info_score espera labels)
    codes, _ = pd.factorize(s.astype(str))
    return codes

def eda_summary(df: pd.DataFrame, target: str):
    rows, cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    missing = df.isna().sum().sort_values(ascending=False).to_dict()
    dup_rows = int(df.duplicated().sum())
    uniq_counts = df.nunique(dropna=True).to_dict()
    tgt_balance = df[target].value_counts(normalize=True).to_dict() if target in df.columns else {}
    return {
        "shape": {"rows": rows, "cols": cols},
        "dtypes": dtypes,
        "missing_top": dict(list(missing.items())[:20]),
        "duplicate_rows": dup_rows,
        "unique_counts_top": dict(list(sorted(uniq_counts.items(), key=lambda x: -x[1])[:20])),
        "target_balance": tgt_balance
    }

def plot_distributions(df, num_cols, cat_cols, target):
    out_dir = MONITOR_DIR / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Histos numéricos
    for c in num_cols:
        plt.figure()
        df[c].hist(bins=30)
        plt.title(f"Histograma: {c}")
        save_fig(out_dir / f"num_{c}.png")

    # Barras top categorías (hasta 20)
    for c in cat_cols:
        plt.figure()
        df[c].astype(str).fillna("__MISSING__").value_counts()[:20].plot(kind="bar")
        plt.title(f"Frecuencias top: {c}")
        save_fig(out_dir / f"cat_{c}.png")

    # Correlación numérica con target (si binario 0/1)
    if target in df.columns and set(df[target].dropna().unique()).issubset({0,1}):
        num_df = df[num_cols].copy()
        corr = num_df.join(df[target]).corr(numeric_only=True)[target].drop(index=target).sort_values(ascending=False)
        plt.figure(figsize=(6, max(2, 0.3*len(corr))))
        corr.plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title("Correlación (Pearson) numérica vs target")
        save_fig(out_dir / "corr_num_vs_target.png")

def leakage_check(df, num_cols, cat_cols, target):
    issues = []
    # MI en numéricas y categóricas por separado (hecho columna a columna)
    if target in df.columns:
        y = df[target].astype(int)
        # num: MI aproximada discretizando en quantiles para evitar escala
        for c in num_cols:
            x = pd.qcut(df[c], q=np.clip(df[c].nunique(), 2, 20), duplicates="drop") if df[c].notna().sum()>0 else df[c]
            mi = mutual_info_score(y, factorize_series(x))
            if mi > 0.2:  # umbral simple, ajustable
                issues.append({"feature": c, "type": "num", "suspected_leak_MI": round(mi,4)})
        # cat
        for c in cat_cols:
            mi = mutual_info_score(y, factorize_series(df[c]))
            if mi > 0.2:
                issues.append({"feature": c, "type": "cat", "suspected_leak_MI": round(mi,4)})
    return issues

def drift_report(X_train, X_test, num_cols, cat_cols):
    rows = []
    # Numéricas: KS
    for c in num_cols:
        a = X_train[c].dropna()
        b = X_test[c].dropna()
        if len(a) > 0 and len(b) > 0:
            stat, p = ks_2samp(a, b)
            rows.append({"feature": c, "type": "num", "test": "KS", "p_value": float(p), "drift_flag": bool(p < 0.01)})
    # Categóricas: Chi2 (tabla 2xK)
    for c in cat_cols:
        tr = X_train[c].astype(str).fillna("__MISSING__")
        te = X_test[c].astype(str).fillna("__MISSING__")
        cats = sorted(set(tr.unique()).union(set(te.unique())))
        cont = np.vstack([
            tr.value_counts().reindex(cats, fill_value=0).to_numpy(),
            te.value_counts().reindex(cats, fill_value=0).to_numpy()
        ])
        if cont.shape[1] >= 2:
            chi2, p, dof, _ = chi2_contingency(cont, correction=False)
            rows.append({"feature": c, "type": "cat", "test": "Chi2", "p_value": float(p), "drift_flag": bool(p < 0.01), "n_categories": int(cont.shape[1])})
    rep = pd.DataFrame(rows).sort_values(["drift_flag","p_value"], ascending=[False, True])
    return rep

def main(args):
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)
    target = args.target

    assert target in df.columns, f"No encuentro la columna objetivo '{target}'"
    num_cols, cat_cols = infer_column_types(df, target)

    # 1) Resumen
    summary = eda_summary(df, target)
    with open(MONITOR_DIR / "eda_overview.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("✅ Resumen EDA -> monitoring/eda_overview.json")
    print(f"Balance target: {summary['target_balance']}")

    # 2) Gráficas
    plot_distributions(df, num_cols, cat_cols, target)
    print("✅ Gráficas -> monitoring/eda/*.png")

    # 3) Chequeo de fuga (MI alta)
    leaks = leakage_check(df, num_cols, cat_cols, target)
    pd.DataFrame(leaks).to_csv(MONITOR_DIR / "leakage_suspects.csv", index=False)
    print(f"⚠️  Posibles fugas -> monitoring/leakage_suspects.csv (filas: {len(leaks)})")

    # 4) Drift entre train y test (usamos tu mismo split estratificado)
    from src.data import split_stratified
    X_train, X_val, X_test, y_train, y_val, y_test = split_stratified(
        df, target=target, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE
    )
    rep = drift_report(X_train, X_test, num_cols, cat_cols)
    rep.to_csv(MONITOR_DIR / "drift_report_train_vs_test.csv", index=False)
    print(f"✅ Drift report -> monitoring/drift_report_train_vs_test.csv")
    # Top 10 en consola
    if len(rep):
        print(rep.head(10).to_string(index=False))

    # 5) Guardar schema simple
    schema = {
        "target": target,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "random_state": RANDOM_STATE
    }
    with open(MONITOR_DIR / "data_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print("✅ Esquema -> monitoring/data_schema.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/retencion.csv")
    parser.add_argument("--target", type=str, default="usuarioPerdido")
    args = parser.parse_args()
    main(args)
