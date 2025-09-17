# src/models.py
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from src.features import build_preprocessor
from src.config import RANDOM_STATE

def build_pipeline(model_name: str, df, target: str):
    pre = build_preprocessor(df, target=target)

    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=1000, solver="lbfgs", class_weight="balanced"
        )
    elif model_name == "xgb":
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist"   # rápido y estable
        )
    elif model_name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            random_state=RANDOM_STATE,
            max_iter=300,
            early_stopping=True
        )
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return Pipeline([("pre", pre), ("clf", clf)])
