# src/features.py
from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def infer_column_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target]) if target in df.columns else df.copy()
    cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
    num_cols = list(X.select_dtypes(include=["number", "bool"]).columns)
    return num_cols, cat_cols

def build_preprocessor(df: pd.DataFrame, target: str = "usuarioPerdido") -> ColumnTransformer:
    num_cols, cat_cols = infer_column_types(df, target)
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return pre

def infer_column_types(df: pd.DataFrame, target: str):
    from src.config import DROP_COLS
    X = df.drop(columns=[c for c in ([target] + DROP_COLS) if c in df.columns]).copy()
    cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
    num_cols = list(X.select_dtypes(include=["number", "bool"]).columns)
    return num_cols, cat_cols

