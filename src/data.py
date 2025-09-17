# src/data.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(csv_path: str | None = None) -> pd.DataFrame:
    path = csv_path if csv_path else "data/raw/retencion.csv"
    df = pd.read_csv(path)
    # Limpieza básica y nombres consistentes
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip()
    return df

def split_stratified(df: pd.DataFrame, target: str = "usuarioPerdido",
                     test_size: float = 0.2, val_size: float = 0.2,
                     random_state: int = 42):
    assert target in df.columns, f"No encuentro la columna objetivo '{target}'"
    X = df.drop(columns=[target])
    y = df[target]

    # train_temp/test
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # train/val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=val_ratio,
        stratify=y_train_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
