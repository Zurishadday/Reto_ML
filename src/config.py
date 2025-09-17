# src/config.py
from pathlib import Path

__version__ = "0.1.0"
RANDOM_STATE = 42

# Rutas
DATA_PATH = Path("data/retencion.csv")
ARTIFACTS_DIR = Path("artifacts")
LOGS_DIR = Path("logs")
MONITOR_DIR = Path("monitoring")

# Versionado simple
MODEL_NAME = "baseline_logreg"
MODEL_VERSION = "0.1.0"
DATA_SCHEMA_VERSION = "1.0.0"

# columnas a excluir del modelado
DROP_COLS = [
    "CustomerID"
]

# Asegurar carpetas
for d in [ARTIFACTS_DIR, LOGS_DIR, MONITOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)
