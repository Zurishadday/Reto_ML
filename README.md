===== README.md =====
# Reto MLOps de Retención — README (entrega)

> **Objetivo**: dejar un repositorio ejecutable, reproducible y auditable que cumpla con el reto: (1) entrenar **múltiples modelos**, (2) **registrar predicciones** de forma estructurada, (3) levantar un **canal de monitoreo** (drift/calidad) y (4) proponer una **arquitectura de versionado y CI/CD**.

---

## 📦 Estructura del repo
mlops-retencion/
├─ data/
│ └─ raw/retencion.csv # dataset fuente
├─ artifacts/ # modelos, metadata y experimentos
├─ logs/ # registros de predicciones por lote/fecha
├─ monitoring/ # EDA, reportes de drift, métricas diarias
├─ notebooks/ # (opcional) exploración manual
├─ src/
│ ├─ init.py
│ ├─ config.py
│ ├─ data.py
│ ├─ features.py
│ ├─ eda.py
│ ├─ train.py # baseline
│ ├─ models.py # definiciones: logreg/xgb/mlp
│ ├─ metrics.py # umbral & métricas
│ ├─ train_multi.py # 3 modelos + logging estructurado
│ ├─ select_champion.py # seleccionar campeón por métrica
│ ├─ eval_champion.py # curvas, matriz confusión, p@k
│ ├─ predict.py # "producción" (genera logs por fecha)
│ ├─ build_reference.py # referencia de scores
│ └─ monitor.py # monitor diario + alertas
└─ requirements.txt


> **Nota**: Python 3.11 recomendado. Entorno sugerido: `conda create -n mlops-retencion python=3.11 -y`.

---

## ⚙️ Instalación rápida
```bash
conda activate mlops-retencion
pip install -r requirements.txt

🧠 Supuestos de datos

Target (binario): usuarioPerdido (1 = se perdió, 0 = retenido).

IDs (no usar como features): CustomerID → se excluye en config.DROP_COLS.

features.py detecta tipos automáticamente (numéricas vs categóricas) e imputa nulos (median/most_frequent). Categóricas → OneHotEncoder(handle_unknown="ignore")

▶️ Guía de ejecución (de cero a resultados)

1) Baseline

python -m src.train --data data/raw/retencion.csv --target usuarioPerdido

Resultado: artifacts/baseline_logreg_*.joblib + artifacts/metadata.json.

2) EDA + Fuga + Drift (train vs test)

python -m src.eda --data data/raw/retencion.csv --target usuarioPerdido

Salida clave:

monitoring/eda_overview.json (nulos, dtypes, balance, duplicados)

monitoring/eda/*.png (histos/barras)

monitoring/leakage_suspects.csv (posibles fugas por MI)

monitoring/drift_report_train_vs_test.csv (KS/Chi²)

3) Tres modelos + umbral + logging de TEST

python -m src.train_multi --data data/raw/retencion.csv --target usuarioPerdido

Salida:

artifacts/experiments.csv (historial de corridas)

{modelo}_{version}_{timestamp}.joblib/.json (artefactos)

logs/preds_{modelo}_{version}_{timestamp}_TEST.csv (preds por fila)

4) Elegir campeón y evaluarlo

python -m src.select_champion
python -m src.eval_champion


Salida:

artifacts/champion_model.joblib, champion_meta.json, champion_info.json

monitoring/eval_{token}.json, *_pr_curve.png, *_roc_curve.png

Precision@K (5%, 10%, 20%) para decisiones operativas.

5) “Producción” (simulada) + Monitoreo

Generar lotes diarios (logs) usando el campeón:

python -m src.predict --data data/raw/retencion.csv --target usuarioPerdido --sample 0.2 --date 2025-09-14
python -m src.predict --data data/raw/retencion.csv --target usuarioPerdido --sample 0.2 --date 2025-09-15

Crear referencia de scores (una vez):

python -m src.build_reference

Correr el monitor:

python -m src.monitor

Salida:

* monitoring/daily_metrics.csv (fecha, n, pos_rate_pred, ks_p_score, AUC/PR/F1 si hay y_true)

* monitoring/alerts.json (reglas & alertas).

🧪 Métricas clave y selección de umbral

Entrenamos con probabilidades y elegimos el umbral que maximiza F1 en VALIDATION. Ese umbral se guarda junto al modelo.

Comparativa de modelos por PR-AUC(test) (prioriza positivos raros) y desempate por F1(test).

Precision@K conecta el modelo con la capacidad operativa (si solo contactas al top 10–20%).

🧾 Esquemas de archivos (contratos)
artifacts/experiments.csv

experiment_id, trained_at_utc, model_token, model_name, model_version,
threshold, roc_auc_val, pr_auc_val, f1_val, roc_auc_test, pr_auc_test, f1_test,
n_train, n_val, n_test


🔎 Monitoreo (reglas por defecto)

* Drift de scores: KS vs referencia (p < 0.01 ⇒ alerta DRIFT_SCORES).

* Tasa de positivos predicha: rango sano [0.05, 0.60] (configurable) ⇒ POS_RATE_OUT_OF_RANGE.

* Si llega y_true en los logs, calculamos ROC-AUC/PR-AUC/F1 diarios.

Extensión rápida: agregar un subset de features a los logs y reutilizar KS/Chi² de src.eda para monitorear inputs.

🧰 Configuración relevante (src/config.py)

* RANDOM_STATE, MODEL_VERSION, DATA_SCHEMA_VERSION.

* DROP_COLS para excluir IDs y posibles fugas (e.g., CustomerID, fechaBaja, flagCancelado).

🧱 CI/CD propuesto (GCP)
Flujo (alto nivel)

flowchart TD
  A[Dev push a main] --> B[GitHub Actions: CI]
  B -->|build & test| C[Construir imagen Docker]
  C --> D[Artifact Registry]
  D --> E[Cloud Run Job: predict diario]
  D --> F[Cloud Run Job: monitor diario]
  E -->|logs CSV| G[(Cloud Storage / BigQuery)]
  F -->|daily_metrics & alerts| G
  F --> H[Notificación (Email/Slack/Teams)]


📚 Comandos de referencia

# 1) Baseline
python -m src.train --data data/retencion.csv --target usuarioPerdido

# 2) EDA + drift
python -m src.eda --data data/retencion.csv --target usuarioPerdido

# 3) Tres modelos + logging TEST
python -m src.train_multi --data data/retencion.csv --target usuarioPerdido

# 4) Campeón + evaluación
python -m src.select_champion
python -m src.eval_champion

# 5) Producción simulada + referencia + monitoreo
python -m src.predict --data data/raw/retencion.csv --target usuarioPerdido --sample 0.2 --date 2025-09-14
python -m src.predict --data data/raw/retencion.csv --target usuarioPerdido --sample 0.2 --date 2025-09-15
python -m src.build_reference
python -m src.monitor
