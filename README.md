# Reto MLOps de Retención — README

> Objetivo: dejar un repositorio **ejecutable, reproducible y auditable** que cumpla con el reto:
> 1) entrenar **múltiples modelos**, 2) **registrar predicciones** de forma estructurada,
> 3) levantar un **canal de monitoreo** (drift/calidad) y 4) proponer una **arquitectura** con versionado y CI/CD.


## Orden de Ejecución

* setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

* train:
	python -m src.train --data data/retention.csv --target usuarioPerdido

* evaluate:
	python -m src.train --data data/retention.csv --target usuarioPerdido

* infer:
	python -m src.infer --model artifacts/gbt__v0.1.pkl --input data/new_batch.csv

* monitor:
	python -m src.monitor --reference data/train_with_score.csv --current data/new_batch_with_score.csv --score_col score


---

## Tabla de contenido
- [Estructura](#estructura)
- [Requisitos e instalación](#requisitos-e-instalación)
- [Datos y supuestos](#datos-y-supuestos)
- [Guía rápida (Makefile)](#guía-rápida-makefile)
- [Entrenamiento y evaluación](#entrenamiento-y-evaluación)
- [Inferencia y logging](#inferencia-y-logging)
- [Monitoreo y umbrales](#monitoreo-y-umbrales)
- [Métricas de negocio y umbral de decisión](#métricas-de-negocio-y-umbral-de-decisión)
- [Arquitectura propuesta](#arquitectura-propuesta)
- [Entregables exigidos](#entregables-exigidos)

---

---
## Estructura
## Requisitos e instalación

conda create -n mlops-retencion python=3.11 -y
conda activate mlops-retencion
pip install -r requirements.txt


## Datos y supuestos

* Target binario: usuarioPerdido (1 = perdido, 0 = retenido).

* IDs (no usar como features): ej. CustomerID → excluir en config.py (DROP_COLS).

* features.py detecta tipos (numéricas/categóricas), imputa nulos (mediana / más frecuente) y aplica OneHotEncoder(handle_unknown="ignore").

* Clase positiva esperada ~26% (ejemplo del reto). Ajustar umbral según capacidad operativa.


## Guía rápida (Makefile)

make setup        # instala dependencias
make train        # entrena 3 modelos (logreg, xgb, mlp) y guarda artefactos
make evaluate     # re-ejecuta entrenamiento y exporta reportes de test
make infer        # genera logs de predicciones (CSV) con el modelo campeón
make monitor      # calcula PSI/KS y emite alertas de drift


Archivos generados clave

* artifacts/experiments_summary.json — historial de corridas/validación

* artifacts/best_test_report.json — métricas del mejor modelo en test

* artifacts/<modelo>__vX.Y.pkl — artefacto serializado

* artifacts/<modelo>__calibration.csv — puntos para curva de calibración

* logs/predictions.csv — ≥ 100 eventos para cumplir el reto

* monitoring/drift_report.json — reporte de drift (PSI/KS + alertas)

* monitoring/train_with_score.csv — baseline de score para drift



Entrenamiento y evaluación

* Entrenamos tres familias de modelos:

* Logistic Regression (lineal)

* XGBoost (árboles impulsados)

* MLP (red neuronal simple)

## Métricas de evaluación (valid/test):

* ROC-AUC, PR-AUC, F1, Precision@K, Lift@K, Curva de calibración.

Selección de campeón:

* Se elige por mejor ROC-AUC en validación (desempate por PR-AUC/F1).

* Se guarda reporte de test del campeón y sus artefactos/versiones.


## Inferencia y logging

make infer

python -m src.predict --data data/raw/retencion.csv --target usuarioPerdido --sample 0.2 --date 2025-09-15



## Entregables exigidos

* Repo con src/, notebooks/, logs/, artifacts/, monitoring/

* README con comandos: make setup/train/evaluate/infer/monitor

* Múltiples modelos entrenados + log de experimentos

* Logs locales de predicción (≥ 100 eventos)

* Reporte de monitoreo con drift + umbrales

* Diagrama de arquitectura y propuesta de CI/CD/versionado


## Estructura
```txt
mlops-retencion/
├─ data/
│  └─ raw/retencion.csv         # dataset fuente (binario, target: usuarioPerdido)
├─ artifacts/                   # modelos, reportes, metadata
├─ logs/                        # registros de predicciones (CSV/fecha)
├─ monitoring/                  # EDA, drift, métricas, alertas
├─ notebooks/                   # opcional: exploración manual
├─ src/
│  ├─ __init__.py
│  ├─ config.py                 # seeds, columnas a descartar, versiones
│  ├─ data.py                   # carga y validaciones básicas
│  ├─ features.py               # imputación y OHE; numéricas/categóricas
│  ├─ eda.py                    # nulos, dtypes, balance, fuga, drift inicial
│  ├─ models.py                 # definiciones: logreg / xgb / mlp
│  ├─ metrics.py                # métricas y búsqueda de umbral
│  ├─ train.py                  # baseline
│  ├─ train_multi.py            # 3 modelos + logging de experimentos
│  ├─ select_champion.py        # elige campeón por métrica
│  ├─ eval_champion.py          # curvas ROC/PR, matriz, Precision@K/Lift
│  ├─ predict.py                # “producción” simulada (genera logs)
│  ├─ build_reference.py        # referencia de scores para drift
│  └─ monitor.py                # monitoreo batch + alertas (PSI/KS)
└─ requirements.txt