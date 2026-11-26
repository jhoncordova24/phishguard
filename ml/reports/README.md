# Reportes del módulo de ML

Resultados **oficiales** de la tesis, correspondientes a la **Iteración 2** de CRISP-DM
(muestreo 4 celdas clase × fuente de 12.500 URL, canonicalización sin "/" final cuando la
ruta es solo "/"). Modelo ganador: **XGBoost**. Ficha completa en `ml/models/model_card.json`.

## Tablas oficiales (`tables/`)

| Subsección | Archivo | Contenido | Generado por |
|---|---|---|---|
| 5.1.2 Comprensión de los datos | `tabla_composicion_repositorios.csv` | Registros por repositorio y clase en los datos originales | `ml/data_understanding.py` |
| 5.1.3 Preparación de los datos | `tabla_descartes_depuracion.csv` | Registros descartados y restantes en cada regla de depuración | `ml/prepare_data.py` |
| 5.1.3 Preparación de los datos | `tabla_muestra_balanceada.csv` | Composición de la muestra de 50.000 URL por clase y fuente | `ml/prepare_data.py` |
| 5.1.3 Preparación de los datos | `tabla_descriptivos_atributos.csv` | Media y mediana de los 20 atributos por clase | `ml/prepare_data.py` |
| 5.1.4 Modelado | `tabla_hiperparametros.csv` | Espacio de búsqueda y valor óptimo de cada hiperparámetro (RF y XGBoost) | `ml/train_eval.py` |
| 5.1.4 Modelado | `tabla_f1_por_fold.csv` | F1 de cada uno de los 5 folds, media y desviación estándar | `ml/train_eval.py` |
| 5.1.5 Evaluación | `tabla_matriz_confusion.csv` | TP, TN, FP, FN en el conjunto de prueba (10.000 URL) | `ml/train_eval.py` |
| 5.1.5 Evaluación | `tabla_metricas_test.csv` | Exactitud, precisión, sensibilidad y F1 en prueba | `ml/train_eval.py` |
| 5.1.5 Evaluación | `tabla_contraste_decision.csv` | Bootstrap pareado de ΔF1, McNemar y regla de decisión | `ml/train_eval.py` |
| 5.1.5 Evaluación | `tabla_t_inf_laboratorio.csv` | Tiempo de inferencia por URL (media, mediana, p95) | `ml/train_eval.py` |
| 5.1.6 Despliegue | `tabla_ficha_artefacto.csv` | Ficha del modelo serializado (versión tabla del `model_card.json`) | `ml/train_eval.py` |

## Figuras oficiales (`figures/`)

| Subsección | Archivo | Contenido |
|---|---|---|
| 5.1.2 Comprensión de los datos | `figura_distribucion_clases.png` | Distribución de clases por repositorio |
| 5.1.5 Evaluación | `figura_matrices_confusion.png` | Matrices de confusión de RF y XGBoost en prueba |

## `tables/diagnosticos/` (auxiliares, NO son criterio de decisión)

Evidencia de apoyo que justifica la Iteración 2. No va como tabla principal en 5.1.

- `diagnostico_atajos_train.csv` y `diagnostico_atributo_unico_train.csv`: análisis de atajos
  del dataset. **Se calcularon sobre `train_40k.csv` de la Iteración 1** (motivaron la Iteración 2);
  hay una copia idéntica en `archive/iteracion_1/`.
- `diagnostico_por_fuente.csv`: F1 en prueba separado por fuente (Iteración 2).
- `diagnostico_cruzado.csv`: entrenar con una fuente y probar en la otra (Iteración 2).
- `tabla_urls_sin_esquema.csv`: cuántas URL sin esquema hay por fuente y clase, y cuántas
  descarta la política `MISSING_SCHEME_POLICY = "drop"`.

## `archive/iteracion_1/` (en `ml/archive/`)

Copia completa de la Iteración 1 (muestreo 25.000 por clase sin controlar la fuente, F1 ≈ 0,993),
conservada como evidencia del atajo detectado: `reports/` (tablas y figuras), `models/model_card.json`
y, solo en local (en `.gitignore`), `data/` con los CSV de entrenamiento, prueba y predicciones, y el `.joblib`.

## Notas

- `python ml/train_eval.py --quick` escribe en `ml/_quick_run/` (también en `.gitignore`),
  nunca en esta carpeta.
