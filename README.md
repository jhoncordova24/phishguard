# Sistema open source basado en Machine Learning para la reducción de la tasa de incidentes por phishing

Monorepo del proyecto **PhishGuard**, que integra el modelo de Machine Learning, la API de inferencia, la extensión de navegador, el dashboard y los análisis de evaluación.

## Estructura del monorepo

```
phishguard/
├── features.py          # Extracción de características de URLs (compartida por ml/ y api/)
├── requirements.txt     # Dependencias de Python
├── ml/                  # Pipeline de Machine Learning
│   ├── config.py        # Rutas, hiperparámetros y constantes
│   ├── prepare_data.py  # Preparación de datos (raw -> processed)
│   ├── train_eval.py    # Entrenamiento y evaluación de modelos
│   ├── data/raw/        # Datasets originales (no versionados)
│   ├── data/processed/  # Datasets procesados (no versionados)
│   ├── notebooks/       # Notebooks exploratorios
│   ├── models/          # Modelos entrenados serializados
│   └── reports/         # Métricas y reportes de evaluación
├── api/                 # API REST (FastAPI) para servir predicciones
│   └── main.py
├── extension/           # Extensión de navegador
├── dashboard/           # Dashboard de monitoreo
└── analysis/            # Análisis de objetivos específicos
    ├── oe2_latencia_recursos/   # OE2: latencia y consumo de recursos
    └── oe3_pretest_postest/     # OE3: evaluación pretest / postest
```
