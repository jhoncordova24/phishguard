"""Configuración central del módulo de ML: rutas, hiperparámetros y constantes.

Todas las rutas se calculan a partir de la raíz del repositorio, así los
scripts funcionan igual sin importar desde qué carpeta se ejecuten.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# Este archivo está en <raíz>/ml/config.py, así que la raíz es dos niveles arriba.
ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml"

DATA_DIR = ML_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # datasets originales (NO se modifican)
PROCESSED_DIR = DATA_DIR / "processed"  # datasets generados por nuestros scripts

MODELS_DIR = ML_DIR / "models"
REPORTS_DIR = ML_DIR / "reports"
TABLES_DIR = REPORTS_DIR / "tables"     # tablas (CSV) para la tesis
FIGURES_DIR = REPORTS_DIR / "figures"   # figuras (PNG) para la tesis
DIAGNOSTICOS_DIR = TABLES_DIR / "diagnosticos"  # tablas auxiliares (no van en 5.1)

# ---------------------------------------------------------------------------
# Datasets originales
# ---------------------------------------------------------------------------
# PhiUSIIL Phishing URL Dataset.
# IMPORTANTE: en este dataset label = 1 significa LEGÍTIMA y label = 0 significa PHISHING.
PHIUSIIL_PATH = RAW_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
PHIUSIIL_ENCODING = "utf-8-sig"         # el archivo trae BOM al inicio
PHIUSIIL_COLS = ["URL", "label"]        # solo cargamos las columnas que usamos
PHIUSIIL_LABEL_LEGITIMA = 1
PHIUSIIL_LABEL_PHISHING = 0

# Malicious URLs Dataset (columnas: url, type).
MALICIOUS_PATH = RAW_DIR / "malicious_phish.csv"
MALICIOUS_COLS = ["url", "type"]
MALICIOUS_CLASES = ["benign", "defacement", "phishing", "malware"]
MALICIOUS_CLASES_USADAS = ["benign", "phishing"]  # las que entran al estudio

# ---------------------------------------------------------------------------
# Convención de etiquetas del proyecto (para fases posteriores)
# ---------------------------------------------------------------------------
NOMBRE_LEGITIMA = "legítima"
NOMBRE_PHISHING = "phishing"

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42  # semilla fija para que los resultados se puedan repetir
RANDOM_SEED = SEED  # alias (nombre usado antes); usa SEED en código nuevo

# ---------------------------------------------------------------------------
# Data Preparation (ml/prepare_data.py)
# ---------------------------------------------------------------------------
# Etiqueta unificada del proyecto: la clase POSITIVA es phishing.
LABEL_PHISHING = 1
LABEL_LEGITIMA = 0

# Nombres de la columna "source" (de qué repositorio viene cada URL).
SOURCE_PHIUSIIL = "phiusiil"
SOURCE_MALICIOUS = "malicious_phish"

# Qué hacer con las URL que no traen esquema (ej. "ejemplo.com/login" sin "http://"):
#   "drop" -> se descartan (lectura literal de la tesis). Valor por defecto.
#   "keep" -> se conservan; features.py igual sabe extraer su host.
# El script imprime cuántas serían afectadas por fuente y clase antes de decidir.
MISSING_SCHEME_POLICY = "drop"

# Muestra balanceada (sin reemplazo). Dos modos:
#   "class"          -> 25.000 por clase; cada fuente entra en la proporción en
#                       que aparezca al azar (iteración 1: ~80 % PhiUSIIL).
#   "class_x_source" -> 4 celdas clase x fuente de 12.500 cada una (iteración 2).
#                       Así ninguna fuente impone su "formato" de URL a una clase.
SAMPLING_MODE = "class_x_source"
MUESTRA_POR_CLASE = 25_000    # usado si SAMPLING_MODE = "class"
MUESTRA_POR_CELDA = 12_500    # usado si SAMPLING_MODE = "class_x_source"

# Salidas de prepare_data.py
URLS_CLEAN_PATH = PROCESSED_DIR / "urls_clean.csv"
DATASET_FEATURES_PATH = PROCESSED_DIR / "dataset_50k_features.csv"
TABLA_DESCARTES_PATH = TABLES_DIR / "tabla_descartes_depuracion.csv"
TABLA_SIN_ESQUEMA_PATH = DIAGNOSTICOS_DIR / "tabla_urls_sin_esquema.csv"
TABLA_MUESTRA_PATH = TABLES_DIR / "tabla_muestra_balanceada.csv"
TABLA_DESCRIPTIVOS_PATH = TABLES_DIR / "tabla_descriptivos_atributos.csv"

# ---------------------------------------------------------------------------
# Modeling / Evaluation / Deployment (ml/train_eval.py)
# ---------------------------------------------------------------------------
# Partición 80/20 estratificada por label (40.000 entrenamiento / 10.000 prueba).
TEST_SIZE = 0.20
TRAIN_40K_PATH = PROCESSED_DIR / "train_40k.csv"
TEST_10K_PATH = PROCESSED_DIR / "test_10k.csv"
PREDICCIONES_TEST_PATH = PROCESSED_DIR / "predicciones_test.csv"

# Búsqueda de hiperparámetros (RandomizedSearchCV).
# N_ITER es IGUAL para ambos algoritmos: así ninguno recibe "más oportunidades"
# de encontrar una buena combinación que el otro (comparación justa).
N_ITER = 30
N_ITER_QUICK = 3          # solo para probar el flujo con --quick
QUICK_N_FILAS = 2_000     # filas que usa --quick (muestra estratificada)
CV_FOLDS = 5
SCORING = "f1"            # F1 de la clase positiva (phishing = 1)

# Grillas: listas de valores discretos. RandomizedSearchCV elige N_ITER
# combinaciones al azar (sin repetir) de todas las posibles.
GRILLA_RF = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],          # None = el árbol crece sin límite
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
GRILLA_XGB = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# Contraste estadístico (hipótesis 1).
N_BOOTSTRAP = 2_000
NIVEL_CONFIANZA = 0.95
ALFA = 0.05
MCNEMAR_EXACTO_SI_MENOS_DE = 25  # si b + c < 25 se usa la prueba binomial exacta

# Medición de T_inf en laboratorio (solo cómputo del modelo).
T_INF_CALENTAMIENTO = 50
T_INF_MEDICIONES = 1_000
T_INF_UMBRAL_MS = 50.0

# Diagnóstico por fuente (NO es criterio de decisión): umbrales de advertencia.
ALERTA_F1_ALTO = 0.99
ALERTA_DIF_FUENTES = 0.05   # diferencia de F1 entre fuentes considerada "grande"

# Artefacto desplegable.
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"
ENVIRONMENT_PATH = REPORTS_DIR / "environment.json"
ALERTA_TAMANO_MB = 50.0     # GitHub rechaza archivos de más de 100 MB

# Carpeta separada para las corridas de prueba (--quick), para no pisar
# los resultados oficiales de la tesis.
QUICK_DIR = ML_DIR / "_quick_run"
