"""Fases CRISP-DM 4, 5 y 6: Modeling, Evaluation y Deployment (serialización).

Qué hace este script, en orden:
  MODELING
    1. Parte el dataset de 50.000 URL en 80/20 estratificado por label
       (40.000 entrenamiento / 10.000 prueba) y guarda ambas partes en CSV.
    2. Busca hiperparámetros de Random Forest y XGBoost con RandomizedSearchCV
       + validación cruzada estratificada de 5 folds, SOLO sobre las 40.000.
    3. Guarda el F1 de cada fold del mejor candidato de cada algoritmo.
  EVALUATION
    4. Evalúa cada mejor modelo UNA sola vez sobre las 10.000 de prueba.
    5. Contraste estadístico: bootstrap pareado de ΔF1 y prueba de McNemar.
    6. Mide T_inf (tiempo de inferencia de una fila) en laboratorio.
    7. Aplica la regla de decisión (tesis 4.9.2.e) y elige el modelo.
    8. Diagnóstico por fuente y diagnóstico cruzado entrenar-en-una/probar-en-otra
       (solo informativos).
  DEPLOYMENT
    9.  Serializa el modelo elegido (joblib) y genera su ficha (model card).
    10. Verifica que el .joblib recargado reproduce las predicciones y que
        funciona con URL crudas pasando por features.extract_features.
    11. Registra el entorno de hardware/software (Tabla XVII).

Uso (desde la raíz del repositorio, con el entorno virtual activo):
    python ml/train_eval.py            # corrida oficial (n_iter = 30)
    python ml/train_eval.py --quick    # prueba rápida: 2.000 filas, n_iter = 3
    python ml/train_eval.py --n-iter 50

Las corridas --quick escriben en ml/_quick_run/ para no pisar los resultados
oficiales.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

# Permite importar "ml.config" y "features" aunque el script se ejecute suelto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib

matplotlib.use("Agg")  # dibuja a archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier

import features
from features import FEATURE_NAMES
from ml import config

ALGORITMOS = ["random_forest", "xgboost"]
NOMBRE_BONITO = {"random_forest": "Random Forest", "xgboost": "XGBoost"}
SUFIJO = {"random_forest": "rf", "xgboost": "xgb"}  # para columnas pred_rf, proba_xgb...

# Librerías cuya versión se registra en la ficha y en environment.json.
LIBRERIAS = ["numpy", "pandas", "scikit-learn", "xgboost", "joblib",
             "scipy", "statsmodels", "matplotlib"]


# ===========================================================================
# Utilidades
# ===========================================================================
def titulo(texto: str):
    print("\n" + "=" * 78 + f"\n{texto}\n" + "=" * 78)


def rutas_salida(quick: bool) -> dict:
    """Devuelve todas las rutas de salida.

    En modo --quick se usan las mismas rutas relativas pero dentro de
    ml/_quick_run/, así una prueba nunca sobrescribe los archivos de la tesis.
    """
    oficiales = {
        "train": config.TRAIN_40K_PATH,
        "test": config.TEST_10K_PATH,
        "predicciones": config.PREDICCIONES_TEST_PATH,
        "modelo": config.BEST_MODEL_PATH,
        "model_card": config.MODEL_CARD_PATH,
        "environment": config.ENVIRONMENT_PATH,
        "tablas": config.TABLES_DIR,
        "diagnosticos": config.DIAGNOSTICOS_DIR,
        "figuras": config.FIGURES_DIR,
    }
    if quick:
        oficiales = {k: config.QUICK_DIR / Path(v).relative_to(config.ML_DIR)
                     for k, v in oficiales.items()}
    for ruta in oficiales.values():
        carpeta = ruta if ruta.suffix == "" else ruta.parent
        carpeta.mkdir(parents=True, exist_ok=True)
    return oficiales


def sha256_archivo(ruta: Path) -> str:
    """Huella digital del archivo: si cambia un solo byte, cambia el hash."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def versiones_librerias() -> dict:
    versiones = {}
    for lib in LIBRERIAS:
        try:
            versiones[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versiones[lib] = "no instalada"
    return versiones


def matriz_X(df: pd.DataFrame) -> np.ndarray:
    """Convierte las 20 columnas de atributos en una matriz numérica.

    Siempre en el orden de FEATURE_NAMES: el modelo no conoce nombres de
    columnas, solo posiciones, así que el orden debe ser idéntico en
    entrenamiento, prueba y producción (api/).
    """
    return df[FEATURE_NAMES].to_numpy(dtype=np.float64)


def f1_rapido(y, pred) -> float:
    """F1 de la clase phishing con conteos (más veloz que sklearn para el bootstrap)."""
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    denominador = 2 * tp + fp + fn
    return 0.0 if denominador == 0 else 2 * tp / denominador


def a_texto(valor) -> str:
    """Formatea un valor de hiperparámetro para las tablas (None -> 'None')."""
    return "None" if valor is None else str(valor)


# ===========================================================================
# MODELING
# ===========================================================================
def particionar(quick: bool, rutas: dict):
    """Paso 1: partición 80/20 estratificada por label con la semilla fija."""
    titulo("MODELING - Paso 1: partición entrenamiento / prueba")
    df = pd.read_csv(config.DATASET_FEATURES_PATH)
    print(f"Dataset: {config.DATASET_FEATURES_PATH.name} -> {len(df):,} filas")

    # Revisiones de seguridad: si algo de esto falla, el dataset está mal armado.
    faltan = [c for c in ["url", "label", "source", *FEATURE_NAMES] if c not in df.columns]
    assert not faltan, f"Faltan columnas en el dataset: {faltan}"
    assert df[FEATURE_NAMES].notna().all().all(), "Hay atributos vacíos (NaN)"
    assert set(df["label"].unique()) <= {0, 1}, "label debe ser 0 o 1"

    if quick:
        # Para la prueba rápida tomamos 2.000 filas, también estratificadas,
        # para que la proporción 50/50 se mantenga.
        df, _ = train_test_split(df, train_size=config.QUICK_N_FILAS,
                                 stratify=df["label"], random_state=config.SEED)
        print(f"[--quick] Se usa una muestra estratificada de {len(df):,} filas")

    # stratify=label garantiza que entrenamiento y prueba conserven la misma
    # proporción de phishing/legítimas que el total (50/50).
    train, test = train_test_split(df, test_size=config.TEST_SIZE,
                                   stratify=df["label"], random_state=config.SEED)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Guardamos la partición: cualquiera puede repetir el experimento con
    # exactamente las mismas filas.
    train.to_csv(rutas["train"], index=False)
    test.to_csv(rutas["test"], index=False)

    for nombre, parte in [("Entrenamiento", train), ("Prueba", test)]:
        n_phish = int((parte["label"] == config.LABEL_PHISHING).sum())
        n_legit = int((parte["label"] == config.LABEL_LEGITIMA).sum())
        print(f"{nombre:<14} {len(parte):>6,} filas | phishing {n_phish:,} "
              f"({n_phish / len(parte):.2%}) | legítima {n_legit:,} "
              f"({n_legit / len(parte):.2%})")
        assert n_phish == n_legit, f"{nombre} no quedó 50/50"
    print("OK: ambos subconjuntos quedaron exactamente 50/50.")
    print(f"Guardado: {rutas['train']}\nGuardado: {rutas['test']}")
    return train, test


def crear_estimador(algoritmo: str):
    """Modelo base (sin tunear) de cada algoritmo, con semilla fija.

    n_jobs=1 dentro del modelo porque el paralelismo (n_jobs=-1) lo hace
    RandomizedSearchCV entrenando varios candidatos/folds a la vez; si ambos
    fueran -1 los núcleos se pelearían entre sí y sería más lento.
    """
    if algoritmo == "random_forest":
        return RandomForestClassifier(random_state=config.SEED, n_jobs=1)
    return XGBClassifier(random_state=config.SEED, n_jobs=1, tree_method="hist",
                         eval_metric="logloss")


def buscar_hiperparametros(algoritmo: str, X_train, y_train, n_iter: int):
    """Paso 2: RandomizedSearchCV con validación cruzada estratificada de 5 folds."""
    grilla = config.GRILLA_RF if algoritmo == "random_forest" else config.GRILLA_XGB
    # StratifiedKFold: cada fold mantiene el 50/50; shuffle con semilla fija.
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    busqueda = RandomizedSearchCV(
        estimator=crear_estimador(algoritmo),
        param_distributions=grilla,
        n_iter=n_iter,
        scoring=config.SCORING,   # F1 de phishing (clase 1)
        cv=cv,
        refit=True,               # al final reentrena el mejor con las 40.000 completas
        random_state=config.SEED,
        n_jobs=-1,
        verbose=1,
    )
    inicio = time.perf_counter()
    busqueda.fit(X_train, y_train)
    segundos = time.perf_counter() - inicio
    print(f"{NOMBRE_BONITO[algoritmo]}: búsqueda terminada en {segundos:.1f} s")
    print(f"  Mejores hiperparámetros: {busqueda.best_params_}")
    return busqueda, grilla


def f1_por_fold(busqueda) -> list:
    """Paso 3: F1 de cada uno de los 5 folds del mejor candidato."""
    i = busqueda.best_index_
    return [float(busqueda.cv_results_[f"split{k}_test_score"][i])
            for k in range(config.CV_FOLDS)]


def fase_modeling(train, n_iter: int, rutas: dict):
    titulo(f"MODELING - Pasos 2 y 3: RandomizedSearchCV (n_iter={n_iter}, "
           f"{config.CV_FOLDS} folds, scoring=F1)")
    X_train = matriz_X(train)
    y_train = train["label"].to_numpy()

    modelos, filas_hp, filas_fold = {}, [], []
    for algoritmo in ALGORITMOS:
        busqueda, grilla = buscar_hiperparametros(algoritmo, X_train, y_train, n_iter)
        modelos[algoritmo] = busqueda

        for parametro, valores in grilla.items():
            filas_hp.append({
                "algoritmo": NOMBRE_BONITO[algoritmo],
                "hiperparametro": parametro,
                "espacio_evaluado": "{" + ", ".join(a_texto(v) for v in valores) + "}",
                "n_iter": n_iter,
                "valor_optimo": a_texto(busqueda.best_params_[parametro]),
            })

        f1s = f1_por_fold(busqueda)
        fila = {"algoritmo": NOMBRE_BONITO[algoritmo]}
        fila.update({f"fold_{k + 1}": round(v, 6) for k, v in enumerate(f1s)})
        fila["media"] = round(float(np.mean(f1s)), 6)
        # Desviación estándar muestral (ddof=1), la usual al reportar 5 mediciones.
        fila["desv_est_muestral"] = round(float(np.std(f1s, ddof=1)), 6)
        filas_fold.append(fila)
        print(f"  F1 por fold: {[round(v, 4) for v in f1s]} -> media "
              f"{fila['media']:.4f} ± {fila['desv_est_muestral']:.4f}")

    pd.DataFrame(filas_hp).to_csv(rutas["tablas"] / "tabla_hiperparametros.csv", index=False)
    pd.DataFrame(filas_fold).to_csv(rutas["tablas"] / "tabla_f1_por_fold.csv", index=False)
    return modelos, filas_fold


# ===========================================================================
# EVALUATION
# ===========================================================================
def metricas(y, pred) -> dict:
    """Matriz de confusión y las 4 métricas, con phishing (1) como clase positiva."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "exactitud": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "sensibilidad": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
    }


def evaluar_en_prueba(modelos, test, rutas):
    """Paso 4: cada modelo predice las 10.000 de prueba UNA sola vez."""
    titulo("EVALUATION - Paso 4: evaluación única sobre el conjunto de prueba")
    X_test = matriz_X(test)
    y_test = test["label"].to_numpy()

    preds, probas, res = {}, {}, {}
    for algoritmo in ALGORITMOS:
        modelo = modelos[algoritmo].best_estimator_
        # Una única llamada de cada tipo: el conjunto de prueba no se vuelve a tocar.
        probas[algoritmo] = modelo.predict_proba(X_test)[:, 1]  # P(phishing)
        preds[algoritmo] = modelo.predict(X_test)
        res[algoritmo] = metricas(y_test, preds[algoritmo])
        m = res[algoritmo]
        print(f"{NOMBRE_BONITO[algoritmo]:<14} TP={m['TP']} TN={m['TN']} FP={m['FP']} "
              f"FN={m['FN']} | exactitud={m['exactitud']:.4f} precisión={m['precision']:.4f} "
              f"sensibilidad={m['sensibilidad']:.4f} F1={m['f1']:.4f}")

    salida = test[["url", "label", "source"]].copy()
    for algoritmo in ALGORITMOS:
        salida[f"pred_{SUFIJO[algoritmo]}"] = preds[algoritmo]
    for algoritmo in ALGORITMOS:
        salida[f"proba_{SUFIJO[algoritmo]}"] = probas[algoritmo]
    salida.to_csv(rutas["predicciones"], index=False)

    pd.DataFrame([{"algoritmo": NOMBRE_BONITO[a], **{k: res[a][k] for k in ["TP", "TN", "FP", "FN"]}}
                  for a in ALGORITMOS]).to_csv(rutas["tablas"] / "tabla_matriz_confusion.csv", index=False)
    pd.DataFrame([{"algoritmo": NOMBRE_BONITO[a], "n_prueba": len(test),
                   **{k: round(res[a][k], 6) for k in ["exactitud", "precision", "sensibilidad", "f1"]}}
                  for a in ALGORITMOS]).to_csv(rutas["tablas"] / "tabla_metricas_test.csv", index=False)
    figura_matrices(res, rutas)
    return X_test, y_test, preds, probas, res


def figura_matrices(res, rutas):
    """Las dos matrices de confusión lado a lado, con etiquetas en español."""
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.8))
    clases = ["Legítima (0)", "Phishing (1)"]
    for eje, algoritmo in zip(ejes, ALGORITMOS):
        m = res[algoritmo]
        matriz = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        eje.imshow(matriz, cmap="Blues")
        nombres = [["VN", "FP"], ["FN", "VP"]]
        for i in range(2):
            for j in range(2):
                color = "white" if matriz[i, j] > matriz.max() / 2 else "black"
                eje.text(j, i, f"{nombres[i][j]}\n{matriz[i, j]:,}", ha="center",
                         va="center", color=color, fontsize=12)
        eje.set_xticks([0, 1], clases)
        eje.set_yticks([0, 1], clases)
        eje.set_xlabel("Clase predicha")
        eje.set_ylabel("Clase real")
        eje.set_title(f"{NOMBRE_BONITO[algoritmo]} (F1 = {m['f1']:.4f})")
    fig.suptitle("Matrices de confusión en el conjunto de prueba")
    fig.tight_layout()
    ruta = rutas["figuras"] / "figura_matrices_confusion.png"
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print(f"Figura guardada: {ruta}")


def bootstrap_delta_f1(y, pred_rf, pred_xgb) -> dict:
    """Paso 5a: bootstrap PAREADO de ΔF1 = F1(RF) - F1(XGBoost).

    "Pareado" = en cada remuestreo se usan las MISMAS filas para ambos modelos,
    así la diferencia refleja solo al modelo y no a la suerte de la muestra.
    """
    rng = np.random.default_rng(config.SEED)
    n = len(y)
    deltas = np.empty(config.N_BOOTSTRAP)
    for b in range(config.N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)  # n filas con reposición
        deltas[b] = f1_rapido(y[idx], pred_rf[idx]) - f1_rapido(y[idx], pred_xgb[idx])
    cola = (1 - config.NIVEL_CONFIANZA) / 2 * 100
    ic_inf, ic_sup = np.percentile(deltas, [cola, 100 - cola])
    return {
        "delta_f1_observado": f1_rapido(y, pred_rf) - f1_rapido(y, pred_xgb),
        "ic_inf": float(ic_inf), "ic_sup": float(ic_sup),
        # Si el intervalo incluye el 0, no se puede afirmar que uno sea mejor.
        "paridad": bool(ic_inf <= 0 <= ic_sup),
    }


def prueba_mcnemar(y, pred_rf, pred_xgb) -> dict:
    """Paso 5b: McNemar sobre aciertos/errores (complemento del bootstrap).

    Solo importan las filas donde los modelos NO coinciden:
      b = RF acierta y XGBoost falla;  c = RF falla y XGBoost acierta.
    """
    acierta_rf = pred_rf == y
    acierta_xgb = pred_xgb == y
    a = int(np.sum(acierta_rf & acierta_xgb))
    b = int(np.sum(acierta_rf & ~acierta_xgb))
    c = int(np.sum(~acierta_rf & acierta_xgb))
    d = int(np.sum(~acierta_rf & ~acierta_xgb))
    # Con pocos casos discordantes la aproximación chi-cuadrado no es fiable:
    # se usa la versión exacta (binomial).
    exacta = (b + c) < config.MCNEMAR_EXACTO_SI_MENOS_DE
    r = mcnemar([[a, b], [c, d]], exact=exacta, correction=True)
    p = float(r.pvalue)
    return {"b": b, "c": c, "exacta": exacta, "estadistico": float(r.statistic),
            "p_valor": p, "significativo": p < config.ALFA}


def medir_t_inf(modelo, X_test) -> dict:
    """Paso 7: tiempo de inferencia de UNA fila (solo el cómputo del modelo).

    Según 3.5.1 no se mide la extracción de atributos ni la red: los vectores
    ya están en memoria. Se usa n_jobs=1, como en un servidor que atiende una
    petición a la vez.
    """
    modelo.set_params(n_jobs=1)
    total = config.T_INF_CALENTAMIENTO + config.T_INF_MEDICIONES
    # Preparamos las filas antes de cronometrar para no medir el "recorte".
    filas = [np.ascontiguousarray(X_test[i % len(X_test): i % len(X_test) + 1])
             for i in range(total)]
    tiempos_ns = []
    for i, fila in enumerate(filas):
        inicio = time.perf_counter_ns()
        modelo.predict_proba(fila)
        fin = time.perf_counter_ns()
        if i >= config.T_INF_CALENTAMIENTO:  # las primeras 50 se descartan (calentamiento)
            tiempos_ns.append(fin - inicio)
    ms = np.array(tiempos_ns) / 1e6
    return {"media_ms": float(ms.mean()), "mediana_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)), "min_ms": float(ms.min()),
            "max_ms": float(ms.max())}


def decidir(res, boot, t_inf) -> tuple:
    """Paso 6: regla de decisión de la tesis (4.9.2.e).

    - Sin paridad estadística: gana el mayor F1 en prueba.
    - Con paridad (el IC 95 % de ΔF1 contiene 0): desempata el menor T_inf (media).
    """
    if not boot["paridad"]:
        ganador = max(ALGORITMOS, key=lambda a: res[a]["f1"])
        rama = ("Rama 1: sin paridad estadística (el IC 95 % de ΔF1 no contiene 0); "
                "se elige el mayor F1 en prueba.")
    else:
        ganador = min(ALGORITMOS, key=lambda a: t_inf[a]["media_ms"])
        rama = ("Rama 2: paridad estadística (el IC 95 % de ΔF1 contiene 0); "
                "desempata el menor T_inf medio.")
    return ganador, rama


def diagnostico_por_fuente(test, preds, rutas) -> pd.DataFrame:
    """Paso 8: F1 por fuente (solo diagnóstico, NO decide nada)."""
    titulo("EVALUATION - Paso 8: diagnóstico por fuente (no es criterio de decisión)")
    filas = []
    for fuente, grupo in test.groupby("source"):
        y = grupo["label"].to_numpy()
        fila = {"source": fuente, "n": len(grupo),
                "n_phishing": int((y == 1).sum()), "n_legitima": int((y == 0).sum())}
        for algoritmo in ALGORITMOS:
            p = preds[algoritmo][grupo.index.to_numpy()]
            fila[f"f1_{SUFIJO[algoritmo]}"] = round(float(f1_score(y, p, pos_label=1, zero_division=0)), 6)
            fila[f"exactitud_{SUFIJO[algoritmo]}"] = round(float(accuracy_score(y, p)), 6)
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False))
    tabla.to_csv(rutas["diagnosticos"] / "diagnostico_por_fuente.csv", index=False)
    return tabla


def diagnostico_cruzado(modelos, train, test, rutas) -> pd.DataFrame:
    """Paso 8b: entrenar con UNA fuente y probar en la OTRA (solo diagnóstico).

    Para cada algoritmo se reentrena un modelo nuevo con los MISMOS
    hiperparámetros ya elegidos por la búsqueda (no se vuelve a tunear), usando
    solo las filas de entrenamiento de una fuente, y se mide F1 en las filas de
    prueba de cada fuente. La fila "misma fuente" sirve de referencia.
    NO decide nada: el modelo desplegado sigue siendo el de la regla 4.9.2.e.
    """
    titulo("EVALUATION - Paso 8b: diagnóstico cruzado entre fuentes (no es criterio de decisión)")
    filas = []
    fuentes = sorted(train["source"].unique())
    for algoritmo in ALGORITMOS:
        for origen in fuentes:
            tr = train[train["source"] == origen]
            modelo = crear_estimador(algoritmo).set_params(**modelos[algoritmo].best_params_)
            modelo.set_params(n_jobs=-1)
            modelo.fit(matriz_X(tr), tr["label"].to_numpy())
            for destino in fuentes:
                te = test[test["source"] == destino]
                y = te["label"].to_numpy()
                p = modelo.predict(matriz_X(te))
                m = metricas(y, p)
                filas.append({
                    "algoritmo": NOMBRE_BONITO[algoritmo],
                    "entrena_con": origen, "prueba_en": destino,
                    "tipo": "misma fuente" if origen == destino else "cruzado",
                    "n_entrenamiento": len(tr), "n_prueba": len(te),
                    **{k: m[k] for k in ["TP", "TN", "FP", "FN"]},
                    **{k: round(m[k], 6) for k in ["exactitud", "precision", "sensibilidad", "f1"]},
                })
    tabla = pd.DataFrame(filas)
    print(tabla[["algoritmo", "entrena_con", "prueba_en", "tipo", "f1", "precision",
                 "sensibilidad"]].to_string(index=False))
    tabla.to_csv(rutas["diagnosticos"] / "diagnostico_cruzado.csv", index=False)
    return tabla


def fase_evaluation(modelos, test, rutas):
    X_test, y_test, preds, probas, res = evaluar_en_prueba(modelos, test, rutas)

    titulo("EVALUATION - Paso 5: contraste estadístico (hipótesis 1)")
    boot = bootstrap_delta_f1(y_test, preds["random_forest"], preds["xgboost"])
    print(f"ΔF1 (RF - XGBoost) observado = {boot['delta_f1_observado']:+.5f}")
    print(f"IC {config.NIVEL_CONFIANZA:.0%} bootstrap pareado ({config.N_BOOTSTRAP:,} "
          f"remuestreos): [{boot['ic_inf']:+.5f}, {boot['ic_sup']:+.5f}] -> "
          f"{'PARIDAD ESTADÍSTICA' if boot['paridad'] else 'diferencia significativa'}")
    mc = prueba_mcnemar(y_test, preds["random_forest"], preds["xgboost"])
    print(f"McNemar ({'exacta' if mc['exacta'] else 'chi-cuadrado con corrección'}): "
          f"b={mc['b']} c={mc['c']} estadístico={mc['estadistico']:.4f} "
          f"p={mc['p_valor']:.4g} -> {'significativo' if mc['significativo'] else 'no significativo'} "
          f"(α={config.ALFA})")

    titulo(f"EVALUATION - Paso 7: T_inf de laboratorio ({config.T_INF_MEDICIONES:,} "
           f"mediciones, {config.T_INF_CALENTAMIENTO} de calentamiento)")
    t_inf, filas_t = {}, []
    for algoritmo in ALGORITMOS:
        t_inf[algoritmo] = medir_t_inf(modelos[algoritmo].best_estimator_, X_test)
        t = t_inf[algoritmo]
        cumple = t["p95_ms"] < config.T_INF_UMBRAL_MS
        print(f"{NOMBRE_BONITO[algoritmo]:<14} media={t['media_ms']:.3f} ms "
              f"mediana={t['mediana_ms']:.3f} ms p95={t['p95_ms']:.3f} ms -> "
              f"{'cumple' if cumple else 'NO cumple'} umbral {config.T_INF_UMBRAL_MS:.0f} ms")
        filas_t.append({"algoritmo": NOMBRE_BONITO[algoritmo],
                        **{k: round(v, 4) for k, v in t.items()},
                        "n_mediciones": config.T_INF_MEDICIONES,
                        "n_calentamiento": config.T_INF_CALENTAMIENTO,
                        "umbral_ms": config.T_INF_UMBRAL_MS,
                        "media_bajo_umbral": t["media_ms"] < config.T_INF_UMBRAL_MS,
                        "p95_bajo_umbral": cumple})
    pd.DataFrame(filas_t).to_csv(rutas["tablas"] / "tabla_t_inf_laboratorio.csv", index=False)

    titulo("EVALUATION - Paso 6: regla de decisión (tesis 4.9.2.e)")
    ganador, rama = decidir(res, boot, t_inf)
    print(rama)
    print(f"Modelo elegido: {NOMBRE_BONITO[ganador]}")

    contraste = [
        ("f1_test_random_forest", res["random_forest"]["f1"]),
        ("f1_test_xgboost", res["xgboost"]["f1"]),
        ("delta_f1_rf_menos_xgb", boot["delta_f1_observado"]),
        ("n_bootstrap", config.N_BOOTSTRAP),
        ("nivel_confianza", config.NIVEL_CONFIANZA),
        ("ic_delta_f1_inferior", boot["ic_inf"]),
        ("ic_delta_f1_superior", boot["ic_sup"]),
        ("paridad_estadistica", boot["paridad"]),
        ("mcnemar_b_rf_acierta_xgb_falla", mc["b"]),
        ("mcnemar_c_rf_falla_xgb_acierta", mc["c"]),
        ("mcnemar_exacta", mc["exacta"]),
        ("mcnemar_estadistico", mc["estadistico"]),
        ("mcnemar_p_valor", mc["p_valor"]),
        ("alfa", config.ALFA),
        ("mcnemar_significativo", mc["significativo"]),
        ("t_inf_media_ms_random_forest", t_inf["random_forest"]["media_ms"]),
        ("t_inf_media_ms_xgboost", t_inf["xgboost"]["media_ms"]),
        ("rama_decision", rama),
        ("modelo_elegido", NOMBRE_BONITO[ganador]),
    ]
    pd.DataFrame(contraste, columns=["concepto", "valor"]).to_csv(
        rutas["tablas"] / "tabla_contraste_decision.csv", index=False)

    diag = diagnostico_por_fuente(test, preds, rutas)
    return {"X_test": X_test, "preds": preds, "probas": probas, "res": res,
            "boot": boot, "mcnemar": mc, "t_inf": t_inf, "ganador": ganador,
            "rama": rama, "diag": diag}


# ===========================================================================
# DEPLOYMENT
# ===========================================================================
def info_entorno() -> dict:
    """Paso 12: hardware y software donde se corrió el experimento (Tabla XVII)."""
    procesador = platform.processor()
    if platform.system() == "Windows":
        try:  # el nombre comercial del CPU está en el registro de Windows
            import winreg
            clave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                   r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            procesador = winreg.QueryValueEx(clave, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    nucleos_fisicos, ram_gb = None, None
    try:
        import psutil
        nucleos_fisicos = psutil.cpu_count(logical=False)
        ram_gb = round(psutil.virtual_memory().total / 1024 ** 3, 2)
    except ImportError:
        if platform.system() == "Windows":  # alternativa sin psutil
            import ctypes

            class Memoria(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = Memoria()
            m.dwLength = ctypes.sizeof(Memoria)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            ram_gb = round(m.ullTotalPhys / 1024 ** 3, 2)
    return {
        "sistema_operativo": f"{platform.system()} {platform.release()} ({platform.version()})",
        "arquitectura": platform.machine(),
        "procesador": procesador,
        "nucleos_fisicos": nucleos_fisicos,
        "nucleos_logicos": os.cpu_count(),
        "ram_gb": ram_gb,
        "python": platform.python_version(),
        "librerias": versiones_librerias(),
    }


def fase_deployment(modelos, ev, test, n_train, n_iter, quick, rutas):
    titulo("DEPLOYMENT - Pasos 9 y 10: serialización y ficha del modelo")
    ganador = ev["ganador"]
    modelo = modelos[ganador].best_estimator_  # ya tiene n_jobs=1 (paso 7)

    # compress=3 reduce bastante el tamaño de los bosques sin perder nada.
    joblib.dump(modelo, rutas["modelo"], compress=3)
    tamano_mb = rutas["modelo"].stat().st_size / 1024 ** 2
    print(f"Modelo serializado: {rutas['modelo']} ({tamano_mb:.2f} MB)")

    m = ev["res"][ganador]
    ficha = {
        "algoritmo": NOMBRE_BONITO[ganador],
        "clase_python": f"{type(modelo).__module__}.{type(modelo).__name__}",
        "hiperparametros_optimos": {k: v for k, v in modelos[ganador].best_params_.items()},
        "feature_names": list(FEATURE_NAMES),
        "n_features": len(FEATURE_NAMES),
        "mapeo_clases": {"0": config.NOMBRE_LEGITIMA, "1": config.NOMBRE_PHISHING},
        "clase_positiva": 1,
        "salida_probabilidad": "predict_proba(X)[:, 1] = probabilidad de phishing",
        "metricas_test": {k: m[k] for k in ["exactitud", "precision", "sensibilidad", "f1",
                                           "TP", "TN", "FP", "FN"]},
        "n_entrenamiento": n_train,
        "n_prueba": len(test),
        "n_iter": n_iter,
        "cv_folds": config.CV_FOLDS,
        "rama_decision": ev["rama"],
        "seed": config.SEED,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "corrida_quick": quick,
        "python": platform.python_version(),
        "versiones_librerias": versiones_librerias(),
        "archivo_modelo": rutas["modelo"].name,
        "tamano_mb": round(tamano_mb, 3),
        "sha256_modelo": sha256_archivo(rutas["modelo"]),
        "dataset": config.DATASET_FEATURES_PATH.name,
        "sha256_dataset": sha256_archivo(config.DATASET_FEATURES_PATH),
    }
    with open(rutas["model_card"], "w", encoding="utf-8") as f:
        json.dump(ficha, f, ensure_ascii=False, indent=2, default=str)
    print(f"Ficha guardada: {rutas['model_card']}")

    # Misma información en formato tabla (campo, valor) para la tesis.
    filas = []
    for campo, valor in ficha.items():
        if isinstance(valor, dict):
            filas += [(f"{campo}.{k}", a_texto(v)) for k, v in valor.items()]
        elif isinstance(valor, list):
            filas.append((campo, ", ".join(valor)))
        else:
            filas.append((campo, a_texto(valor)))
    pd.DataFrame(filas, columns=["campo", "valor"]).to_csv(
        rutas["tablas"] / "tabla_ficha_artefacto.csv", index=False)

    titulo("DEPLOYMENT - Paso 11: verificación del artefacto")
    recargado = joblib.load(rutas["modelo"])
    X_100 = ev["X_test"][:100]
    iguales_pred = np.array_equal(recargado.predict(X_100), ev["preds"][ganador][:100])
    iguales_proba = np.array_equal(recargado.predict_proba(X_100)[:, 1], ev["probas"][ganador][:100])
    print(f"Predicciones idénticas en 100 filas de prueba: {iguales_pred}")
    print(f"Probabilidades idénticas en 100 filas de prueba: {iguales_proba}")
    assert iguales_pred and iguales_proba, "El modelo recargado NO reproduce las predicciones"

    # URL crudas: el mismo camino que seguirá la API (texto -> atributos -> modelo).
    urls = [
        ("legítima raíz", "https://www.google.com/"),
        ("legítima con ruta", "https://es.wikipedia.org/wiki/Phishing"),
        ("legítima con ruta", "https://www.google.com/search?q=phishing"),
        ("legítima con ruta", "https://github.com/scikit-learn/scikit-learn"),
        ("legítima con ruta", "https://docs.python.org/3/library/urllib.parse.html"),
        ("legítima con ruta", "https://stackoverflow.com/questions/tagged/python"),
        ("legítima con ruta", "https://www.youtube.com/watch?v=jNQXAC9IVRw"),
        ("legítima con ruta", "https://www.bbc.com/news/technology"),
        ("legítima con ruta", "https://www.microsoft.com/es-es/windows"),
        ("legítima con ruta", "https://www.amazon.com/gp/help/customer/display.html"),
        ("legítima con ruta", "https://outlook.live.com/mail/0/inbox"),
        ("inventada sospechosa","http://192.168.45.12/secure-login/verify.php?user=admin&acc=1"),
        ("inventada sospechosa", "http://paypal.com.account-verify-security.xyz/login/update@info"),
        ("inventada sospechosa", "http://banco-seguro-actualiza-datos.top//cuenta/confirmar.php?id=8231&token=%2F%3D"),
    ]
    pruebas_urls = []
    for tipo, url in urls:
        atributos = features.extract_features(url)
        vector = np.array([[atributos[n] for n in FEATURE_NAMES]], dtype=np.float64)
        p = float(recargado.predict_proba(vector)[0, 1])
        pruebas_urls.append({"tipo": tipo, "url": url, "proba_phishing": p})
        print(f"  P(phishing) = {p:.4f}  [{tipo}] {url}")

    alerta_tamano = tamano_mb > config.ALERTA_TAMANO_MB
    if alerta_tamano:
        print(f"ADVERTENCIA: el .joblib pesa {tamano_mb:.1f} MB (> {config.ALERTA_TAMANO_MB:.0f} MB). "
              "GitHub rechaza archivos de más de 100 MB.")

    titulo("DEPLOYMENT - Paso 12: entorno de ejecución")
    entorno = info_entorno()
    with open(rutas["environment"], "w", encoding="utf-8") as f:
        json.dump(entorno, f, ensure_ascii=False, indent=2)
    print(json.dumps(entorno, ensure_ascii=False, indent=2))
    return {"tamano_mb": tamano_mb, "alerta_tamano": alerta_tamano, "pruebas_urls": pruebas_urls}


# ===========================================================================
# Resumen final
# ===========================================================================
def resumen(modelos, filas_fold, ev, dep, rutas):
    titulo("RESUMEN")
    print("Archivos generados:")
    for clave in ["train", "test", "predicciones", "modelo", "model_card", "environment"]:
        print(f"  {rutas[clave]}")
    for nombre in ["tabla_hiperparametros.csv", "tabla_f1_por_fold.csv", "tabla_matriz_confusion.csv",
                   "tabla_metricas_test.csv", "tabla_contraste_decision.csv",
                   "tabla_t_inf_laboratorio.csv", "tabla_ficha_artefacto.csv"]:
        print(f"  {rutas['tablas'] / nombre}")
    for nombre in ["diagnostico_por_fuente.csv", "diagnostico_cruzado.csv"]:
        print(f"  {rutas['diagnosticos'] / nombre}")
    print(f"  {rutas['figuras'] / 'figura_matrices_confusion.png'}")

    print("\nMejores hiperparámetros y F1 en validación cruzada:")
    for algoritmo, fila in zip(ALGORITMOS, filas_fold):
        print(f"  {NOMBRE_BONITO[algoritmo]}: {modelos[algoritmo].best_params_}")
        print(f"    F1 medio por fold = {fila['media']:.4f} ± {fila['desv_est_muestral']:.4f}")

    print("\nMétricas en prueba:")
    for algoritmo in ALGORITMOS:
        m = ev["res"][algoritmo]
        print(f"  {NOMBRE_BONITO[algoritmo]:<14} exactitud={m['exactitud']:.4f} "
              f"precisión={m['precision']:.4f} sensibilidad={m['sensibilidad']:.4f} F1={m['f1']:.4f}")

    b, mc = ev["boot"], ev["mcnemar"]
    print(f"\nContraste: ΔF1 (RF-XGB) = {b['delta_f1_observado']:+.5f}, IC 95 % = "
          f"[{b['ic_inf']:+.5f}, {b['ic_sup']:+.5f}] -> "
          f"{'paridad estadística' if b['paridad'] else 'diferencia significativa'}")
    print(f"McNemar: b={mc['b']}, c={mc['c']}, p={mc['p_valor']:.4g} "
          f"({'significativo' if mc['significativo'] else 'no significativo'} con α={config.ALFA})")
    print(f"Decisión: {ev['rama']}")
    print(f"Modelo serializado: {NOMBRE_BONITO[ev['ganador']]} ({dep['tamano_mb']:.2f} MB)")

    print("\nT_inf de laboratorio:")
    for algoritmo in ALGORITMOS:
        t = ev["t_inf"][algoritmo]
        print(f"  {NOMBRE_BONITO[algoritmo]:<14} media={t['media_ms']:.3f} ms, "
              f"mediana={t['mediana_ms']:.3f} ms, p95={t['p95_ms']:.3f} ms "
              f"(umbral {config.T_INF_UMBRAL_MS:.0f} ms)")

    advertencias = []
    for algoritmo in ALGORITMOS:
        if ev["res"][algoritmo]["f1"] > config.ALERTA_F1_ALTO:
            advertencias.append(
                f"F1 de {NOMBRE_BONITO[algoritmo]} en prueba = {ev['res'][algoritmo]['f1']:.4f} "
                f"(> {config.ALERTA_F1_ALTO}). Resultado sospechosamente alto: revisar posibles "
                "atajos del dataset (p. ej., diferencias de formato entre fuentes).")
        columna = f"f1_{SUFIJO[algoritmo]}"
        diag = ev["diag"]
        if len(diag) > 1:
            dif = float(diag[columna].max() - diag[columna].min())
            if dif > config.ALERTA_DIF_FUENTES:
                advertencias.append(
                    f"{NOMBRE_BONITO[algoritmo]}: diferencia de F1 entre fuentes = {dif:.4f} "
                    f"(> {config.ALERTA_DIF_FUENTES}). "
                    + "; ".join(f"{r.source}: {getattr(r, columna):.4f}" for r in diag.itertuples()))
    if dep["alerta_tamano"]:
        advertencias.append(f"El .joblib pesa {dep['tamano_mb']:.1f} MB (> {config.ALERTA_TAMANO_MB:.0f} MB).")
    print("\nAdvertencias:")
    for a in advertencias or ["ninguna"]:
        print(f"  - {a}")


def main():
    # La consola de Windows usa cp1252 por defecto y no sabe escribir "Δ" o "±":
    # forzamos UTF-8 para que los mensajes no rompan el script.
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description="Modeling, Evaluation y Deployment de PhishGuard")
    parser.add_argument("--quick", action="store_true",
                        help=f"prueba rápida: {config.QUICK_N_FILAS:,} filas y n_iter={config.N_ITER_QUICK}")
    parser.add_argument("--n-iter", type=int, default=None,
                        help=f"combinaciones a probar por algoritmo (por defecto {config.N_ITER})")
    args = parser.parse_args()
    n_iter = args.n_iter or (config.N_ITER_QUICK if args.quick else config.N_ITER)

    rutas = rutas_salida(args.quick)
    if args.quick:
        print(f"*** MODO --quick: resultados en {config.QUICK_DIR} (NO son los de la tesis) ***")

    train, test = particionar(args.quick, rutas)
    modelos, filas_fold = fase_modeling(train, n_iter, rutas)
    ev = fase_evaluation(modelos, test, rutas)
    ev["cruzado"] = diagnostico_cruzado(modelos, train, test, rutas)
    dep = fase_deployment(modelos, ev, test, len(train), n_iter, args.quick, rutas)
    resumen(modelos, filas_fold, ev, dep, rutas)


if __name__ == "__main__":
    main()
