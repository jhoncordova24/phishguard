"""Fase CRISP-DM 3: Data Preparation (preparación de los datos).

Qué hace este script (NO entrena modelos y NO parte en train/test; eso va en
train_eval.py):
  1. Consolida PhiUSIIL y Malicious Phish (solo benign y phishing) en una
     sola tabla con columnas url, label (phishing=1, legítima=0) y source.
  2. Normaliza cada URL con features.normalize_url.
  3. Descarta URL nulas o vacías.
  4. Descarta URL malformadas (sin host reconocible) y aplica la política de
     URL sin esquema (config.MISSING_SCHEME_POLICY = "drop" | "keep").
  5. Elimina duplicados exactos. Si una URL aparece como phishing Y como
     legítima, se eliminan TODAS sus copias (no sabemos cuál etiqueta es la buena).
  6. Toma una muestra aleatoria balanceada con la semilla fija, según
     config.SAMPLING_MODE: "class" (25.000 por clase) o "class_x_source"
     (12.500 por cada celda clase x fuente).
  7. Calcula los 20 atributos con features.extract_features (la misma función
     que usará la API).

Después de cada paso se guarda cuántos registros quedan, para la tabla de
descartes de la tesis. Los archivos de ml/data/raw NUNCA se modifican.

Uso (desde la raíz del repositorio):
    python ml/prepare_data.py
"""

import sys
from pathlib import Path

# Permite importar "ml.config" y "features" aunque el script se ejecute suelto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import features
from ml import config

NOMBRE_CLASE = {config.LABEL_LEGITIMA: "legitima", config.LABEL_PHISHING: "phishing"}
FUENTES = [config.SOURCE_PHIUSIIL, config.SOURCE_MALICIOUS]


# ---------------------------------------------------------------------------
# Registro de conteos (para la tabla de descartes)
# ---------------------------------------------------------------------------
class Bitacora:
    """Va anotando, paso a paso, cuántos registros se descartan y cuántos quedan."""

    def __init__(self):
        self.filas = []

    def anotar(self, regla: str, descartados: pd.DataFrame, restantes: pd.DataFrame):
        fila = {"regla": regla}
        for fuente in FUENTES:
            fila[f"descartados_{fuente}"] = int((descartados["source"] == fuente).sum())
        fila["descartados_total"] = len(descartados)
        for fuente in FUENTES:
            fila[f"restantes_{fuente}"] = int((restantes["source"] == fuente).sum())
        for label, nombre in NOMBRE_CLASE.items():
            fila[f"restantes_{nombre}"] = int((restantes["label"] == label).sum())
        fila["restantes_total"] = len(restantes)
        self.filas.append(fila)

        print(f"\n[{regla}]")
        print(f"  descartados: {len(descartados):>9,}  "
              + "  ".join(f"{f}={fila[f'descartados_{f}']:,}" for f in FUENTES))
        print(f"  restantes:   {len(restantes):>9,}  "
              + "  ".join(f"{f}={fila[f'restantes_{f}']:,}" for f in FUENTES)
              + f"  | legitima={fila['restantes_legitima']:,}"
              + f"  phishing={fila['restantes_phishing']:,}")

    def tabla(self) -> pd.DataFrame:
        return pd.DataFrame(self.filas)


# ---------------------------------------------------------------------------
# Paso 1: consolidar las dos fuentes
# ---------------------------------------------------------------------------
def leer_csv(ruta: Path, columnas: list, encoding: str = "utf-8") -> pd.DataFrame:
    # keep_default_na=False + na_values=[""]: solo una celda VACÍA cuenta como
    # nulo. Así una URL que literalmente diga "null" o "nan" no se pierde.
    return pd.read_csv(ruta, usecols=columnas, encoding=encoding, dtype=str,
                       keep_default_na=False, na_values=[""])


def consolidar(bitacora: Bitacora) -> pd.DataFrame:
    # --- PhiUSIIL: label 1 = LEGÍTIMA, 0 = PHISHING (al revés que nosotros) ---
    phi = leer_csv(config.PHIUSIIL_PATH, config.PHIUSIIL_COLS, config.PHIUSIIL_ENCODING)
    mapa_phi = {str(config.PHIUSIIL_LABEL_PHISHING): config.LABEL_PHISHING,
                str(config.PHIUSIIL_LABEL_LEGITIMA): config.LABEL_LEGITIMA}
    phi = pd.DataFrame({
        "url_original": phi["URL"],
        "label": phi["label"].map(mapa_phi),  # etiqueta rara -> NaN
        "source": config.SOURCE_PHIUSIIL,
    })

    # --- Malicious Phish: nos quedamos solo con benign y phishing ---
    mal = leer_csv(config.MALICIOUS_PATH, config.MALICIOUS_COLS)
    mapa_mal = {"phishing": config.LABEL_PHISHING, "benign": config.LABEL_LEGITIMA}
    mal = pd.DataFrame({
        "url_original": mal["url"],
        "label": mal["type"].map(mapa_mal),  # defacement / malware -> NaN
        "source": config.SOURCE_MALICIOUS,
    })

    todo = pd.concat([phi, mal], ignore_index=True)
    fuera = todo["label"].isna()  # defacement, malware o etiquetas inesperadas
    bitacora.anotar("1. Consolidar fuentes (fuera de estudio: defacement, malware, "
                    "etiquetas no reconocidas)", todo[fuera], todo[~fuera])
    todo = todo[~fuera].copy()
    todo["label"] = todo["label"].astype(int)
    return todo.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pasos 2 a 4
# ---------------------------------------------------------------------------
def normalizar(df: pd.DataFrame, bitacora: Bitacora) -> pd.DataFrame:
    # Guardamos también la URL original: extract_features la necesita para
    # contar las secuencias %xx (que desaparecen al normalizar).
    df["url"] = df["url_original"].map(features.normalize_url)
    cambiadas = (df["url"] != df["url_original"].fillna("")).groupby(df["source"]).sum()
    print("\n  URL que cambiaron al normalizar: "
          + "  ".join(f"{f}={int(cambiadas.get(f, 0)):,}" for f in FUENTES))
    bitacora.anotar("2. Normalizar (minúsculas, %xx, espacios)", df.iloc[0:0], df)
    return df


def descartar_vacias(df: pd.DataFrame, bitacora: Bitacora) -> pd.DataFrame:
    vacia = df["url"] == ""  # normalize_url devuelve "" para nulos y solo-espacios
    bitacora.anotar("3. Descartar nulos o vacíos", df[vacia], df[~vacia])
    return df[~vacia].copy()


def _host_valido(url: str) -> bool:
    try:
        features.split_url(url)
        return True
    except features.URLInvalidaError:
        return False


def desglose_sin_esquema(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla fuente x clase: cuántas URL no tienen esquema y cuántas se descartarían."""
    filas = []
    for (fuente, label), g in df.groupby(["source", "label"]):
        sin = ~g["tiene_esquema"]
        filas.append({
            "source": fuente,
            "clase": NOMBRE_CLASE[label],
            "registros_bien_formados": int(g["host_valido"].sum()),
            "sin_esquema_total": int(sin.sum()),
            # Las malformadas ya caen en la regla anterior; las que quedan son
            # las que "drop" eliminaría adicionalmente.
            "sin_esquema_descartadas_si_drop": int((sin & g["host_valido"]).sum()),
        })
    tabla = pd.DataFrame(filas)
    tabla["pct_descartadas_si_drop"] = (
        100 * tabla["sin_esquema_descartadas_si_drop"] / tabla["registros_bien_formados"]
    ).round(2)
    tabla["politica_aplicada"] = config.MISSING_SCHEME_POLICY
    return tabla


def descartar_malformadas(df: pd.DataFrame, bitacora: Bitacora) -> pd.DataFrame:
    if config.MISSING_SCHEME_POLICY not in ("drop", "keep"):
        sys.exit(f"MISSING_SCHEME_POLICY debe ser 'drop' o 'keep', no "
                 f"{config.MISSING_SCHEME_POLICY!r}")

    df["host_valido"] = df["url"].map(_host_valido)
    df["tiene_esquema"] = df["url"].map(features.has_scheme)

    # 4a. Sin host reconocible
    mala = ~df["host_valido"]
    bitacora.anotar("4a. Descartar malformadas (sin host reconocible)", df[mala], df[~mala])

    # Desglose de URL sin esquema (se imprime y guarda SIEMPRE, para decidir después)
    tabla = desglose_sin_esquema(df)
    tabla.to_csv(config.TABLA_SIN_ESQUEMA_PATH, index=False, encoding="utf-8-sig")
    print("\n  Desglose de URL sin esquema (fuente x clase):")
    print("  " + tabla.to_string(index=False).replace("\n", "\n  "))

    df = df[~mala]

    # 4b. Política de URL sin esquema
    if config.MISSING_SCHEME_POLICY == "drop":
        sin = ~df["tiene_esquema"]
    else:
        sin = pd.Series(False, index=df.index)
    bitacora.anotar(f"4b. URL sin esquema (política = {config.MISSING_SCHEME_POLICY})",
                    df[sin], df[~sin])
    return df[~sin].drop(columns=["host_valido"]).copy()


# ---------------------------------------------------------------------------
# Paso 5: duplicados y etiquetas contradictorias
# ---------------------------------------------------------------------------
def deduplicar(df: pd.DataFrame, bitacora: Bitacora) -> pd.DataFrame:
    # Informativo: URL que aparecen en las DOS fuentes.
    fuentes_por_url = df.groupby("url")["source"].nunique()
    print(f"\n  URL distintas presentes en ambas fuentes: "
          f"{int((fuentes_por_url > 1).sum()):,}")

    # 5a. Una misma URL con etiqueta phishing y legítima: se eliminan todas sus copias.
    etiquetas_por_url = df.groupby("url")["label"].nunique()
    urls_conflicto = etiquetas_por_url[etiquetas_por_url > 1].index
    conflicto = df["url"].isin(urls_conflicto)
    print(f"  URL distintas con etiquetas contradictorias: {len(urls_conflicto):,} "
          f"({int(conflicto.sum()):,} registros)")
    bitacora.anotar("5a. Etiquetas contradictorias (se eliminan todas las copias)",
                    df[conflicto], df[~conflicto])
    df = df[~conflicto]

    # 5b. Duplicados exactos: se conserva la primera aparición. Como PhiUSIIL
    # se cargó primero, si una URL está en ambas fuentes se queda como phiusiil.
    dup = df.duplicated(subset="url", keep="first")
    bitacora.anotar("5b. Duplicados exactos (se conserva la primera aparición)",
                    df[dup], df[~dup])
    return df[~dup].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Paso 6: muestra balanceada
# ---------------------------------------------------------------------------
def muestrear(df: pd.DataFrame) -> pd.DataFrame:
    if config.SAMPLING_MODE == "class":
        partes = _muestrear_por_clase(df)
    elif config.SAMPLING_MODE == "class_x_source":
        partes = _muestrear_por_clase_y_fuente(df)
    else:
        sys.exit(f"SAMPLING_MODE debe ser 'class' o 'class_x_source', no "
                 f"{config.SAMPLING_MODE!r}")
    # Mezclamos las filas (con la misma semilla) para que no queden en bloques por clase.
    muestra = pd.concat(partes).sample(frac=1, random_state=config.SEED)
    return muestra.reset_index(drop=True)


def _muestrear_por_clase(df: pd.DataFrame) -> list:
    """Modo "class": 25.000 por clase, sin controlar la fuente (iteración 1)."""
    n = config.MUESTRA_POR_CLASE
    disponibles = df["label"].value_counts()
    faltan = {NOMBRE_CLASE[c]: int(disponibles.get(c, 0))
              for c in NOMBRE_CLASE if disponibles.get(c, 0) < n}
    if faltan:
        sys.exit(f"\nALTO: no hay suficientes URL para muestrear {n:,} por clase. "
                 f"Disponibles: {faltan}. Revisa MISSING_SCHEME_POLICY o el tamaño "
                 f"de muestra en ml/config.py.")
    return [df[df["label"] == c].sample(n=n, replace=False, random_state=config.SEED)
            for c in NOMBRE_CLASE]


def _muestrear_por_clase_y_fuente(df: pd.DataFrame) -> list:
    """Modo "class_x_source": 4 celdas clase x fuente del mismo tamaño.

    Si alguna celda no alcanza, el script se DETIENE (no se completa con otra
    fuente ni se muestrea con reemplazo).
    """
    n = config.MUESTRA_POR_CELDA
    disponibles = df.groupby(["label", "source"]).size()
    print(f"\n  Disponibles por celda (se piden {n:,} en cada una):")
    faltan = {}
    for c in NOMBRE_CLASE:
        for fuente in FUENTES:
            k = int(disponibles.get((c, fuente), 0))
            print(f"    {NOMBRE_CLASE[c]:<9} x {fuente:<16} {k:>9,}"
                  + ("   <- NO ALCANZA" if k < n else ""))
            if k < n:
                faltan[f"{NOMBRE_CLASE[c]} x {fuente}"] = k
    if faltan:
        sys.exit(f"\nALTO: no hay suficientes URL para muestrear {n:,} por celda "
                 f"clase x fuente. Disponibles: {faltan}.")
    return [df[(df["label"] == c) & (df["source"] == fuente)]
            .sample(n=n, replace=False, random_state=config.SEED)
            for c in NOMBRE_CLASE for fuente in FUENTES]


def tabla_muestra(muestra: pd.DataFrame, limpio: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for label, nombre in NOMBRE_CLASE.items():
        m = muestra[muestra["label"] == label]
        pool = limpio[limpio["label"] == label]
        fila = {"clase": nombre, "disponibles_tras_depuracion": len(pool),
                "en_muestra": len(m)}
        for fuente in FUENTES:
            k = int((m["source"] == fuente).sum())
            fila[f"disponibles_{fuente}"] = int((pool["source"] == fuente).sum())
            fila[f"muestra_{fuente}"] = k
            fila[f"pct_muestra_{fuente}"] = round(100 * k / len(m), 2)
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    total = {"clase": "total"}
    for col in tabla.columns[1:]:
        if not col.startswith("pct_"):
            total[col] = int(tabla[col].sum())
    for fuente in FUENTES:
        total[f"pct_muestra_{fuente}"] = round(
            100 * total[f"muestra_{fuente}"] / total["en_muestra"], 2)
    return pd.concat([tabla, pd.DataFrame([total])], ignore_index=True)


# ---------------------------------------------------------------------------
# Paso 7: atributos
# ---------------------------------------------------------------------------
def extraer_atributos(muestra: pd.DataFrame) -> pd.DataFrame:
    # Se le pasa la URL ORIGINAL: extract_features normaliza por dentro, igual
    # que hará la API cuando reciba una URL nueva.
    filas, errores = [], 0
    for url in muestra["url_original"]:
        try:
            filas.append(features.extract_features(url))
        except features.URLInvalidaError:
            errores += 1
            filas.append({nombre: None for nombre in features.FEATURE_NAMES})
    if errores:
        print(f"\n  AVISO: {errores} URL fallaron en extract_features (quedan con NaN)")
    atributos = pd.DataFrame(filas, columns=features.FEATURE_NAMES)
    base = muestra[["url", "label", "source", "url_original"]].reset_index(drop=True)
    return pd.concat([base, atributos], axis=1)


def tabla_descriptivos(dataset: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for nombre in features.FEATURE_NAMES:
        fila = {"atributo": nombre}
        for label, clase in NOMBRE_CLASE.items():
            valores = dataset.loc[dataset["label"] == label, nombre]
            fila[f"media_{clase}"] = round(float(valores.mean()), 4)
            fila[f"mediana_{clase}"] = round(float(valores.median()), 4)
        filas.append(fila)
    return pd.DataFrame(filas)


def revisar_rarezas(dataset: pd.DataFrame) -> None:
    """Imprime atributos con nulos o casi constantes (>= 95 % del mismo valor)."""
    print("\n=== Revisión de atributos ===")
    nada_raro = True
    for nombre in features.FEATURE_NAMES:
        col = dataset[nombre]
        nulos = int(col.isna().sum())
        if nulos:
            nada_raro = False
            print(f"  {nombre}: {nulos:,} valores nulos")
        moda_global = col.value_counts(normalize=True, dropna=False)
        if moda_global.iloc[0] >= 0.95:
            nada_raro = False
            print(f"  {nombre}: casi constante en toda la muestra "
                  f"({100 * moda_global.iloc[0]:.2f} % vale {moda_global.index[0]})")
        for label, clase in NOMBRE_CLASE.items():
            por_clase = col[dataset["label"] == label].value_counts(normalize=True)
            if por_clase.iloc[0] >= 0.99 and moda_global.iloc[0] < 0.95:
                nada_raro = False
                print(f"  {nombre}: casi constante dentro de '{clase}' "
                      f"({100 * por_clase.iloc[0]:.2f} % vale {por_clase.index[0]})")
    if nada_raro:
        print("  Sin nulos ni atributos casi constantes.")


# ---------------------------------------------------------------------------
def main() -> None:
    for ruta in (config.PHIUSIIL_PATH, config.MALICIOUS_PATH):
        if not ruta.exists():
            sys.exit(f"No se encontró el archivo: {ruta}")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.DIAGNOSTICOS_DIR.mkdir(parents=True, exist_ok=True)

    tamano = (f"muestra por clase: {config.MUESTRA_POR_CLASE:,}"
              if config.SAMPLING_MODE == "class"
              else f"muestra por celda clase x fuente: {config.MUESTRA_POR_CELDA:,}")
    print(f"Política de URL sin esquema: {config.MISSING_SCHEME_POLICY!r}  |  "
          f"semilla: {config.SEED}  |  muestreo: {config.SAMPLING_MODE!r} ({tamano})")

    bitacora = Bitacora()
    df = consolidar(bitacora)
    df = normalizar(df, bitacora)
    df = descartar_vacias(df, bitacora)
    df = descartar_malformadas(df, bitacora)
    limpio = deduplicar(df, bitacora)

    # Se guardan url, label, source y además url_original (necesaria para
    # recalcular num_obfuscated_chars, que se mide antes de decodificar).
    columnas = ["url", "label", "source", "url_original"]
    limpio[columnas].to_csv(config.URLS_CLEAN_PATH, index=False, encoding="utf-8")
    descartes = bitacora.tabla()
    descartes.to_csv(config.TABLA_DESCARTES_PATH, index=False, encoding="utf-8-sig")

    muestra = muestrear(limpio)
    t_muestra = tabla_muestra(muestra, limpio)
    t_muestra.to_csv(config.TABLA_MUESTRA_PATH, index=False, encoding="utf-8-sig")
    print("\n=== Muestra balanceada (mezcla por fuente) ===")
    print(t_muestra.to_string(index=False))

    dataset = extraer_atributos(muestra)
    dataset.to_csv(config.DATASET_FEATURES_PATH, index=False, encoding="utf-8")
    descriptivos = tabla_descriptivos(dataset)
    descriptivos.to_csv(config.TABLA_DESCRIPTIVOS_PATH, index=False, encoding="utf-8-sig")
    print("\n=== Descriptivos por clase ===")
    print(descriptivos.to_string(index=False))

    revisar_rarezas(dataset)

    print("\nArchivos generados:")
    for ruta in (config.URLS_CLEAN_PATH, config.DATASET_FEATURES_PATH,
                 config.TABLA_DESCARTES_PATH, config.TABLA_SIN_ESQUEMA_PATH,
                 config.TABLA_MUESTRA_PATH, config.TABLA_DESCRIPTIVOS_PATH):
        print(f"  {ruta.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
