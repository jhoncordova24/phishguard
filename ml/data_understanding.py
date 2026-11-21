"""Fase CRISP-DM 2: Data Understanding (comprensión de los datos).

Qué hace este script (NO entrena ningún modelo):
  1. Lee los dos repositorios originales cargando solo las columnas necesarias.
  2. Cuenta cuántas URL legítimas y de phishing hay en cada fuente.
  3. Calcula el consolidado de ambas fuentes (todavía SIN quitar duplicados).
  4. Guarda una tabla CSV y una figura PNG para la tesis.
  5. Compara los conteos obtenidos con las cifras esperadas de la tesis
     y muestra coincidencias y diferencias (sin tocar los datos).

Uso (desde la raíz del repositorio):
    python ml/data_understanding.py
"""

import sys
from pathlib import Path

# Permite importar "ml.config" aunque el script se ejecute como archivo suelto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: solo guardamos la imagen en disco
import matplotlib.pyplot as plt
import pandas as pd

from ml import config

# Cifras esperadas según el documento de tesis (solo para comparar, NO se usan
# para modificar los datos).
VALORES_TESIS = {
    "PhiUSIIL - total registros": 235_795,
    "PhiUSIIL - legítimas": 134_850,
    "PhiUSIIL - phishing": 100_945,
    "Malicious Phish - total registros": 651_191,
    "Malicious Phish - benign": 428_103,
    "Malicious Phish - defacement": 96_457,
    "Malicious Phish - phishing": 94_111,
    "Malicious Phish - malware": 32_520,
    "Malicious Phish - subtotal benign+phishing": 522_214,
    "Consolidado - total": 758_009,
    "Consolidado - legítimas": 562_953,
    "Consolidado - phishing": 195_056,
}


# ---------------------------------------------------------------------------
# 1. Lectura de datos
# ---------------------------------------------------------------------------
def cargar_phiusiil() -> pd.DataFrame:
    """Lee PhiUSIIL con solo las columnas URL y label."""
    df = pd.read_csv(
        config.PHIUSIIL_PATH,
        encoding=config.PHIUSIIL_ENCODING,  # quita el BOM del encabezado
        usecols=config.PHIUSIIL_COLS,       # no cargamos las otras ~50 columnas
    )
    return df


def cargar_malicious() -> pd.DataFrame:
    """Lee Malicious Phish con solo las columnas url y type."""
    df = pd.read_csv(
        config.MALICIOUS_PATH,
        usecols=config.MALICIOUS_COLS,
        dtype={"type": "category"},  # 'category' ahorra memoria en texto repetido
    )
    return df


# ---------------------------------------------------------------------------
# 2. Conteos por fuente
# ---------------------------------------------------------------------------
def resumen_phiusiil(df: pd.DataFrame) -> dict:
    """Cuenta legítimas y phishing en PhiUSIIL.

    Recordatorio: en PhiUSIIL label=1 es LEGÍTIMA y label=0 es PHISHING.
    """
    conteo = df["label"].value_counts(dropna=False)
    legitimas = int(conteo.get(config.PHIUSIIL_LABEL_LEGITIMA, 0))
    phishing = int(conteo.get(config.PHIUSIIL_LABEL_PHISHING, 0))
    otros = len(df) - legitimas - phishing  # etiquetas inesperadas o vacías

    print("\n=== PhiUSIIL ===")
    print(f"Registros totales:              {len(df):>10,}")
    print(f"label=1 -> legítima:            {legitimas:>10,}")
    print(f"label=0 -> phishing:            {phishing:>10,}")
    print(f"Etiquetas fuera de {{0,1}}/nulas: {otros:>10,}")
    print(f"URL nulas:                      {int(df['URL'].isna().sum()):>10,}")

    return {"total_registros": len(df), "legitimas": legitimas,
            "phishing": phishing, "otros": otros}


def resumen_malicious(df: pd.DataFrame) -> dict:
    """Cuenta las 4 clases de Malicious Phish y el subtotal benign+phishing."""
    conteo = df["type"].value_counts(dropna=False)
    por_clase = {c: int(conteo.get(c, 0)) for c in config.MALICIOUS_CLASES}
    otros = len(df) - sum(por_clase.values())  # clases inesperadas o vacías

    print("\n=== Malicious Phish ===")
    print(f"Registros totales:              {len(df):>10,}")
    for clase, n in por_clase.items():
        print(f"  {clase:<29} {n:>10,}")
    print(f"Clases inesperadas/nulas:       {otros:>10,}")
    print(f"URL nulas:                      {int(df['url'].isna().sum()):>10,}")

    # Nos quedamos solo con benign (= legítima) y phishing.
    legitimas = por_clase["benign"]
    phishing = por_clase["phishing"]
    print(f"Subtotal benign + phishing:     {legitimas + phishing:>10,}")

    return {"total_registros": len(df), "por_clase": por_clase,
            "legitimas": legitimas, "phishing": phishing, "otros": otros}


# ---------------------------------------------------------------------------
# 3. Tabla de composición
# ---------------------------------------------------------------------------
def fila(fuente: str, legitimas: int, phishing: int) -> dict:
    """Arma una fila de la tabla con cantidades y proporciones (%)."""
    total = legitimas + phishing
    return {
        "fuente": fuente,
        "url_legitimas": legitimas,
        "url_phishing": phishing,
        "total": total,
        "pct_legitimas": round(100 * legitimas / total, 2),
        "pct_phishing": round(100 * phishing / total, 2),
    }


def construir_tabla(phi: dict, mal: dict) -> pd.DataFrame:
    filas = [
        fila("PhiUSIIL", phi["legitimas"], phi["phishing"]),
        fila("Malicious Phish (benign + phishing)", mal["legitimas"], mal["phishing"]),
        # Consolidado = simple suma de ambas fuentes, sin eliminar duplicados aún.
        fila("Consolidado (sin deduplicar)",
             phi["legitimas"] + mal["legitimas"],
             phi["phishing"] + mal["phishing"]),
    ]
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 4. Figura
# ---------------------------------------------------------------------------
def graficar(tabla: pd.DataFrame, ruta: Path) -> None:
    """Barras agrupadas: por cada fuente, una barra de legítimas y otra de phishing."""
    etiquetas = ["PhiUSIIL", "Malicious Phish\n(benign + phishing)",
                 "Consolidado\n(sin deduplicar)"]
    x = range(len(tabla))
    ancho = 0.38
    # Colores distinguibles también en escala de grises / daltonismo.
    color_leg, color_phi = "#2B6CB0", "#DD6B20"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    barras_leg = ax.bar([i - ancho / 2 for i in x], tabla["url_legitimas"],
                        ancho, label="Legítima", color=color_leg)
    barras_phi = ax.bar([i + ancho / 2 for i in x], tabla["url_phishing"],
                        ancho, label="Phishing", color=color_phi, hatch="//",
                        edgecolor="white")

    # Etiqueta con la cantidad encima de cada barra (separador de miles).
    for barras in (barras_leg, barras_phi):
        ax.bar_label(barras, labels=[f"{int(v):,}" for v in barras.datavalues],
                     padding=3, fontsize=10)

    ax.set_title("Distribución de clases por repositorio de URL", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("Repositorio de origen", fontsize=12)
    ax.set_ylabel("Cantidad de URL", fontsize=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(etiquetas, fontsize=11)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_ylim(0, tabla[["url_legitimas", "url_phishing"]].max().max() * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(title="Clase", fontsize=11, title_fontsize=11, frameon=False)

    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Comparación con la tesis
# ---------------------------------------------------------------------------
def comparar_con_tesis(phi: dict, mal: dict, tabla: pd.DataFrame) -> pd.DataFrame:
    consolidado = tabla.iloc[-1]
    obtenidos = {
        "PhiUSIIL - total registros": phi["total_registros"],
        "PhiUSIIL - legítimas": phi["legitimas"],
        "PhiUSIIL - phishing": phi["phishing"],
        "Malicious Phish - total registros": mal["total_registros"],
        "Malicious Phish - benign": mal["por_clase"]["benign"],
        "Malicious Phish - defacement": mal["por_clase"]["defacement"],
        "Malicious Phish - phishing": mal["por_clase"]["phishing"],
        "Malicious Phish - malware": mal["por_clase"]["malware"],
        "Malicious Phish - subtotal benign+phishing": mal["legitimas"] + mal["phishing"],
        "Consolidado - total": int(consolidado["total"]),
        "Consolidado - legítimas": int(consolidado["url_legitimas"]),
        "Consolidado - phishing": int(consolidado["url_phishing"]),
    }
    comp = pd.DataFrame({
        "concepto": list(VALORES_TESIS),
        "esperado_tesis": list(VALORES_TESIS.values()),
        "obtenido": [obtenidos[k] for k in VALORES_TESIS],
    })
    comp["diferencia"] = comp["obtenido"] - comp["esperado_tesis"]
    comp["coincide"] = comp["diferencia"].eq(0).map({True: "SÍ", False: "NO"})
    return comp


def main() -> None:
    # Crear carpetas de salida si no existen.
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for ruta in (config.PHIUSIIL_PATH, config.MALICIOUS_PATH):
        if not ruta.exists():
            sys.exit(f"No se encontró el archivo: {ruta}")

    phi = resumen_phiusiil(cargar_phiusiil())
    mal = resumen_malicious(cargar_malicious())

    tabla = construir_tabla(phi, mal)
    ruta_tabla = config.TABLES_DIR / "tabla_composicion_repositorios.csv"
    tabla.to_csv(ruta_tabla, index=False, encoding="utf-8-sig")  # BOM: Excel lee bien tildes
    print("\n=== Composición por repositorio ===")
    print(tabla.to_string(index=False))

    ruta_figura = config.FIGURES_DIR / "figura_distribucion_clases.png"
    graficar(tabla, ruta_figura)

    comp = comparar_con_tesis(phi, mal, tabla)
    print("\n=== Comparación con valores de la tesis ===")
    print(comp.to_string(index=False, formatters={
        c: "{:,}".format for c in ["esperado_tesis", "obtenido", "diferencia"]}))
    n_ok = (comp["coincide"] == "SÍ").sum()
    print(f"\nCoincidencias: {n_ok} de {len(comp)}")

    print(f"\nTabla guardada en:  {ruta_tabla.relative_to(config.ROOT_DIR)}")
    print(f"Figura guardada en: {ruta_figura.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
