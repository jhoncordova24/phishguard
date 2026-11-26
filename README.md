# PhishGuard — Sistema open source basado en Machine Learning para la reducción de la tasa de incidentes por phishing

Monorepo del proyecto **PhishGuard**: modelo de Machine Learning, API de inferencia, extensión de
navegador, dashboard y análisis de evaluación. Este documento es la **guía técnica de traspaso**
para el equipo de tesis (y sus agentes de programación).

## 1. Estado del proyecto

| Objetivo | Estado |
|---|---|
| **OE1** — Pipeline CRISP-DM y modelo entrenado | ✅ **Completado y validado** (Iteración 2, modelo XGBoost) |
| **OE2** — API, extensión de Chrome y dashboard | ⏳ Pendiente (ver §7) |

Resultados oficiales del OE1 (conjunto de prueba de 10.000 URL, 50/50):

| Modelo | F1 (CV 5 folds) | F1 prueba | Precisión | Sensibilidad | T_inf medio |
|---|---|---|---|---|---|
| Random Forest | 0,9724 ± 0,0010 | 0,9723 | 0,9842 | 0,9606 | 12,42 ms |
| **XGBoost (elegido)** | 0,9735 ± 0,0014 | **0,9728** | 0,9872 | 0,9588 | **0,48 ms** |

Hubo paridad estadística: el IC 95 % bootstrap de ΔF1 fue [−0,0020; +0,0009] y McNemar dio p = 0,50.
Por la regla de decisión 4.9.2.e, el desempate fue el menor T_inf, y por eso se eligió **XGBoost**.
La Iteración 1 (F1 ≈ 0,993) se descartó por un atajo del dataset y está archivada en
`ml/archive/iteracion_1/`. Detalle de tablas y figuras: [`ml/reports/README.md`](ml/reports/README.md).

## 2. Estructura del repositorio

```
phishguard/
├── features.py            # Extracción de los 20 atributos de una URL (compartido por ml/ y api/)
├── requirements.txt       # Dependencias con versiones fijas (Python 3.12)
├── tests/
│   └── test_features.py   # Pruebas de features.py con valores calculados a mano
├── ml/                    # OE1: pipeline CRISP-DM
│   ├── config.py          # Rutas, semilla (42), hiperparámetros, modo de muestreo
│   ├── data_understanding.py  # Fase 2: comprensión de los datos
│   ├── prepare_data.py    # Fase 3: depuración, muestreo 4 × 12.500 y extracción de atributos
│   ├── train_eval.py      # Fases 4-6: modelado, evaluación, contraste y serialización
│   ├── data/raw/          # Datasets originales (no versionados)
│   ├── data/processed/    # Datasets generados (no versionados)
│   ├── models/            # best_model.joblib + model_card.json  ← artefacto para la API
│   ├── reports/           # Tablas y figuras oficiales del Capítulo 5.1 (ver su README)
│   └── archive/           # Corridas anteriores conservadas como evidencia (iteracion_1/)
├── api/                   # OE2: API REST FastAPI (pendiente; solo esqueleto)
├── extension/             # OE2: extensión de Chrome (pendiente)
├── dashboard/             # OE2: panel de monitoreo (pendiente)
└── analysis/              # Análisis posteriores (latencia/recursos, pretest/postest)
```

- **`features.py`**: es la pieza clave de la integración. Solo usa la biblioteca estándar y calcula
  todo a partir del texto de la URL (sin DNS, WHOIS ni descarga de la página). Si la URL no tiene
  host reconocible, lanza `URLInvalidaError`.
- **`ml/`**: el flujo reproducible es `prepare_data.py` → `train_eval.py`. Toda la configuración
  vive en `ml/config.py`.
- **`tests/`**: si se modifica `features.py`, estas pruebas tienen que seguir pasando. Cualquier
  cambio en los atributos obliga a reentrenar el modelo.

## 3. Instalación y ejecución

Requisito: Python 3.12.

```bash
# 1. Crear el entorno virtual (solo la primera vez)
python -m venv .venv

# 2. Activarlo
.venv\Scripts\Activate.ps1          # Windows (PowerShell)
source .venv/Scripts/activate       # Windows (Git Bash)
source .venv/bin/activate           # Linux / macOS

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Pruebas unitarias
python -m pytest tests -v
```

### Datos

Los datasets originales no se versionan. Hay que colocarlos en `ml/data/raw/`:

- `PhiUSIIL_Phishing_URL_Dataset.csv`: ojo, en este archivo `label = 1` es **legítima** y `label = 0` es **phishing**
  (`prepare_data.py` lo invierte a la convención del proyecto: phishing = 1).
- `malicious_phish.csv`: columnas `url` y `type`; solo se usan `benign` y `phishing`.

### Pipeline CRISP-DM (desde la raíz del repositorio)

```bash
python ml/data_understanding.py   # Fase 2: composición de los repositorios
python ml/prepare_data.py         # Fase 3: genera ml/data/processed/dataset_50k_features.csv
python ml/train_eval.py           # Fases 4-6: corrida oficial (n_iter = 30, 5 folds)
python ml/train_eval.py --quick   # Prueba rápida del flujo (2.000 filas, n_iter = 3)
```

`--quick` escribe en `ml/_quick_run/` (ignorado por git) y **nunca** sobrescribe los resultados
oficiales. La corrida oficial sí sobrescribe `ml/models/` y `ml/reports/`.

## 4. Contrato de integración para la API (`api/`)

| Elemento | Valor |
|---|---|
| Artefacto | `ml/models/best_model.joblib` (XGBoost, `xgboost.sklearn.XGBClassifier`, 0,572 MB) |
| Ficha | `ml/models/model_card.json` (hiperparámetros, métricas, versiones, SHA-256 del modelo) |
| Entrada | Vector de **20 atributos** en el orden de `features.FEATURE_NAMES` |
| Salida | `predict_proba(X)[:, 1]` = probabilidad de **phishing** |
| Clases | `0 = legítima`, `1 = phishing` |
| T_inf de laboratorio | media 0,48 ms, p95 0,56 ms por URL (solo cómputo del modelo) |

Ejemplo mínimo:

```python
import joblib
import numpy as np

import features  # features.py en la raíz del repositorio

modelo = joblib.load("ml/models/best_model.joblib")  # cargarlo UNA vez al iniciar la API

def analizar(url: str) -> dict:
    try:
        atributos = features.extract_features(url)  # normaliza internamente
    except features.URLInvalidaError as error:
        raise ValueError(str(error))                # en FastAPI: responder 400/422
    X = np.array([[atributos[n] for n in features.FEATURE_NAMES]], dtype=np.float64)
    p = float(modelo.predict_proba(X)[0, 1])
    return {"proba_phishing": p, "clase": "phishing" if p >= 0.5 else "legítima"}
```

Reglas del contrato:

- **Llamar siempre a `features.extract_features(url)` con la URL cruda.** Esa función ya aplica la
  misma canonicalización que el entrenamiento: quita espacios, decodifica `%xx` una sola vez, pasa
  todo a minúsculas y quita la barra final cuando la ruta es solo `/`
  (`https://www.google.com/` → `https://www.google.com`). No hay que normalizar la URL antes ni
  después de llamarla.
- **No reordenar los atributos.** El modelo no conoce nombres de columnas, solo posiciones.
- **Verificar las versiones de las librerías** contra `model_card.json` (sobre todo `xgboost`
  y `scikit-learn`) para evitar incompatibilidades al cargar el `.joblib`.
- El umbral 0,5 es el que se usó en la evaluación. Si se cambia, hay que documentarlo aparte.

## 5. Nota técnica importante para producción

**Hallazgo de generalización:** el modelo clasifica como phishing, con P ≈ 0,99, URLs
legítimas reales **profundas**, es decir con `https://` y una ruta
(p. ej. `https://es.wikipedia.org/wiki/Phishing`, `https://github.com/...`).
La página raíz sí sale bien (`https://www.google.com/` → P = 0,03).

**Causa:** la composición de los dos datasets. En los datos de entrenamiento casi no hay
URLs legítimas con https y ruta:

- Las legítimas de **PhiUSIIL** son solo dominios raíz (`https://www.dominio.tld`): 0 de 9.965 tienen ruta.
- Las legítimas de **Malicious Phish** casi nunca usan https: solo 549 de 10.035 combinan https y ruta.
- En cambio, 4.360 URLs de phishing tienen https y ruta.

El diagnóstico cruzado lo confirma: entrenar con una fuente y probar en la otra da F1 ≈ 0,66,
igual que marcar todo como phishing (`ml/reports/tables/diagnosticos/diagnostico_cruzado.csv`).
El F1 de prueba (0,973) es válido **dentro de la distribución de estos datasets**, pero no
garantiza el mismo comportamiento con tráfico real.

**Recomendación:** en FastAPI (`api/`), poner **antes** del modelo una capa de **lista blanca de
dominios de alta reputación verificados**, sin modificar `best_model.joblib`:

1. Extraer el host con `features.split_url(features.normalize_url(url)).host`.
2. Si el host **es** un dominio de la lista o un **subdominio real** de uno de ellos
   (`host == d` o `host.endswith("." + d)`), responder "legítima" e indicar en la respuesta
   que la decisión vino de la lista blanca.
3. Si no, consultar el modelo.

Precauciones:

- **Nunca** usar coincidencias por subcadena: `paypal.com.account-verify-security.xyz`
  contiene `paypal.com`, pero **no** es PayPal.
- No incluir en la lista dominios donde cualquiera puede publicar contenido
  (p. ej. `firebaseapp.com`, `github.io`, acortadores de URL).
- Versionar la lista y registrar en el dashboard cuántas decisiones toma cada capa.
- La lista blanca **no corrige** el sesgo para sitios legítimos que no estén en ella. Corregirlo
  de verdad requiere una nueva iteración de CRISP-DM con una fuente de URLs legítimas profundas
  con https.

## 6. Reproducibilidad

- Semilla fija `SEED = 42` en todo el pipeline (`ml/config.py`).
- `ml/reports/environment.json` registra el hardware y el software de la corrida oficial.
- `model_card.json` incluye el SHA-256 del modelo y del dataset de atributos.

## 7. Próximos pasos — Objetivo Específico 2 (tablero Kanban)

**API — FastAPI (`api/`)**
- [ ] Cargar el modelo y `model_card.json` al iniciar la aplicación.
- [ ] `POST /predict`: recibe `{"url": "..."}` y devuelve `proba_phishing`, `clase`, `origen` (`lista_blanca` | `modelo`) y la versión del modelo.
- [ ] `GET /health` y `GET /model-info` (contenido de la ficha).
- [ ] Capa de lista blanca (§5) con sus propias pruebas.
- [ ] Respuesta 400/422 ante `URLInvalidaError`; CORS para la extensión.
- [ ] Registro de cada predicción (sin datos personales) para el dashboard.

**Extensión de Chrome (`extension/`)**
- [ ] Manifest V3; enviar la URL de la pestaña activa a `POST /predict`.
- [ ] Mostrar una advertencia al usuario cuando la clase sea phishing.
- [ ] Configurar la URL de la API y gestionar errores y tiempos de espera.

**Panel de monitoreo (`dashboard/`)**
- [ ] Volumen de consultas, proporción de URLs marcadas como phishing y decisiones por capa (lista blanca / modelo).
- [ ] Latencia de extremo a extremo de la API (insumo para `analysis/oe2_latencia_recursos/`).
