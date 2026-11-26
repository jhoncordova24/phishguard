"""Extracción de características (features) a partir de URLs para la detección de phishing.

Este módulo es el "corazón" compartido del proyecto: lo usan
  - ml/  (para construir el dataset de entrenamiento), y
  - api/ (para analizar una URL nueva en producción).
Como ambos llaman exactamente a la misma función `extract_features`, el modelo
ve en producción los mismos números que vio al entrenar.

Reglas de diseño:
  * Solo biblioteca estándar de Python (nada de pandas, requests, etc.).
  * Algoritmo "autocontenido": todo se calcula mirando el TEXTO de la URL.
    No se hacen consultas DNS, WHOIS ni descargas de la página.
  * Nunca revienta con una excepción inesperada: si la URL no se puede
    analizar, lanza `URLInvalidaError` (un error propio y controlado) y quien
    llama decide qué hacer (descartarla, responder 400 en la API, etc.).

Uso rápido:
    >>> from features import extract_features
    >>> extract_features("http://192.168.1.10/login.php")["is_ip"]
    1
"""

import ipaddress
import math
import re
from collections import Counter
from typing import NamedTuple
from urllib.parse import unquote, urlsplit

# ---------------------------------------------------------------------------
# Orden FIJO de los 20 atributos (Tabla XIV de la tesis).
# El modelo recibe las columnas en este orden: NO lo cambies sin reentrenar.
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "url_length",            # 1
    "domain_length",         # 2
    "is_ip",                 # 3
    "num_subdomains",        # 4
    "has_at_symbol",         # 5
    "has_double_slash",      # 6
    "has_dash_in_domain",    # 7
    "is_https",              # 8
    "https_token",           # 9
    "num_dots",              # 10
    "num_slashes",           # 11
    "num_digits",            # 12
    "num_equal",             # 13
    "num_qmark",             # 14
    "num_ampersand",         # 15
    "num_special_chars",     # 16
    "num_obfuscated_chars",  # 17
    "letter_ratio",          # 18
    "url_entropy",           # 19
    "path_level",            # 20
]

# Un esquema es "letras/dígitos/+/-/." seguido de "://", por ejemplo "http://".
# Exigimos el "://" a propósito: así "localhost:8080/x" NO se confunde con un
# esquema llamado "localhost".
_RE_ESQUEMA = re.compile(r"^[a-z][a-z0-9+.\-]*://")

# Una secuencia de ofuscación es "%" seguido de 2 dígitos hexadecimales (ej. %20).
_RE_PORCENTAJE_HEX = re.compile(r"%[0-9a-f]{2}")

_DIGITOS = "0123456789"


class URLInvalidaError(ValueError):
    """Se lanza cuando la URL está vacía o no tiene un host reconocible."""


class PartesURL(NamedTuple):
    """Piezas de una URL ya normalizada."""
    tiene_esquema: bool  # ¿empezaba con algo como "http://"?
    esquema: str         # "http", "https", ... o "" si no tenía
    host: str            # dominio o IP, sin puerto ni usuario (ej. "www.ejemplo.com")
    ruta: str            # la parte "/carpeta/archivo" (sin "?consulta")


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------
def normalize_url(raw) -> str:
    """Deja la URL en una forma estándar antes de medirla.

    Pasos:
      1. Quita espacios al inicio y al final.
      2. Decodifica las secuencias %xx UNA sola vez (ej. "%41" -> "A").
         Una sola pasada: "%2541" queda como "%41" (no se decodifica otra vez).
      3. Pasa todo a minúsculas.
      4. Vuelve a quitar espacios por si la decodificación dejó alguno en los bordes.
      5. Si la ruta es SOLO "/" (sin consulta ni fragmento), quita esa barra final:
         "https://www.google.com/" -> "https://www.google.com". Así la página
         principal de un sitio mide lo mismo escrita con o sin barra.
         Solo se toca ese caso: "http://x.com/a/" y "http://x.com/?q=1" quedan igual.

    Nota: hacemos minúsculas DESPUÉS de decodificar porque "%41" se convierte
    en "A" mayúscula; si bajáramos a minúsculas antes, esa "A" se colaría.

    Si `raw` no es texto (por ejemplo None o un NaN de pandas) devuelve "".
    """
    if not isinstance(raw, str):
        return ""
    texto = unquote(raw.strip())  # unquote nunca lanza error: si no es %xx válido lo deja igual
    texto = texto.lower().strip()
    return _quitar_barra_raiz(texto)


def _quitar_barra_raiz(url: str) -> str:
    """Quita la "/" final cuando la ruta es exactamente "/" (sin "?" ni "#")."""
    if not url.endswith("/"):
        return url
    texto_a_parsear = url if has_scheme(url) else "//" + url
    try:
        partes = urlsplit(texto_a_parsear)
    except ValueError:  # URL malformada: se deja igual y split_url decidirá
        return url
    if partes.path == "/" and not partes.query and not partes.fragment:
        return url[:-1]
    return url


def has_scheme(url: str) -> bool:
    """True si la URL empieza con un esquema del tipo "algo://"."""
    return bool(_RE_ESQUEMA.match(url))


def split_url(url_normalizada: str) -> PartesURL:
    """Separa una URL YA normalizada en esquema, host y ruta.

    Si la URL no trae esquema (ej. "ejemplo.com/login"), le anteponemos "//"
    SOLO para que urlsplit reconozca el host. El texto original no se toca:
    los conteos (longitud, puntos, etc.) se hacen sobre la URL tal cual llegó.

    Lanza URLInvalidaError si no hay un host reconocible.
    """
    if not url_normalizada:
        raise URLInvalidaError("URL vacía")

    tiene_esquema = has_scheme(url_normalizada)
    texto_a_parsear = url_normalizada if tiene_esquema else "//" + url_normalizada

    try:
        partes = urlsplit(texto_a_parsear)
        # .hostname quita el usuario ("user@"), el puerto (":8080") y los
        # corchetes de IPv6. Puede ser None si no hay host.
        host = partes.hostname or ""
    except ValueError as error:  # ej. "http://[::1" (corchete IPv6 sin cerrar)
        raise URLInvalidaError(f"No se pudo analizar la URL: {error}") from error

    if not _host_reconocible(host):
        raise URLInvalidaError(f"Host no reconocible: {host!r}")

    return PartesURL(tiene_esquema, partes.scheme, host, partes.path)


def _es_ip(host: str) -> bool:
    """True si el host es una dirección IPv4 o IPv6 válida.

    Limitación: no detecta IPs escritas en formatos raros (hexadecimal
    "0x7f.0.0.1" o decimal "2130706433"); esas se tratan como dominio.
    """
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _host_reconocible(host: str) -> bool:
    """Regla simple para decidir si un host "parece" un host real.

    Se acepta si:
      * no está vacío y no tiene espacios, y
      * es una IP válida, o tiene al menos un punto con texto después del
        último punto (ej. "ejemplo.com"; "localhost" o "ejemplo." NO pasan).
    """
    if not host or any(c.isspace() for c in host):
        return False
    if _es_ip(host):
        return True
    etiquetas = host.split(".")
    return len(etiquetas) >= 2 and all(etiquetas)  # sin etiquetas vacías ("a..b")


# ---------------------------------------------------------------------------
# Extracción de los 20 atributos
# ---------------------------------------------------------------------------
def _entropia_shannon(texto: str) -> float:
    """Entropía de Shannon en bits: mide qué tan "desordenado" es el texto.

    Fórmula: H = -suma( p * log2(p) ) para cada carácter distinto,
    donde p = (veces que aparece el carácter) / (longitud del texto).
    Un texto como "aaaa" da 0; uno con muchos caracteres distintos da más.
    """
    if not texto:
        return 0.0
    n = len(texto)
    return -sum((c / n) * math.log2(c / n) for c in Counter(texto).values())


def extract_features(raw) -> dict:
    """Calcula los 20 atributos de una URL, en el orden de FEATURE_NAMES.

    Recibe la URL "cruda" (tal como viene del dataset o de la API) y aplica
    internamente normalize_url, así entrenamiento y API siguen el mismo camino.

    Lanza URLInvalidaError si la URL está vacía o no tiene host reconocible.
    """
    url = normalize_url(raw)
    partes = split_url(url)  # puede lanzar URLInvalidaError
    host = partes.host

    # URL original solo en minúsculas (SIN decodificar %xx). Se necesita para
    # contar las secuencias %xx: después de decodificar ya no quedaría ninguna
    # y el atributo valdría siempre 0.
    original_minusculas = raw.strip().lower()

    es_ip = _es_ip(host)

    # num_subdomains = etiquetas del host - 2 ("www.ejemplo.com" -> 3 - 2 = 1).
    # LIMITACIÓN: no conocemos la lista de sufijos públicos, así que en
    # dominios como "ejemplo.co.uk" cuenta 1 subdominio aunque no haya ninguno.
    # Para una IP no tiene sentido hablar de subdominios: vale 0.
    if es_ip:
        num_subdominios = 0
    else:
        num_subdominios = max(0, len(host.split(".")) - 2)

    # "//" después del "//" del esquema: indica una posible redirección
    # (ej. "http://sitio.com//http://malo.com"). Si no hay esquema, se busca
    # en toda la URL.
    if partes.tiene_esquema:
        resto = url[url.index("://") + 3:]
    else:
        resto = url
    doble_barra = "//" in resto

    # Letras y dígitos: solo ASCII (a-z, 0-9). Una letra con tilde o de otro
    # alfabeto (posible homógrafo) cuenta como carácter especial.
    n_letras = sum(1 for c in url if "a" <= c <= "z")
    n_digitos = sum(1 for c in url if c in _DIGITOS)
    n_especiales = len(url) - n_letras - n_digitos

    # Segmentos no vacíos de la ruta: "/a/b/c.html" -> 3; "/" -> 0.
    niveles_ruta = len([s for s in partes.ruta.split("/") if s])

    valores = {
        "url_length": len(url),
        "domain_length": len(host),
        "is_ip": int(es_ip),
        "num_subdomains": num_subdominios,
        "has_at_symbol": int("@" in url),
        "has_double_slash": int(doble_barra),
        "has_dash_in_domain": int("-" in host),
        "is_https": int(partes.esquema == "https"),
        # El token "https" escondido en el host o en la ruta (no cuenta el esquema).
        "https_token": int("https" in host or "https" in partes.ruta),
        "num_dots": url.count("."),
        "num_slashes": url.count("/"),
        "num_digits": n_digitos,
        "num_equal": url.count("="),
        "num_qmark": url.count("?"),
        "num_ampersand": url.count("&"),
        "num_special_chars": n_especiales,
        "num_obfuscated_chars": len(_RE_PORCENTAJE_HEX.findall(original_minusculas)),
        "letter_ratio": n_letras / len(url) if url else 0.0,
        "url_entropy": _entropia_shannon(url),
        "path_level": niveles_ruta,
    }
    # Devolvemos el diccionario respetando el orden fijo de FEATURE_NAMES.
    return {nombre: valores[nombre] for nombre in FEATURE_NAMES}
