"""Pruebas de features.py con URLs escritas a mano.

Cada valor esperado se calculó A MANO (contando caracteres), no copiando la
salida del programa. Los comentarios muestran la cuenta para poder revisarla.

Ejecutar desde la raíz del repositorio:
    python -m pytest tests -v
"""

import math
import sys
from pathlib import Path

import pytest

# Permite importar features.py (que está en la raíz del repositorio).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import (  # noqa: E402
    FEATURE_NAMES,
    URLInvalidaError,
    extract_features,
    normalize_url,
)


def revisar(url, esperado):
    """Compara solo los atributos indicados en `esperado` (los ratios con tolerancia)."""
    obtenido = extract_features(url)
    for nombre, valor in esperado.items():
        assert obtenido[nombre] == pytest.approx(valor), nombre


# ---------------------------------------------------------------------------
# Estructura general
# ---------------------------------------------------------------------------
def test_hay_20_atributos_en_orden_fijo():
    assert len(FEATURE_NAMES) == 20
    assert len(set(FEATURE_NAMES)) == 20
    resultado = extract_features("http://ejemplo.com")
    assert list(resultado) == FEATURE_NAMES  # mismo orden que FEATURE_NAMES


# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------
def test_normalize_minusculas_espacios_y_decodificacion():
    # %41 es "A": se decodifica y luego pasa a minúscula.
    assert normalize_url("  HTTP://EXAMPLE.COM/%41  ") == "http://example.com/a"


def test_normalize_decodifica_una_sola_vez():
    # "%2541" -> una pasada -> "%41" (NO se vuelve a decodificar a "a").
    assert normalize_url("http://x.com/%2541") == "http://x.com/%41"


@pytest.mark.parametrize("entrada, esperada", [
    # Ruta solo "/" -> se quita la barra final.
    ("https://www.google.com/", "https://www.google.com"),
    ("HTTPS://WWW.Google.COM/  ", "https://www.google.com"),
    ("ejemplo.com/", "ejemplo.com"),                          # sin esquema
    ("http://ejemplo.com:8080/", "http://ejemplo.com:8080"),  # con puerto
    ("http://ejemplo.com%2F", "http://ejemplo.com"),          # %2F decodificado = "/"
    # Cualquier otra ruta, consulta o fragmento -> sin cambios.
    ("https://www.google.com", "https://www.google.com"),
    ("https://es.wikipedia.org/wiki/", "https://es.wikipedia.org/wiki/"),
    ("http://ejemplo.com//", "http://ejemplo.com//"),
    ("http://ejemplo.com/?q=1/", "http://ejemplo.com/?q=1/"),
    ("http://ejemplo.com/#/", "http://ejemplo.com/#/"),
])
def test_normalize_quita_barra_solo_si_ruta_es_raiz(entrada, esperada):
    assert normalize_url(entrada) == esperada


def test_misma_url_con_y_sin_barra_da_mismos_atributos():
    assert extract_features("https://www.google.com/") == extract_features("https://www.google.com")


def test_normalize_valores_no_texto():
    assert normalize_url(None) == ""
    assert normalize_url(float("nan")) == ""


# ---------------------------------------------------------------------------
# 1. URL con IP
# ---------------------------------------------------------------------------
def test_url_con_ip():
    # "http://192.168.1.10/login.php"
    # longitud: "http://"(7) + "192.168.1.10"(12) + "/login.php"(10) = 29
    # puntos: 3 en la IP + 1 en login.php = 4 ; barras: 2 + 1 = 3
    # dígitos: 192(3) 168(3) 1(1) 10(2) = 9
    # letras: "http"(4) + "loginphp"(8) = 12
    # especiales: ":"(1) + "/"(3) + "."(4) = 8
    revisar("http://192.168.1.10/login.php", {
        "url_length": 29, "domain_length": 12, "is_ip": 1, "num_subdomains": 0,
        "has_at_symbol": 0, "has_double_slash": 0, "has_dash_in_domain": 0,
        "is_https": 0, "https_token": 0, "num_dots": 4, "num_slashes": 3,
        "num_digits": 9, "num_equal": 0, "num_qmark": 0, "num_ampersand": 0,
        "num_special_chars": 8, "num_obfuscated_chars": 0,
        "letter_ratio": 12 / 29, "path_level": 1,
    })


def test_url_con_ipv6():
    # host sin corchetes: "2001:db8::1" -> 11 caracteres
    revisar("http://[2001:db8::1]/", {"is_ip": 1, "domain_length": 11,
                                      "num_subdomains": 0, "path_level": 0})


# ---------------------------------------------------------------------------
# 2. URL con "@"
# ---------------------------------------------------------------------------
def test_url_con_arroba():
    # "http://paypal.com@evil.com/x": el navegador va a evil.com, no a paypal.
    # longitud: 7 + "paypal.com"(10) + "@"(1) + "evil.com"(8) + "/x"(2) = 28
    # host real = "evil.com" (8). letras: 4 + 9 + 7 + 1 = 21
    # especiales: ":"1 + "/"3 + "."2 + "@"1 = 7
    revisar("http://paypal.com@evil.com/x", {
        "url_length": 28, "domain_length": 8, "has_at_symbol": 1,
        "num_dots": 2, "num_slashes": 3, "num_special_chars": 7,
        "letter_ratio": 21 / 28, "path_level": 1, "num_subdomains": 0,
    })


# ---------------------------------------------------------------------------
# 3. URL con secuencias %xx
# ---------------------------------------------------------------------------
def test_url_con_porcentaje_hex():
    # Original: "http://example.com/a%20b%2Fc" -> 2 secuencias (%20 y %2f)
    # Normalizada: "http://example.com/a b/c"
    # longitud: 7 + 11 + "/a b/c"(6) = 24 ; barras: 2 + 2 = 4 (el %2F ahora es "/")
    # letras: 4 + "examplecom"(10) + a,b,c(3) = 17
    # especiales: ":"1 + "/"4 + "."1 + " "1 = 7 ; ruta "/a b/c" -> 2 niveles
    revisar("http://example.com/a%20b%2Fc", {
        "num_obfuscated_chars": 2, "url_length": 24, "num_slashes": 4,
        "num_dots": 1, "num_special_chars": 7, "letter_ratio": 17 / 24,
        "path_level": 2, "domain_length": 11,
    })


# ---------------------------------------------------------------------------
# 4. URL sin esquema
# ---------------------------------------------------------------------------
def test_url_sin_esquema():
    # "  Example.com/Path/To/Page.html  " -> "example.com/path/to/page.html"
    # longitud: 11 + "/path/to/page.html"(18) = 29 ; puntos 2 ; barras 3
    # especiales: 2 + 3 = 5 ; letras: 29 - 5 = 24
    revisar("  Example.com/Path/To/Page.html  ", {
        "url_length": 29, "domain_length": 11, "is_https": 0,
        "has_double_slash": 0, "num_dots": 2, "num_slashes": 3,
        "num_special_chars": 5, "letter_ratio": 24 / 29, "path_level": 3,
        "num_subdomains": 0,
    })


# ---------------------------------------------------------------------------
# 5. URL con varios subdominios (y "https" solo en el esquema)
# ---------------------------------------------------------------------------
def test_url_con_subdominios():
    # host "login.secure.bank.example.com": 5+1+6+1+4+1+7+1+3 = 29 caracteres
    # 5 etiquetas -> 5 - 2 = 3 subdominios
    # La "/" final (ruta solo "/") se quita al normalizar:
    #   "https://login.secure.bank.example.com"
    # longitud: "https://"(8) + 29 = 37 ; puntos 4 ; barras 2
    # especiales: ":"1 + "/"2 + "."4 = 7 ; letras: 5 + 25 = 30
    # https_token = 0 porque "https" solo está en el esquema.
    revisar("https://login.secure.bank.example.com/", {
        "url_length": 37, "domain_length": 29, "num_subdomains": 3,
        "is_https": 1, "https_token": 0, "num_dots": 4, "num_slashes": 2,
        "num_special_chars": 7, "letter_ratio": 30 / 37, "path_level": 0,
    })


def test_limitacion_co_uk():
    # Limitación documentada: "bbc.co.uk" no tiene subdominios reales, pero
    # al no usar lista de sufijos públicos se cuenta 3 - 2 = 1.
    revisar("http://bbc.co.uk", {"num_subdomains": 1})


# ---------------------------------------------------------------------------
# 6. URL con "//" en la ruta (redirección)
# ---------------------------------------------------------------------------
def test_url_con_doble_barra_en_ruta():
    # "http://site-a.com//http://evil.com"
    # longitud: 7 + "site-a.com"(10) + "//http://evil.com"(17) = 34
    # barras 6 ; puntos 2 ; especiales: ":"2 + "/"6 + "."2 + "-"1 = 11
    # letras: 4 + "siteacom"(8) + 4 + "evilcom"(7) = 23
    # ruta "//http://evil.com" -> segmentos no vacíos "http:" y "evil.com" = 2
    revisar("http://site-a.com//http://evil.com", {
        "url_length": 34, "domain_length": 10, "has_double_slash": 1,
        "has_dash_in_domain": 1, "num_slashes": 6, "num_dots": 2,
        "num_special_chars": 11, "letter_ratio": 23 / 34, "path_level": 2,
        "https_token": 0,
    })


def test_sin_doble_barra_extra():
    # Solo el "//" del esquema: no cuenta como redirección.
    revisar("https://ejemplo.com/a/b", {"has_double_slash": 0})


# ---------------------------------------------------------------------------
# 7. "https" dentro del dominio
# ---------------------------------------------------------------------------
def test_https_dentro_del_dominio():
    # "http://https-secure.com/login"
    # longitud: 7 + "https-secure.com"(16) + "/login"(6) = 29
    # especiales: ":"1 + "/"3 + "-"1 + "."1 = 6 ; letras: 4 + 14 + 5 = 23
    revisar("http://https-secure.com/login", {
        "url_length": 29, "domain_length": 16, "https_token": 1, "is_https": 0,
        "has_dash_in_domain": 1, "num_dots": 1, "num_slashes": 3,
        "num_special_chars": 6, "letter_ratio": 23 / 29, "path_level": 1,
    })


def test_conteos_de_consulta():
    # "http://a.com/p?x=1&y=2&z=3": 3 "=", 1 "?", 2 "&", dígitos 1,2,3 = 3
    revisar("http://a.com/p?x=1&y=2&z=3", {
        "num_equal": 3, "num_qmark": 1, "num_ampersand": 2, "num_digits": 3,
        "path_level": 1,
    })


# ---------------------------------------------------------------------------
# 8. Entropía (casos pequeños calculables a mano)
# ---------------------------------------------------------------------------
def test_entropia_caracteres_todos_distintos():
    # "ab.cd": 5 caracteres distintos, cada uno con p = 1/5 -> H = log2(5)
    revisar("ab.cd", {"url_entropy": math.log2(5)})


def test_entropia_caracteres_repetidos():
    # "aaa.aaa": "a" 6 veces (p=6/7) y "." 1 vez (p=1/7)
    h = -(6 / 7 * math.log2(6 / 7) + 1 / 7 * math.log2(1 / 7))  # ≈ 0.5917
    revisar("aaa.aaa", {"url_entropy": h})


# ---------------------------------------------------------------------------
# 9. URLs vacías o malformadas -> error controlado
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("url_mala", [
    "",                 # vacía
    "    ",             # solo espacios
    None,               # nulo
    "http://",          # esquema sin host
    "http://[::1",      # corchete IPv6 sin cerrar
    "esto no es url",   # host con espacios
    "localhost",        # host sin punto
])
def test_url_vacia_o_malformada(url_mala):
    with pytest.raises(URLInvalidaError):
        extract_features(url_mala)
