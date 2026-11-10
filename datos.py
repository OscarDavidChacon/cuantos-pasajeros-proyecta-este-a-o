"""
datos.py
--------
Aquí guardamos la función que trae los datos de pasajeros de avión.

Regla sencilla:
  1) Primero intentamos bajarlos de internet.
  2) Si no hay internet (o falla), usamos el archivo que está en la carpeta "datos".

El archivo tiene dos columnas:
  - Month       -> el mes (por ejemplo 1949-01)
  - Passengers  -> cuántos pasajeros hubo ese mes
"""

from pathlib import Path

import pandas as pd

# Dirección de internet con los datos originales.
URL_DATOS = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
)

# Carpeta donde vive este archivo (datos.py), para encontrar el archivo local
# sin importar desde dónde ejecutemos el programa.
CARPETA_PROYECTO = Path(__file__).resolve().parent

# Posibles nombres de la carpeta/archivo local (por si cambian mayúsculas).
RUTAS_LOCALES = [
    CARPETA_PROYECTO / "datos" / "airline-passengers.csv",
    CARPETA_PROYECTO / "Datos" / "airline-passengers.csv",
    CARPETA_PROYECTO / "datos" / "airline_passengers.csv",
]


def _leer_de_internet() -> pd.DataFrame:
    """Intenta leer los datos desde la URL. Si falla, lanza el error hacia arriba."""
    return pd.read_csv(URL_DATOS)


def _leer_de_archivo() -> pd.DataFrame:
    """Busca el archivo local y lo lee. Si no lo encuentra, avisa con un error claro."""
    for ruta in RUTAS_LOCALES:
        if ruta.exists():
            return pd.read_csv(ruta)
    rutas = "\n".join(f"  - {r}" for r in RUTAS_LOCALES)
    raise FileNotFoundError(
        "No se encontró el archivo local de pasajeros. Busqué en:\n" + rutas
    )


def _limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deja la tabla lista para usar:
      - Month se convierte en fecha real (tipo fecha de pandas).
      - Passengers se convierte en número entero.
      - Quitamos filas vacías y ordenamos por fecha.
    """
    df = df.copy()

    # Convertimos el texto del mes (ej: "1949-01") en una fecha de verdad.
    df["Month"] = pd.to_datetime(df["Month"])

    # Nos aseguramos de que los pasajeros sean números enteros.
    df["Passengers"] = df["Passengers"].astype(int)

    # Ordenamos por fecha y reiniciamos el contador de filas (0, 1, 2, ...).
    df = df.sort_values("Month").reset_index(drop=True)

    return df


def cargar_pasajeros(usar_internet: bool = True) -> pd.DataFrame:
    """
    Devuelve una tabla (DataFrame) con los pasajeros mensuales.

    Parámetros
    ----------
    usar_internet : bool
        Si es True (por defecto), primero intenta bajar los datos de internet.
        Si es False, va directo al archivo local.

    Devuelve
    --------
    pandas.DataFrame con dos columnas:
        Month       -> fecha del mes (datetime)
        Passengers  -> número de pasajeros (entero)
    """
    df = None

    if usar_internet:
        try:
            df = _leer_de_internet()
        except Exception as error:
            print(f"[datos.py] No se pudo usar internet ({error}). Usando archivo local...")

    # Si no hay datos todavía (sin internet o falló), vamos al archivo local.
    if df is None:
        df = _leer_de_archivo()

    return _limpiar(df)


# Si ejecutamos este archivo directamente (python datos.py), hacemos una prueba rápida.
if __name__ == "__main__":
    tabla = cargar_pasajeros()
    print(tabla.head())
    print("...")
    print(tabla.tail())
    print(f"\nTotal de meses: {len(tabla)}")
