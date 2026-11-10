"""
app.py
------
Aplicación web (hecha con Streamlit) que muestra los pasajeros de avión
y hace pronósticos sencillos para el próximo año con varios modelos.

Para abrirla, ejecuta el archivo "iniciar_app.bat".
"""

import logging
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from datos import cargar_pasajeros

# Prophet es opcional: si no está instalado, la app sigue funcionando sin él.
try:
    from prophet import Prophet

    PROPHET_DISPONIBLE = True
except Exception:
    PROPHET_DISPONIBLE = False

# Silenciamos avisos de los modelos (no afectan a los resultados).
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN BÁSICA DE LA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pasajeros de avión",
    page_icon="✈️",
    layout="wide",
)

# Nombres de los meses en español (1 = enero, ... 12 = diciembre).
NOMBRES_MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]


# ---------------------------------------------------------------------------
# CARGA DE DATOS (se guarda en memoria para no bajarla cada vez)
# ---------------------------------------------------------------------------
@st.cache_data
def obtener_datos() -> pd.DataFrame:
    """Trae la tabla de pasajeros usando la función de datos.py."""
    return cargar_pasajeros()


# ===========================================================================
# LOS MODELOS DE PRONÓSTICO
# ---------------------------------------------------------------------------
# Todos reciben:
#   serie -> los pasajeros ordenados por fecha (con índice de fechas)
#   pasos -> cuántos meses hay que pronosticar
# y devuelven una lista de números (los pasajeros esperados).
# ===========================================================================

def modelo_copiar(serie: pd.Series, pasos: int) -> np.ndarray:
    """Copia el último año visible: cada mes igual al mismo mes del año pasado."""
    ultimo_anio = serie.iloc[-12:].to_numpy(dtype=float)
    repeticiones = int(np.ceil(pasos / 12))
    return np.tile(ultimo_anio, repeticiones)[:pasos]


def modelo_temporada(serie: pd.Series, pasos: int) -> np.ndarray:
    """
    Modelo de temporada (Holt-Winters):
      - tendencia aditiva (la serie sube poco a poco)
      - temporada multiplicativa (los meses altos crecen en proporción)
      - 12 meses de temporada (un año)
    """
    modelo = ExponentialSmoothing(
        serie.astype(float),
        trend="add",
        seasonal="mul",
        seasonal_periods=12,
        initialization_method="estimated",
    )
    return modelo.fit().forecast(pasos).to_numpy()


def modelo_suavizado(serie: pd.Series, pasos: int) -> np.ndarray:
    """Suavizamiento exponencial simple (sin tendencia ni temporada)."""
    modelo = ExponentialSmoothing(
        serie.astype(float),
        trend=None,
        seasonal=None,
        initialization_method="estimated",
    )
    return modelo.fit().forecast(pasos).to_numpy()


def modelo_sarima(serie: pd.Series, pasos: int) -> np.ndarray:
    """SARIMA: modelo clásico para series con tendencia y temporada (modelo 'airline')."""
    modelo = SARIMAX(
        serie.astype(float),
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    ajuste = modelo.fit(disp=False)
    return ajuste.forecast(pasos).to_numpy()


def modelo_regresion(serie: pd.Series, pasos: int) -> np.ndarray:
    """Regresión lineal: una recta en el tiempo + un ajuste para cada mes del año."""
    tabla = pd.DataFrame(
        {
            "y": serie.to_numpy(dtype=float),
            "t": np.arange(len(serie)),
            "mes": serie.index.month.astype(str),
        }
    )
    modelo = smf.ols("y ~ t + C(mes)", data=tabla).fit()

    # Preparamos los meses futuros para predecir.
    futuro_fechas = pd.date_range(
        start=serie.index[-1], periods=pasos + 1, freq="MS"
    )[1:]
    futuro = pd.DataFrame(
        {
            "t": np.arange(len(serie), len(serie) + pasos),
            "mes": futuro_fechas.month.astype(str),
        }
    )
    return modelo.predict(futuro).to_numpy()


def modelo_prophet(serie: pd.Series, pasos: int) -> np.ndarray:
    """Prophet: modelo de Meta que maneja tendencia y temporada automáticamente."""
    tabla = pd.DataFrame(
        {"ds": serie.index, "y": serie.to_numpy(dtype=float)}
    )
    modelo = Prophet()
    # Silenciamos los mensajes internos de Prophet/cmdstanpy durante el ajuste.
    logging.disable(logging.WARNING)
    try:
        modelo.fit(tabla)
        futuro = modelo.make_future_dataframe(periods=pasos, freq="MS")
        pronostico = modelo.predict(futuro)
    finally:
        logging.disable(logging.NOTSET)
    return pronostico["yhat"].to_numpy()[-pasos:]


# Diccionario con todos los modelos disponibles (el orden es el que se muestra).
MODELOS = {
    "Copiar el año pasado": modelo_copiar,
    "Modelo de temporada (Holt-Winters)": modelo_temporada,
    "Suavizamiento exponencial": modelo_suavizado,
    "SARIMA": modelo_sarima,
    "Regresión lineal": modelo_regresion,
}
if PROPHET_DISPONIBLE:
    MODELOS["Prophet"] = modelo_prophet

# Colores para los gráficos (uno por modelo).
COLORES = {
    "Copiar el año pasado": "orange",
    "Modelo de temporada (Holt-Winters)": "#1f77b4",
    "Suavizamiento exponencial": "#2ca02c",
    "SARIMA": "#d62728",
    "Regresión lineal": "#9467bd",
    "Prophet": "#8c564b",
}


# ---------------------------------------------------------------------------
# FUNCIÓN PARA MEDIR EL ERROR EN PORCENTAJE (error promedio)
# ---------------------------------------------------------------------------
def error_promedio_porcentaje(real: pd.Series, predicho: np.ndarray) -> float:
    """Devuelve el error promedio en porcentaje entre lo real y lo pronosticado."""
    real = np.asarray(real, dtype=float)
    predicho = np.asarray(predicho, dtype=float)
    return float(np.mean(np.abs(real - predicho) / real) * 100)


# ---------------------------------------------------------------------------
# CARGAMOS LOS DATOS
# ---------------------------------------------------------------------------
datos = obtener_datos()

# Serie con índice de fechas, que usan los modelos.
serie_completa = pd.Series(
    datos["Passengers"].to_numpy(dtype=float),
    index=pd.DatetimeIndex(datos["Month"]),
)

# ---------------------------------------------------------------------------
# TÍTULO
# ---------------------------------------------------------------------------
st.title("¿Cuántos pasajeros tendremos el próximo año?")
st.caption("Datos mensuales de pasajeros de avión entre 1949 y 1960.")

# ---------------------------------------------------------------------------
# 3 INDICADORES DE ARRIBA
# ---------------------------------------------------------------------------
total_1949 = datos.loc[datos["Month"].dt.year == 1949, "Passengers"].sum()
total_1960 = datos.loc[datos["Month"].dt.year == 1960, "Passengers"].sum()
veces_crecio = total_1960 / total_1949

col1, col2, col3 = st.columns(3)
col1.metric("Pasajeros en 1949", f"{total_1949:,.0f}")
col2.metric("Pasajeros en 1960", f"{total_1960:,.0f}")
col3.metric("Cuántas veces creció", f"{veces_crecio:.1f} veces")

st.divider()

# ---------------------------------------------------------------------------
# LAS CUATRO PESTAÑAS
# ---------------------------------------------------------------------------
pestana1, pestana2, pestana3, pestana4 = st.tabs(
    ["1. La historia", "2. La temporada", "3. El examen", "4. El pronóstico"]
)

# ===========================================================================
# PESTAÑA 1: LA HISTORIA
# ===========================================================================
with pestana1:
    st.subheader("La historia: pasajeros mes a mes")

    fig1 = px.line(
        datos,
        x="Month",
        y="Passengers",
        labels={"Month": "Año", "Passengers": "Pasajeros"},
        title="Pasajeros de avión por mes (1949-1960)",
    )
    fig1.update_traces(line_color="#1f77b4", line_width=2)
    fig1.update_layout(hovermode="x unified")
    st.plotly_chart(fig1, width="stretch")

    st.info(
        "Aquí ves cuántos pasajeros hubo cada mes desde 1949 hasta 1960. "
        "La línea sube año tras año: el negocio crece de forma clara. "
        "También se notan picos y bajones que se repiten todos los años."
    )

# ===========================================================================
# PESTAÑA 2: LA TEMPORADA
# ===========================================================================
with pestana2:
    st.subheader("La temporada: qué meses son los más fuertes")

    datos_temp = datos.copy()
    datos_temp["Mes"] = datos_temp["Month"].dt.month
    promedio_mes = (
        datos_temp.groupby("Mes")["Passengers"].mean().reindex(range(1, 13))
    )
    promedio_mes.index = NOMBRES_MESES

    # "Un mes normal" = el promedio de todos los meses.
    mes_normal = datos["Passengers"].mean()

    fig2 = px.bar(
        x=promedio_mes.index,
        y=promedio_mes.values,
        labels={"x": "Mes", "y": "Pasajeros promedio"},
        title="Promedio de pasajeros por mes del año",
    )
    fig2.update_traces(marker_color="#1f77b4")
    fig2.add_hline(
        y=mes_normal,
        line_dash="dash",
        line_color="red",
        annotation_text="Un mes normal",
        annotation_position="top right",
    )
    st.plotly_chart(fig2, width="stretch")

    st.info(
        "Cada barra es el promedio de ese mes a lo largo de los años. "
        "La línea roja marca un 'mes normal'. Julio, agosto y marzo están por "
        "encima (temporada alta) y noviembre está por debajo (temporada baja)."
    )

# ===========================================================================
# PESTAÑA 3: EL EXAMEN
# ===========================================================================
with pestana3:
    st.subheader("El examen: comparamos varios modelos")

    st.write(
        "Escondemos los últimos **24 meses** y pedimos a cada modelo que los "
        "adivine. Así vemos cuál se equivoca menos."
    )

    # Entrenamiento (todo menos los últimos 24) y prueba (los 24 escondidos).
    train = datos.iloc[:-24].reset_index(drop=True)
    test = datos.iloc[-24:].reset_index(drop=True)
    serie_train = pd.Series(
        train["Passengers"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(train["Month"]),
    )

    # Calculamos el pronóstico de cada modelo.
    pronosticos = {}
    for nombre, funcion in MODELOS.items():
        try:
            pronosticos[nombre] = funcion(serie_train, len(test))
        except Exception as error:
            st.warning(f"No se pudo usar el modelo «{nombre}»: {error}")

    # --- Gráfico: real vs cada modelo --------------------------------------
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=test["Month"], y=test["Passengers"],
        name="Real (lo que pasó)", mode="lines+markers",
        line=dict(color="black", width=3),
    ))
    for nombre, pred in pronosticos.items():
        fig3.add_trace(go.Scatter(
            x=test["Month"], y=pred,
            name=nombre, mode="lines+markers",
            line=dict(color=COLORES.get(nombre, "gray"), dash="dash"),
        ))
    fig3.update_layout(
        title="Últimos 24 meses: lo real frente a cada modelo",
        xaxis_title="Año",
        yaxis_title="Pasajeros",
        hovermode="x unified",
    )
    st.plotly_chart(fig3, width="stretch")

    # --- Tabla de errores (ordenada: el mejor arriba) ----------------------
    filas = [
        {
            "Modelo": nombre,
            "Error promedio (%)": round(
                error_promedio_porcentaje(test["Passengers"], pred), 1
            ),
        }
        for nombre, pred in pronosticos.items()
    ]
    tabla_errores = (
        pd.DataFrame(filas)
        .sort_values("Error promedio (%)")
        .reset_index(drop=True)
    )
    st.write("**Error promedio de cada modelo** (más bajo es mejor):")
    st.dataframe(tabla_errores, width="stretch", hide_index=True)

    # --- Cuál ganó ----------------------------------------------------------
    if len(tabla_errores) > 0:
        ganador = tabla_errores.iloc[0]["Modelo"]
        error_ganador = tabla_errores.iloc[0]["Error promedio (%)"]
        st.success(
            f"🏆 Ganó **{ganador}**, con un error promedio de {error_ganador:.1f}%. "
            "Es el modelo en el que más podemos confiar para este conjunto de datos."
        )
    else:
        ganador = "Modelo de temporada (Holt-Winters)"

    st.info(
        "Comparamos lo que de verdad pasó (línea negra) con lo que dijo cada "
        "modelo. El número de la tabla dice, en promedio, cuánto se equivocó cada "
        "uno en porcentaje: cuanto más bajo, mejor. El que queda arriba es el más "
        "confiable."
    )

# ===========================================================================
# PESTAÑA 4: EL PRONÓSTICO
# ===========================================================================
with pestana4:
    st.subheader("El pronóstico: los 12 meses de 1961")

    st.write(
        "Elige el modelo que quieres usar. Por defecto dejamos el que ganó el examen."
    )

    # El modelo por defecto es el que ganó el examen de la pestaña anterior.
    opciones = list(MODELOS.keys())
    if ganador in opciones:
        indice_inicial = opciones.index(ganador)
    else:
        indice_inicial = opciones.index("Modelo de temporada (Holt-Winters)")

    modelo_elegido = st.selectbox(
        "¿Qué modelo quieres usar para el pronóstico?",
        opciones,
        index=indice_inicial,
    )

    # Pronóstico de los 12 meses de 1961.
    pronostico = MODELOS[modelo_elegido](serie_completa, 12)
    fechas_futuras = pd.date_range(start="1961-01-01", periods=12, freq="MS")

    # --- Gráfico: historia + pronóstico ------------------------------------
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=datos["Month"], y=datos["Passengers"],
        name="Historia (1949-1960)", mode="lines",
        line=dict(color="black", width=2),
    ))
    fig4.add_trace(go.Scatter(
        x=fechas_futuras, y=pronostico,
        name=f"Pronóstico ({modelo_elegido})", mode="lines+markers",
        line=dict(color=COLORES.get(modelo_elegido, "#1f77b4"), width=3),
    ))
    fig4.update_layout(
        title="Pronóstico de pasajeros para 1961",
        xaxis_title="Año",
        yaxis_title="Pasajeros",
        hovermode="x unified",
    )
    st.plotly_chart(fig4, width="stretch")

    # --- Tabla mes a mes ----------------------------------------------------
    tabla_pronostico = pd.DataFrame(
        {
            "Mes": [f"{NOMBRES_MESES[f.month - 1]} {f.year}" for f in fechas_futuras],
            "Pasajeros esperados": [int(round(v)) for v in pronostico],
        }
    )
    st.write("**Pasajeros esperados, mes a mes:**")
    st.dataframe(tabla_pronostico, width="stretch", hide_index=True)

    st.metric(
        f"Total esperado en 1961 ({modelo_elegido})",
        f"{int(round(pronostico.sum())):,.0f}",
    )

    st.info(
        "Este es el pronóstico de los 12 meses de 1961 usando el modelo que "
        "elegiste arriba. Se espera que el negocio siga creciendo y que se repita "
        "el patrón de siempre: meses fuertes en verano y meses flojos a final de año."
    )
