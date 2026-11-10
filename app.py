"""
app.py
------
Aplicación web (hecha con Streamlit) que muestra los pasajeros de avión
y hace un pronóstico sencillo para el próximo año.

Pestañas 1 a 4: la especificación original (examen con dos formas).
Pestaña 5: modelos extra (SARIMA, Suavizamiento exponencial, Regresión
lineal y Prophet).

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
# MODELOS DE PRONÓSTICO
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
    tabla = pd.DataFrame({"ds": serie.index, "y": serie.to_numpy(dtype=float)})
    modelo = Prophet()
    logging.disable(logging.WARNING)
    try:
        modelo.fit(tabla)
        futuro = modelo.make_future_dataframe(periods=pasos, freq="MS")
        pronostico = modelo.predict(futuro)
    finally:
        logging.disable(logging.NOTSET)
    return pronostico["yhat"].to_numpy()[-pasos:]


# Diccionario con todos los modelos (el orden es el que se muestra).
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
# LAS PESTAÑAS
# ---------------------------------------------------------------------------
pestana1, pestana2, pestana3, pestana4, pestana5 = st.tabs(
    [
        "1. La historia",
        "2. La temporada",
        "3. El examen",
        "4. El pronóstico",
        "5. Más modelos",
    ]
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
    st.subheader("El examen: probamos dos formas de pronosticar")

    st.write(
        "Escondemos los últimos **24 meses** y tratamos de adivinarlos. "
        "Así comprobamos qué forma de pronosticar se equivoca menos."
    )

    train = datos.iloc[:-24].reset_index(drop=True)
    test = datos.iloc[-24:].reset_index(drop=True)

    # --- Forma 1: "Copiar el año pasado" -----------------------------------
    ultimo_anio_visible = train.iloc[-12:]
    valor_por_mes = dict(
        zip(ultimo_anio_visible["Month"].dt.month, ultimo_anio_visible["Passengers"])
    )
    copiar = test["Month"].dt.month.map(valor_por_mes).to_numpy()

    # --- Forma 2: "Modelo de temporada" (Holt-Winters) ----------------------
    serie_train = pd.Series(
        train["Passengers"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(train["Month"]),
    )
    modelo_pred = modelo_temporada(serie_train, pasos=len(test))

    # --- Errores promedio ---------------------------------------------------
    error_copiar = error_promedio_porcentaje(test["Passengers"], copiar)
    error_modelo = error_promedio_porcentaje(test["Passengers"], modelo_pred)

    # --- Gráfico: real vs las dos formas -----------------------------------
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=test["Month"], y=test["Passengers"],
        name="Real (lo que pasó)", mode="lines+markers",
        line=dict(color="black", width=3),
    ))
    fig3.add_trace(go.Scatter(
        x=test["Month"], y=copiar,
        name="Copiar el año pasado", mode="lines+markers",
        line=dict(color="orange", dash="dash"),
    ))
    fig3.add_trace(go.Scatter(
        x=test["Month"], y=modelo_pred,
        name="Modelo de temporada", mode="lines+markers",
        line=dict(color="#1f77b4", dash="dash"),
    ))
    fig3.update_layout(
        title="Últimos 24 meses: lo real frente a los dos pronósticos",
        xaxis_title="Año",
        yaxis_title="Pasajeros",
        hovermode="x unified",
    )
    st.plotly_chart(fig3, width="stretch")

    tabla_errores = pd.DataFrame(
        {
            "Forma de pronosticar": ["Copiar el año pasado", "Modelo de temporada"],
            "Error promedio (%)": [round(error_copiar, 1), round(error_modelo, 1)],
        }
    )
    st.write("**Error promedio de cada forma** (más bajo es mejor):")
    st.dataframe(tabla_errores, width="stretch", hide_index=True)

    if error_modelo < error_copiar:
        st.success(
            f"🏆 Ganó el **Modelo de temporada**, con un error promedio de "
            f"{error_modelo:.1f}%, frente al {error_copiar:.1f}% de copiar el año pasado."
        )
    else:
        st.success(
            f"🏆 Ganó **Copiar el año pasado**, con un error promedio de "
            f"{error_copiar:.1f}%, frente al {error_modelo:.1f}% del modelo de temporada."
        )

    st.info(
        "Comparamos lo que de verdad pasó (línea negra) con las dos formas de "
        "pronosticar. El número de la tabla dice, en promedio, cuánto se equivocó "
        "cada una en porcentaje. Cuanto más bajo, mejor: esa es la forma en la que "
        "más podemos confiar."
    )

# ===========================================================================
# PESTAÑA 4: EL PRONÓSTICO
# ===========================================================================
with pestana4:
    st.subheader("El pronóstico: los 12 meses de 1961")

    pronostico = modelo_temporada(serie_completa, pasos=12)
    fechas_futuras = pd.date_range(start="1961-01-01", periods=12, freq="MS")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=datos["Month"], y=datos["Passengers"],
        name="Historia (1949-1960)", mode="lines",
        line=dict(color="black", width=2),
    ))
    fig4.add_trace(go.Scatter(
        x=fechas_futuras, y=pronostico,
        name="Pronóstico 1961", mode="lines+markers",
        line=dict(color="#1f77b4", width=3),
    ))
    fig4.update_layout(
        title="Pronóstico de pasajeros para 1961",
        xaxis_title="Año",
        yaxis_title="Pasajeros",
        hovermode="x unified",
    )
    st.plotly_chart(fig4, width="stretch")

    tabla_pronostico = pd.DataFrame(
        {
            "Mes": [f"{NOMBRES_MESES[f.month - 1]} {f.year}" for f in fechas_futuras],
            "Pasajeros esperados": [int(round(v)) for v in pronostico],
        }
    )
    st.write("**Pasajeros esperados, mes a mes:**")
    st.dataframe(tabla_pronostico, width="stretch", hide_index=True)

    st.metric("Total esperado en 1961", f"{int(round(pronostico.sum())):,.0f}")

    st.info(
        "Esta es la mejor estimación para el próximo año, usando el Modelo de "
        "temporada con todos los datos. Se espera que el negocio siga creciendo y "
        "que se repita el patrón de siempre: meses fuertes en verano y meses flojos "
        "a final de año."
    )

# ===========================================================================
# PESTAÑA 5: MÁS MODELOS
# ===========================================================================
with pestana5:
    st.subheader("Más modelos: comparamos seis formas de pronosticar")

    st.write(
        "Además de las dos formas del examen, probamos cuatro modelos más. "
        "Escondemos los últimos **24 meses** y vemos cuál se equivoca menos."
    )

    train5 = datos.iloc[:-24].reset_index(drop=True)
    test5 = datos.iloc[-24:].reset_index(drop=True)
    serie_train5 = pd.Series(
        train5["Passengers"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(train5["Month"]),
    )

    # Calculamos el pronóstico de cada modelo.
    pronosticos = {}
    for nombre, funcion in MODELOS.items():
        try:
            pronosticos[nombre] = funcion(serie_train5, len(test5))
        except Exception as error:
            st.warning(f"No se pudo usar el modelo «{nombre}»: {error}")

    # --- Gráfico: real vs cada modelo --------------------------------------
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=test5["Month"], y=test5["Passengers"],
        name="Real (lo que pasó)", mode="lines+markers",
        line=dict(color="black", width=3),
    ))
    for nombre, pred in pronosticos.items():
        fig5.add_trace(go.Scatter(
            x=test5["Month"], y=pred,
            name=nombre, mode="lines+markers",
            line=dict(color=COLORES.get(nombre, "gray"), dash="dash"),
        ))
    fig5.update_layout(
        title="Últimos 24 meses: lo real frente a cada modelo",
        xaxis_title="Año",
        yaxis_title="Pasajeros",
        hovermode="x unified",
    )
    st.plotly_chart(fig5, width="stretch")

    # --- Tabla de errores (el mejor arriba) --------------------------------
    filas = [
        {
            "Modelo": nombre,
            "Error promedio (%)": round(
                error_promedio_porcentaje(test5["Passengers"], pred), 1
            ),
        }
        for nombre, pred in pronosticos.items()
    ]
    tabla_modelos = (
        pd.DataFrame(filas)
        .sort_values("Error promedio (%)")
        .reset_index(drop=True)
    )
    st.write("**Error promedio de cada modelo** (más bajo es mejor):")
    st.dataframe(tabla_modelos, width="stretch", hide_index=True)

    # Gráfico de barras: el error de cada modelo (más bajo es mejor).
    fig5b = px.bar(
        tabla_modelos,
        x="Modelo",
        y="Error promedio (%)",
        title="Error promedio de cada modelo (más bajo es mejor)",
        color="Error promedio (%)",
        color_continuous_scale="RdYlGn_r",
    )
    fig5b.update_layout(xaxis_title="Modelo", yaxis_title="Error promedio (%)")
    st.plotly_chart(fig5b, width="stretch")

    if len(tabla_modelos) > 0:
        ganador5 = tabla_modelos.iloc[0]["Modelo"]
        st.success(
            f"🏆 Entre todos, ganó **{ganador5}**, con un error promedio de "
            f"{tabla_modelos.iloc[0]['Error promedio (%)']:.1f}%."
        )
    else:
        ganador5 = "Modelo de temporada (Holt-Winters)"

    st.info(
        "Aquí comparamos más modelos a la vez. El que queda arriba en la tabla es "
        "el que mejor adivina los meses escondidos. En general los que entienden la "
        "temporada (Holt-Winters, Prophet, SARIMA) aciertan más que los más simples."
    )

    st.divider()

    # --- Pronóstico de 1961 con el modelo que elijas ------------------------
    st.subheader("Pronostica 1961 con el modelo que quieras")

    opciones = list(MODELOS.keys())
    indice_inicial = opciones.index(ganador5) if ganador5 in opciones else 0
    modelo_elegido = st.selectbox(
        "¿Qué modelo quieres usar para el pronóstico de 1961?",
        opciones,
        index=indice_inicial,
    )

    pronostico5 = MODELOS[modelo_elegido](serie_completa, 12)
    fechas_futuras5 = pd.date_range(start="1961-01-01", periods=12, freq="MS")

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=datos["Month"], y=datos["Passengers"],
        name="Historia (1949-1960)", mode="lines",
        line=dict(color="black", width=2),
    ))
    fig6.add_trace(go.Scatter(
        x=fechas_futuras5, y=pronostico5,
        name=f"Pronóstico ({modelo_elegido})", mode="lines+markers",
        line=dict(color=COLORES.get(modelo_elegido, "#1f77b4"), width=3),
    ))
    fig6.update_layout(
        title=f"Pronóstico de pasajeros para 1961 — {modelo_elegido}",
        xaxis_title="Año",
        yaxis_title="Pasajeros",
        hovermode="x unified",
    )
    st.plotly_chart(fig6, width="stretch")

    tabla_pron5 = pd.DataFrame(
        {
            "Mes": [f"{NOMBRES_MESES[f.month - 1]} {f.year}" for f in fechas_futuras5],
            "Pasajeros esperados": [int(round(v)) for v in pronostico5],
        }
    )
    st.write("**Pasajeros esperados, mes a mes:**")
    st.dataframe(tabla_pron5, width="stretch", hide_index=True)

    st.metric(
        f"Total esperado en 1961 ({modelo_elegido})",
        f"{int(round(pronostico5.sum())):,.0f}",
    )

    st.info(
        "Cambia el modelo de arriba para ver cómo cambia el pronóstico de 1961. "
        "Si varios modelos coinciden en el número, más tranquilos podemos estar; "
        "si se separan mucho, hay más incertidumbre."
    )
