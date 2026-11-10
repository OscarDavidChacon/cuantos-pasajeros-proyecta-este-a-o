"""
app.py
------
Aplicación web (hecha con Streamlit) que muestra los pasajeros de avión
y hace un pronóstico sencillo para el próximo año.

Para abrirla, ejecuta el archivo "iniciar_app.bat".
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from datos import cargar_pasajeros

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


# ---------------------------------------------------------------------------
# FUNCIÓN DEL "MODELO DE TEMPORADA" (Holt-Winters)
# ---------------------------------------------------------------------------
def pronostico_temporada(serie: pd.Series, pasos: int) -> np.ndarray:
    """
    Pronostica varios meses hacia adelante usando el modelo Holt-Winters:
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
    ajuste = modelo.fit()
    return ajuste.forecast(pasos).to_numpy()


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

# ---------------------------------------------------------------------------
# TÍTULO
# ---------------------------------------------------------------------------
st.title("¿Cuántos pasajeros tendremos el próximo año?")
st.caption("Datos mensuales de pasajeros de avión entre 1949 y 1960.")

# ---------------------------------------------------------------------------
# 3 INDICADORES DE ARRIBA
# ---------------------------------------------------------------------------
# Total de pasajeros en cada año.
total_1949 = datos.loc[datos["Month"].dt.year == 1949, "Passengers"].sum()
total_1960 = datos.loc[datos["Month"].dt.year == 1960, "Passengers"].sum()

# Cuántas veces creció 1960 respecto a 1949.
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

    # Promedio de pasajeros para cada mes del año (enero a diciembre).
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
    # Línea del "mes normal" para comparar.
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

    # Separamos: entrenamiento (todo menos los últimos 24) y prueba (los 24 escondidos).
    train = datos.iloc[:-24].reset_index(drop=True)
    test = datos.iloc[-24:].reset_index(drop=True)

    # --- Forma 1: "Copiar el año pasado" -----------------------------------
    # Tomamos el último año visible y repetimos sus valores.
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
    modelo_pred = pronostico_temporada(serie_train, pasos=len(test))

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

    # --- Tabla de errores ---------------------------------------------------
    tabla_errores = pd.DataFrame(
        {
            "Forma de pronosticar": ["Copiar el año pasado", "Modelo de temporada"],
            "Error promedio (%)": [round(error_copiar, 1), round(error_modelo, 1)],
        }
    )
    st.write("**Error promedio de cada forma** (más bajo es mejor):")
    st.dataframe(tabla_errores, width="stretch", hide_index=True)

    # --- Cuál ganó ----------------------------------------------------------
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

    # Usamos el Modelo de temporada con TODOS los datos para mirar hacia adelante.
    serie_completa = pd.Series(
        datos["Passengers"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(datos["Month"]),
    )
    pronostico = pronostico_temporada(serie_completa, pasos=12)
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

    # --- Tabla mes a mes ----------------------------------------------------
    tabla_pronostico = pd.DataFrame(
        {
            "Mes": [f"{NOMBRES_MESES[f.month - 1]} {f.year}" for f in fechas_futuras],
            "Pasajeros esperados": [int(round(v)) for v in pronostico],
        }
    )
    st.write("**Pasajeros esperados, mes a mes:**")
    st.dataframe(tabla_pronostico, width="stretch", hide_index=True)

    total_1961 = int(round(pronostico.sum()))
    st.metric("Total esperado en 1961", f"{total_1961:,.0f}")

    st.info(
        "Esta es la mejor estimación para el próximo año, usando el Modelo de "
        "temporada con todos los datos. Se espera que el negocio siga creciendo y "
        "que se repita el patrón de siempre: meses fuertes en verano y meses flojos "
        "a final de año."
    )
