"""
generar_graficos.py
-------------------
Genera los mismos gráficos de la app como imágenes PNG, para guardarlos
en el repositorio (carpeta "graficos/").

Ejecuta con:  uv run python generar_graficos.py
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from datos import cargar_pasajeros

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

try:
    from prophet import Prophet

    PROPHET_DISPONIBLE = True
except Exception:
    PROPHET_DISPONIBLE = False

NOMBRES_MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]


# ---------------------------------------------------------------------------
# MODELOS (los mismos que usa app.py)
# ---------------------------------------------------------------------------
def modelo_copiar(serie: pd.Series, pasos: int) -> np.ndarray:
    ultimo_anio = serie.iloc[-12:].to_numpy(dtype=float)
    repeticiones = int(np.ceil(pasos / 12))
    return np.tile(ultimo_anio, repeticiones)[:pasos]


def modelo_temporada(serie: pd.Series, pasos: int) -> np.ndarray:
    modelo = ExponentialSmoothing(
        serie.astype(float),
        trend="add",
        seasonal="mul",
        seasonal_periods=12,
        initialization_method="estimated",
    )
    return modelo.fit().forecast(pasos).to_numpy()


def modelo_suavizado(serie: pd.Series, pasos: int) -> np.ndarray:
    modelo = ExponentialSmoothing(
        serie.astype(float),
        trend=None,
        seasonal=None,
        initialization_method="estimated",
    )
    return modelo.fit().forecast(pasos).to_numpy()


def modelo_sarima(serie: pd.Series, pasos: int) -> np.ndarray:
    modelo = SARIMAX(
        serie.astype(float),
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return modelo.fit(disp=False).forecast(pasos).to_numpy()


def modelo_regresion(serie: pd.Series, pasos: int) -> np.ndarray:
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


MODELOS = {
    "Copiar el año pasado": modelo_copiar,
    "Modelo de temporada (Holt-Winters)": modelo_temporada,
    "Suavizamiento exponencial": modelo_suavizado,
    "SARIMA": modelo_sarima,
    "Regresión lineal": modelo_regresion,
}
if PROPHET_DISPONIBLE:
    MODELOS["Prophet"] = modelo_prophet

COLORES = {
    "Copiar el año pasado": "orange",
    "Modelo de temporada (Holt-Winters)": "#1f77b4",
    "Suavizamiento exponencial": "#2ca02c",
    "SARIMA": "#d62728",
    "Regresión lineal": "#9467bd",
    "Prophet": "#8c564b",
}


def error_promedio_porcentaje(real, predicho) -> float:
    real = np.asarray(real, dtype=float)
    predicho = np.asarray(predicho, dtype=float)
    return float(np.mean(np.abs(real - predicho) / real) * 100)


# ---------------------------------------------------------------------------
# DATOS
# ---------------------------------------------------------------------------
datos = cargar_pasajeros()
serie_completa = pd.Series(
    datos["Passengers"].to_numpy(dtype=float),
    index=pd.DatetimeIndex(datos["Month"]),
)

SALIDA = Path("graficos")
SALIDA.mkdir(exist_ok=True)


def guardar(fig: go.Figure, nombre: str) -> None:
    """Guarda la figura como PNG y como HTML."""
    ruta_png = SALIDA / f"{nombre}.png"
    ruta_html = SALIDA / f"{nombre}.html"
    fig.write_image(str(ruta_png), width=1200, height=600, scale=2)
    fig.write_html(str(ruta_html), include_plotlyjs="cdn")
    print(f"  guardado: {ruta_png}  y  {ruta_html}")


# ===========================================================================
# 1. LA HISTORIA
# ===========================================================================
fig1 = px.line(
    datos,
    x="Month",
    y="Passengers",
    labels={"Month": "Año", "Passengers": "Pasajeros"},
    title="Pasajeros de avión por mes (1949-1960)",
)
fig1.update_traces(line_color="#1f77b4", line_width=2)
fig1.update_layout(hovermode="x unified")
guardar(fig1, "1_la_historia")

# ===========================================================================
# 2. LA TEMPORADA
# ===========================================================================
datos_temp = datos.copy()
datos_temp["Mes"] = datos_temp["Month"].dt.month
promedio_mes = datos_temp.groupby("Mes")["Passengers"].mean().reindex(range(1, 13))
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
guardar(fig2, "2_la_temporada")

# ===========================================================================
# 3. EL EXAMEN
# ===========================================================================
train = datos.iloc[:-24].reset_index(drop=True)
test = datos.iloc[-24:].reset_index(drop=True)

ultimo_anio_visible = train.iloc[-12:]
valor_por_mes = dict(
    zip(ultimo_anio_visible["Month"].dt.month, ultimo_anio_visible["Passengers"])
)
copiar = test["Month"].dt.month.map(valor_por_mes).to_numpy()

serie_train = pd.Series(
    train["Passengers"].to_numpy(dtype=float),
    index=pd.DatetimeIndex(train["Month"]),
)
modelo_pred = modelo_temporada(serie_train, pasos=len(test))

error_copiar = error_promedio_porcentaje(test["Passengers"], copiar)
error_modelo = error_promedio_porcentaje(test["Passengers"], modelo_pred)

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
    title=(
        "Últimos 24 meses: lo real frente a los dos pronósticos "
        f"(error: copiar {error_copiar:.1f} % vs modelo {error_modelo:.1f} %)"
    ),
    xaxis_title="Año",
    yaxis_title="Pasajeros",
    hovermode="x unified",
)
guardar(fig3, "3_el_examen")

# ===========================================================================
# 4. EL PRONÓSTICO
# ===========================================================================
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
guardar(fig4, "4_el_pronostico")

# ===========================================================================
# 5. MÁS MODELOS
# ===========================================================================
train5 = datos.iloc[:-24].reset_index(drop=True)
test5 = datos.iloc[-24:].reset_index(drop=True)
serie_train5 = pd.Series(
    train5["Passengers"].to_numpy(dtype=float),
    index=pd.DatetimeIndex(train5["Month"]),
)

pronosticos = {}
for nombre, funcion in MODELOS.items():
    try:
        pronosticos[nombre] = funcion(serie_train5, len(test5))
    except Exception as error:
        print(f"  aviso: no se pudo usar «{nombre}»: {error}")

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
guardar(fig5, "5_mas_modelos")

# ===========================================================================
# 6. PRONÓSTICO CON EL MODELO GANADOR
# ===========================================================================
filas = [
    {
        "Modelo": nombre,
        "Error promedio (%)": round(
            error_promedio_porcentaje(test5["Passengers"], pred), 1
        ),
    }
    for nombre, pred in pronosticos.items()
]
tabla_modelos = pd.DataFrame(filas).sort_values("Error promedio (%)")
ganador = tabla_modelos.iloc[0]["Modelo"] if len(tabla_modelos) else "Modelo de temporada (Holt-Winters)"

# Barra: error de cada modelo (más bajo es mejor).
fig7 = px.bar(
    tabla_modelos,
    x="Modelo",
    y="Error promedio (%)",
    title="Error promedio de cada modelo (más bajo es mejor)",
    color="Error promedio (%)",
    color_continuous_scale="RdYlGn_r",
)
fig7.update_layout(xaxis_title="Modelo", yaxis_title="Error promedio (%)")
guardar(fig7, "7_error_por_modelo")

pronostico5 = MODELOS[ganador](serie_completa, 12)
fechas_futuras5 = pd.date_range(start="1961-01-01", periods=12, freq="MS")

fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x=datos["Month"], y=datos["Passengers"],
    name="Historia (1949-1960)", mode="lines",
    line=dict(color="black", width=2),
))
fig6.add_trace(go.Scatter(
    x=fechas_futuras5, y=pronostico5,
    name=f"Pronóstico ({ganador})", mode="lines+markers",
    line=dict(color=COLORES.get(ganador, "#1f77b4"), width=3),
))
fig6.update_layout(
    title=f"Pronóstico de pasajeros para 1961 — {ganador}",
    xaxis_title="Año",
    yaxis_title="Pasajeros",
    hovermode="x unified",
)
guardar(fig6, "6_pronostico_ganador")

print()
print("Listo. Gráficos generados en la carpeta 'graficos/'.")
print(f"Ganador del examen: {ganador}")
print(f"Total esperado 1961: {int(round(pronostico.sum())):,}")
