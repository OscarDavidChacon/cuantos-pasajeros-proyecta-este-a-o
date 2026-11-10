"""Prueba rápida: lee los datos y muestra un resumen."""

from datos import cargar_pasajeros

# 1) Leemos los datos (primero intenta internet, si no, archivo local).
tabla = cargar_pasajeros()

# 2) Cuántos meses hay.
total_meses = len(tabla)

# 3) Desde y hasta cuándo.
desde = tabla["Month"].min()
hasta = tabla["Month"].max()

# 4) Mes con menos pasajeros y mes con más pasajeros.
fila_min = tabla.loc[tabla["Passengers"].idxmin()]
fila_max = tabla.loc[tabla["Passengers"].idxmax()]

# 5) Mostramos todo en pantalla.
print("=" * 50)
print("RESUMEN DE PASAJEROS DE AVIÓN")
print("=" * 50)
print(f"Total de meses      : {total_meses}")
print(f"Desde               : {desde:%Y-%m}")
print(f"Hasta               : {hasta:%Y-%m}")
print(f"Mes con MENOS pasajeros : {fila_min['Month']:%Y-%m}  ({fila_min['Passengers']} pasajeros)")
print(f"Mes con MÁS pasajeros   : {fila_max['Month']:%Y-%m}  ({fila_max['Passengers']} pasajeros)")
print("=" * 50)
