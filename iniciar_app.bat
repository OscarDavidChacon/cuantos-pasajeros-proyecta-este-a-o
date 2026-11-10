@echo off
REM Nos ubicamos en la carpeta de este archivo.
cd /d "%~dp0"

REM Usamos el Python del entorno del proyecto (.venv), que SI tiene Streamlit.
REM La configuracion de .streamlit\config.toml ya fuerza IPv4 (127.0.0.1).
".venv\Scripts\python.exe" -m streamlit run app.py

pause
