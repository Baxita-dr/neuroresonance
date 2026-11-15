@echo off
setlocal
cd /d "%~dp0"
title NeuroResonance

echo Uruchamianie NeuroResonance...
echo Projekt: %CD%
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Tworzenie lokalnego srodowiska Python...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.12 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 goto setup_error
)

".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
if errorlevel 1 goto python_version_error

echo Sprawdzanie zaleznosci...
".venv\Scripts\python.exe" -m pip --version >nul 2>nul
if errorlevel 1 (
    echo Brak pip w srodowisku projektu. Odtwarzanie pip...
    ".venv\Scripts\python.exe" -m ensurepip --upgrade
    if errorlevel 1 goto pip_error
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto setup_error

echo.
echo Otwieranie aplikacji pod adresem http://localhost:8501
echo To okno pozostaw otwarte podczas pracy aplikacji.
echo.
".venv\Scripts\python.exe" -m streamlit run src/dashboard.py --server.address=127.0.0.1 --server.port=8501
if errorlevel 1 goto run_error
exit /b 0

:setup_error
echo.
echo Nie udalo sie przygotowac Pythona lub wymaganych bibliotek.
echo Sprawdz, czy Python 3 jest zainstalowany i czy masz dostep do internetu.
pause
exit /b 1

:pip_error
echo.
echo Nie udalo sie odtworzyc pip. Instalacja Pythona moze nie zawierac ensurepip.
echo Zainstaluj Python 3.12 z python.org, wlaczajac pip.
pause
exit /b 1

:python_version_error
echo.
echo NeuroResonance wymaga Pythona 3.12.
pause
exit /b 1

:run_error
echo.
echo Aplikacja zakonczyla sie bledem. Szczegoly sa powyzej.
pause
exit /b 1
