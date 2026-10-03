@echo off
title Footwear Style & Shopping Hub (Flask)
echo ===================================================
echo   Starting Footwear Style & Shopping Advisor (Flask)
echo ===================================================
echo.

cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "venv\Scripts\activate.bat"
) else (
    echo Warning: venv\Scripts\activate.bat not found. Attempting default python...
)

echo Launching Flask application...
python app.py

pause
