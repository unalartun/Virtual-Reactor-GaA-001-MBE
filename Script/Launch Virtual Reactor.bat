@echo off
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment was not found.
    echo Expected: %CD%\.venv\Scripts\python.exe
    echo.
    echo Please send this message to Codex.
    pause
    exit /b 1
)

if not exist "Script\main.py" (
    echo main.py was not found in the project folder.
    echo Please check that this file is inside the Virtual Reactor folder.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" Script\main.py
set "APP_EXIT_CODE=%ERRORLEVEL%"

if not "%APP_EXIT_CODE%"=="0" (
    echo.
    echo Virtual Reactor exited with code %APP_EXIT_CODE%.
    echo Copy the error message above and send it to Codex.
)

pause

