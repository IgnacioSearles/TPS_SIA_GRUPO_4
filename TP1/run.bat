@echo off
setlocal

set VENV_DIR=%~dp0.venv

if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo Could not create a virtual environment. Make sure Python 3
        echo is installed and available as "python" from a terminal.
        echo Download it from https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"

echo Installing dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r "%~dp0requirements-notebook.txt"

echo.
python "%~dp0interactive_main.py" %*

pause