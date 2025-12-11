@echo off
echo ==========================================
echo SASTRA Research Finder - Author ID Version
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Create virtual environment if not exists
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing/Updating dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo WARNING: Some packages may have failed. Trying alternative method...
    python -m pip install --upgrade pip
    pip install pandas openpyxl streamlit google-generativeai python-dotenv tqdm
)

REM Run preprocessing if data files don't exist
if not exist "data\publications.pkl" (
    echo.
    echo ========================================
    echo Running first-time preprocessing...
    echo This may take 1-2 minutes...
    echo ========================================
    python src\preprocess.py
    if errorlevel 1 (
        echo ERROR: Preprocessing failed
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo Starting SASTRA Research Finder
echo Open http://localhost:8501 in browser
echo Press Ctrl+C to stop
echo ========================================
echo.
streamlit run app.py

pause
