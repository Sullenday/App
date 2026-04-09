@echo off
setlocal
cd /d %~dp0
if "%~1"=="" (
  echo Usage: check_folder.bat C:\path\to\images
  exit /b 1
)
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_folder.py "%~1"
