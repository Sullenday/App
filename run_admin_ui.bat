@echo off
python -m uvicorn app.api.admin_ui:app --host 0.0.0.0 --port 8020
pause
