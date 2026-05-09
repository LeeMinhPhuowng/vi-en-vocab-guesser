@echo off
title Vocab Solver Auto
color 0A

echo [1/3] Dang tat cac cua so Edge cu (de tranh xung dot)...
taskkill /F /IM msedge.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/3] Dang mo Edge ho tro tu dong hoa (CDP port 9222)...
start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --remote-debugging-port=9222 --remote-allow-origins=*
timeout /t 3 /nobreak >nul

echo [3/3] Dang khoi dong AI Solver...
python main.py

pause
