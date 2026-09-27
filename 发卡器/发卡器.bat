@echo off
cd /d "%~dp0"
title Qwen Studio - Issue Card
"%~dp0py\python.exe" "%~dp0issue_card.py"
if errorlevel 1 pause
