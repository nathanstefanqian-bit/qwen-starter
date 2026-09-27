@echo off
cd /d "%~dp0"
title Qwen Image 2.1 Studio

set "PY=%~dp0python\python.exe"

if not exist "%PY%" goto nopython

echo ============================================================
echo   Qwen Image 2.1 Studio
echo   正在启动界面，ComfyUI 服务会在后台自动拉起。
echo   关闭本窗口即退出界面（ComfyUI 会继续在后台运行）。
echo ============================================================
echo.

"%PY%" app.py

echo.
echo 界面已退出。
pause
exit /b 0

:nopython
echo [错误] 找不到实例 python：
echo        %PY%
echo 请修改本文件里的 PY 路径，或修改 config.json 里的 instance_dir。
pause
exit /b 1
