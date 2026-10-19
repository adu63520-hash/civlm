@echo off
chcp 65001 >nul
cd /d %~dp0
python play.py
pause
