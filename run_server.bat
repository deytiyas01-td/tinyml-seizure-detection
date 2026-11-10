@echo off
title TinyML Seizure Detection Web Server
echo ========================================================
echo Starting TinyML Seizure Detection Web Application...
echo Model: CHB-MIT Scalp EEG Tiny 1D-CNN (INT8 Quantized)
echo ========================================================
cd /d "%~dp0"
python app.py
pause
