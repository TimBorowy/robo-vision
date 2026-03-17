@echo off
if "%1"=="color" (
  python calibrate_color.py
) else if "%1"=="x" (
  python calibrate_camera.py
) else (
  python main.py
)
