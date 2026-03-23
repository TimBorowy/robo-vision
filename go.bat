@echo off
if "%1"=="train" (
  echo Starting YOLOv8 Training Routine...
  python train_yolo.py
) else (
  python main.py
)