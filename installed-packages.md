# Phase 2 Vision Dependencies
# IMPORTANT: Use the specific CUDA 11.8 link below to enable your GPU.
# WARNING: Do NOT install 'opencv-python-headless'. If 'cv2.imshow' fails,
# run: pip uninstall opencv-python-headless

pip install "numpy<2.0.0"
pip install opencv-python opencv-contrib-python
pip install ultralytics
pip install pupil-apriltags
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118