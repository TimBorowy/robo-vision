# train_yolo.py
from ultralytics import YOLO

def main():
    # Load the base model
    model = YOLO('yolov8n.pt')

    # Start training
    # We specify device=0 to use your 3060 Ti
    model.train(
        data='./robot-detection-dataset-v2/data.yaml', 
        epochs=100, 
        imgsz=640, 
        device=0,
        workers=4  # Reduced from 8 to 4 to be safe with 8GB VRAM
    )

if __name__ == '__main__':
    main()