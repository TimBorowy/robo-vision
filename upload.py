import os
from roboflow import Roboflow

# ==============================
# CONFIGURATION
# ==============================
API_KEY = "HLipjLlisNcBNeegoV6N"
WORKSPACE_ID = "tims-workspace-rzpyu" # Found in your browser URL or Settings
PROJECT_ID = "robot-detection-zvsof" # The 'URL' name of your project
IMAGE_DIRECTORY = "./video_input/Trainingdata/nhrl-the-wall/" # Local folder with your .jpg/.png files

# ==============================
# UPLOAD LOGIC
# ==============================

def batch_upload():
    # Initialize Roboflow
    rf = Roboflow(api_key=API_KEY)
    
    # Retrieve the project
    project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)

    print(f"Starting batch upload from: {IMAGE_DIRECTORY}")

    # Supported extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # Loop through files in directory
    for filename in os.listdir(IMAGE_DIRECTORY):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(IMAGE_DIRECTORY, filename)
            
            print(f"Uploading: {filename}...", end="\r")
            
            try:
                # Upload the image
                # split="train" places it in the training set automatically
                project.upload(image_path, num_retry=3) 
            except Exception as e:
                print(f"\nFailed to upload {filename}: {e}")

    print("\nBatch upload complete!")

if __name__ == "__main__":
    batch_upload()