import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. Extract Frames from Video
video_path = '/kaggle/input/ml-hackathon-ec-campus-set-2/set_2_train/train_data/dia0_utt11.mp4'
output_folder = '/kaggle/working/extracted_frames/'

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the video file
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate
frame_count = 0

# Extract frames and save them as images
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % int(frame_rate) == 0:  # Extract one frame per second
        frame_name = f"frame_{frame_count}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)
    frame_count += 1

cap.release()

# Display a sample frame
sample_frame = cv2.imread(os.path.join(output_folder, "frame_0.jpg"))
plt.imshow(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes
plt.show()

# 2. Load Emotion Labels from CSV
csv_path = '/kaggle/input/ml-hackathon-ec-campus-set-2/set_2_train/train_emotion.csv'
df = pd.read_csv(csv_path, encoding='ISO-8859-1')

# Map emotions to numerical labels
emotion_mapping = {emotion: idx for idx, emotion in enumerate(df['Emotion'].unique())}
print("Emotion Mapping:", emotion_mapping)

# 3. Preprocess Frames and Match with Labels
# Example: Resizing the first extracted frame
frame_path = os.path.join(output_folder, "frame_0.jpg")
img = image.load_img(frame_path, target_size=(224, 224))  # Resize frame to 224x224
img_array = image.img_to_array(img)  # Convert to numpy array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize pixel values

# Example label (assuming "neutral" corresponds to the first frame)
# Replace "neutral" with the appropriate emotion from the CSV as per your logic
emotion_label = 'neutral'  # You can map this based on CSV processing

# Display preprocessed frame
plt.imshow(img)
plt.axis('off')
plt.title(emotion_label)
plt.show()
