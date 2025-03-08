import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

# Paths
video_path = '/kaggle/input/ml-hackathon-ec-campus-set-2/set_2_train/train_data/'
csv_path = '/kaggle/input/ml-hackathon-ec-campus-set-2/set_2_train/train_emotion.csv'
output_folder = '/kaggle/working/extracted_frames/'

# Load emotion labels
df = pd.read_csv(csv_path, encoding='ISO-8859-1')
emotion_mapping = {emotion: idx for idx, emotion in enumerate(df['Emotion'].unique())}
num_classes = len(emotion_mapping)

# Use the entire dataset
df_sampled = df 

# Update frame and resolution parameters
frames_per_video = 5  # Reduce frames per video
frame_height, frame_width = 112, 112  # Reduce resolution to 112x112

X, y = [], []

for idx, row in df_sampled.iterrows():
    video_file = os.path.join(video_path, f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4")
    cap = cv2.VideoCapture(video_file)
    
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_sample_interval = max(1, frame_count // frames_per_video)
    
    for i in range(frames_per_video):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_sample_interval)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (frame_height, frame_width))
            frames.append(frame / 255.0)  # Normalize
        else:
            break
    
    cap.release()
    
    if len(frames) == frames_per_video:
        X.append(np.array(frames))
        y.append(emotion_mapping[row['Emotion']])

X = np.array(X)
y = to_categorical(y, num_classes=num_classes)

# 2. Split into training and testing sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Define the Model (CNN + LSTM) with updated input shape
input_shape = (frames_per_video, frame_height, frame_width, 3)
cnn_lstm_model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

cnn_lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the Model
cnn_lstm_model.fit(X_train, y_train, epochs=5, batch_size=8, validation_split=0.2)

# 5. Evaluate the Model on Test Data
test_loss, test_accuracy = cnn_lstm_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
