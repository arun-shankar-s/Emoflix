import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained emotion detection model
model = load_model('best.h5')

# Define emotion labels (adjust according to your model's output)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Neutral', 'Sad', 'Happy', 'Surprise']

# Function to preprocess the frame for emotion detection
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (48, 48))  # Resize to match the model's input size
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame_normalized = frame_gray.astype('float32') / 255.0  # Normalize pixel values
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    frame_expanded = np.expand_dims(frame_expanded, axis=-1)  # Add channel dimension
    return frame_expanded

# Set page configuration
st.set_page_config(page_title="Emotion Detection in Real-Time", layout="centered")

# Title of the app
st.title("Real-Time Emotion Detection")

# Checkbox for running the webcam
run_webcam = st.checkbox("Run Webcam")

# Initialize image display
FRAME_WINDOW = st.image([])  # Placeholder for the webcam feed

# Variable to hold the OpenCV capture object
cap = None

# Capture webcam feed only if the checkbox is checked
if run_webcam:
    # Start capturing video from the webcam
    if cap is None:
        cap = cv2.VideoCapture(0)  # Initialize the webcam only once
        if not cap.isOpened():
            st.error("Unable to access the webcam. Please check your camera settings.")
            st.stop()  # Stop execution if the webcam cannot be opened

    # Continuously capture frames from the webcam
    while run_webcam:
        ret, frame = cap.read()  # Read the frame from the webcam
        if ret:
            # Preprocess the frame for emotion detection
            processed_frame = preprocess_frame(frame)

            # Predict emotion
            emotion_probs = model.predict(processed_frame)
            emotion_index = np.argmax(emotion_probs)  # Get the index of the highest probability
            emotion = emotion_labels[emotion_index]  # Get the emotion label

            # Display the emotion on the frame
            cv2.putText(frame, f'Emotion: {emotion}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Convert the frame to RGB (Streamlit uses RGB format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the resulting frame in Streamlit
            FRAME_WINDOW.image(frame_rgb, channels="RGB")

        else:
            st.error("Failed to capture image from webcam")
            break  # Exit the loop if frame capture fails

# Release the webcam when not running
if cap is not None:
    cap.release()
    cap = None
