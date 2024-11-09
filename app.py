import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64


# Load your trained emotion detection model
model = load_model('best.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion to genre mapping
emotion_genre_map = {
    'Angry': 'Action',
    'Disgust': 'Horror',
    'Fear': 'Thriller',
    'Happy': 'Comedy',
    'Neutral': 'Drama',
    'Sad': 'Romance',
    'Surprise': 'Adventure'
}

# Function to preprocess the frame for emotion detection
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (48, 48))
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    frame_normalized = frame_gray.astype('float32') / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    frame_expanded = np.expand_dims(frame_expanded, axis=-1)
    return frame_expanded

# Set page configuration
st.set_page_config(page_title="Emotion-Based Movie Recommendation", layout="centered")

# Title of the app
st.title("EmoFlix")
# CSS to set background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
        }}
        input[type="text"] {{
            background-color: lightgrey !important;
            color: black !important;
            border-radius: 8px;
            padding: 10px;
            border: 2px solid lightblack;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function to set background
add_bg_from_local('bg.jpg')



# User input components
actor = st.text_input("Enter preferred actor:")
language = st.text_input("Enter preferred language:")

# Checkbox for running the webcam
run_webcam = st.checkbox("Run Webcam", key="webcam_checkbox")

# Initialize image display
FRAME_WINDOW = st.image([])

# Variable to hold the OpenCV capture object
cap = None
detected_emotion = None  # To hold the latest detected emotion

# Capture webcam feed only if the checkbox is checked
if run_webcam:
    # Start capturing video from the webcam
    if cap is None:
        cap = cv2.VideoCapture(0)  # Initialize the webcam only once

    # Read frame from the webcam
    ret, frame = cap.read()  # Read the frame from the webcam
    if ret:
        # Preprocess the frame for emotion detection
        processed_frame = preprocess_frame(frame)

        # Predict emotion
        emotion_probs = model.predict(processed_frame)
        emotion_index = np.argmax(emotion_probs)  # Get the index of the highest probability
        detected_emotion = emotion_labels[emotion_index]  # Get the emotion label

        # Display the emotion on the frame
        cv2.putText(frame, f'Emotion: {detected_emotion}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Convert the frame to RGB (Streamlit uses RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the resulting frame in Streamlit
        FRAME_WINDOW.image(frame_rgb, channels="RGB")

# Button to finalize the detected emotion
if st.button("Fix Emotion"):
    if detected_emotion is not None:
        # Map emotion to genre
        genre = emotion_genre_map.get(detected_emotion, "Any")  # Default to "Any" if emotion not found
        search_query = f"{actor} {language} {genre} movie list"
        st.write(f"Search Query: {search_query}")

        # Redirect to IMDb search results
        google_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
        st.write(f"[Click here to see results on Google]({google_url})")
    else:
        st.warning("No emotion detected yet. Please ensure the webcam is running.")

# Release the webcam when not running
if not run_webcam and cap is not None:
    cap.release()
    cap = None

# Optionally, ensure webcam capture is released on closing the app
if cap is not None:
    cap.release()
