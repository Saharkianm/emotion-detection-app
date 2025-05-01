import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Get absolute paths
model_path = 'emotion_model.h5'
tokenizer_path = 'tokenizer.pkl'

# Load model and tokenizer
model = load_model(model_path)

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
    
# Set max length (must match training!)
max_length = 100  # or whatever you used in pad_sequences

# Label map (adjust if your dataset is different)
label_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# Emoji map (optional)
emoji_map = {
    'sadness': '😢',
    'joy': '😊',
    'love': '❤',
    'anger': '😠',
    'fear': '😨',
    'surprise': '😲'
}

# App title
st.set_page_config(page_title="Emotion Detector", page_icon="🧠")
st.title("Emotion Detection from Text")
st.markdown("Enter a sentence and see which emotion it expresses!")

# Input box
user_input = st.text_area("Type your sentence here:")

if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')

        # Predict
        prediction = model.predict(padded)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)

        # Get emotion and emoji
        emotion = label_map[predicted_label]
        emoji = emoji_map[emotion]

        # Display result
        st.success(f"*Predicted Emotion:* {emotion.capitalize()} {emoji}")
        st.info(f"*Confidence:* {confidence:.2f}")
