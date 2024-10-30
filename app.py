import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

model = load_model()

class_labels = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

# Image preprocessing
def preprocess_image(image, target_size=(224, 224)):

     # Convert image to grayscale
    image = ImageOps.grayscale(image)

    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image = ImageOps.fit(image, target_size, Image.LANCZOS)
    image_array = np.asarray(image)
    if image_array.ndim == 2:  # If grayscale, repeat across 3 channels
        image_array = np.stack((image_array,) * 3, axis=-1)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit UI
st.title("Facial Emotion Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image", width=200)
    
    st.markdown('<p style="font-size:14px; color:grey; margin: 0; padding: 0;">Predicting...</p>', unsafe_allow_html=True)
    
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)

    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence_scores = predictions[0]

    confidence_percentages = [f"{score * 100:.2f}%" for score in confidence_scores]

    st.markdown('<p style="font-size:14px; color:grey;">Done!</p>', unsafe_allow_html=True)
    
    st.markdown(f'<p style="font-size:25px;"> <u>Predicted Emotion:</u> <b><big>{class_labels[predicted_class]}</big></b></p>', unsafe_allow_html=True)

    st.markdown('<p style="font-size:20px; margin: 0; padding: 0;"> All Predictions:', unsafe_allow_html=True)
    for label, score in zip(class_labels, confidence_percentages):
        st.markdown(f'<p style="font-size:14px; color:grey; margin: 0; padding: 0;">{label}: {score}</p>', unsafe_allow_html=True)