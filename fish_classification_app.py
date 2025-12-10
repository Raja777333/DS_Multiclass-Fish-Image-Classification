import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Set page configuration to use full width
st.set_page_config(page_title="Fish Classification", page_icon="🐟", layout="wide")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",  
        options=["Home", "Upload Image", "Classify", "Model Insights", "About"],
        icons=["house", "cloud-upload", "search", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#E6E6FA"},
            "icon": {"color": "#FF00FF", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "color": "#333",
                "padding": "10px",
                "border-radius": "8px",
            },
            "nav-link-selected": {"background-color": "#DDA0DD", "color": "white"},
        }
    )

# Display selected section
if selected == "Home":
    st.markdown("<h1 style='color: #8A2BE2;'>🏠 Home</h1>", unsafe_allow_html=True)
    st.write("Welcome to the Multiclass Fish Image Classification App!")

elif selected == "Upload Image":
    st.markdown("<h1 style='color: #8A2BE2;'>📤 Upload Image</h1>", unsafe_allow_html=True)
    st.write("Upload a fish image for classification.")

elif selected == "Classify":
    st.markdown("<h1 style='color: #8A2BE2;'>🔍 Classify</h1>", unsafe_allow_html=True)
    st.write("The model will classify the fish image into its respective category.")

elif selected == "Model Insights":
    st.markdown("<h1 style='color: #8A2BE2;'>📊 Model Insights</h1>", unsafe_allow_html=True)
    st.write("View model performance, accuracy, and other insights.")

elif selected == "About":
    st.markdown("<h1 style='color: #8A2BE2;'>ℹ️ About</h1>", unsafe_allow_html=True)
    st.write("Learn more about this project and its objectives.")

# ---------------- HOME SECTION ----------------
if selected == "Home":
    col1 = st.columns([2])[0]  # ✅ Fix: Get the first (and only) column
    with col1:
        st.markdown("<h1 style='color: #C71585;'>🎯 Multiclass Fish Image Classification</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #DB7093;'>🔍 Project Overview</h2>", unsafe_allow_html=True)
        st.write("""
        This **Multiclass Fish Image Classification** project focuses on **identifying different fish species** using Deep Learning.
        The model has been trained on multiple fish categories using **MobileNet**, a powerful pre-trained Convolutional Neural Network (CNN).
        """)
        st.markdown("<h2 style='color: #DB7093;'>📌 Features of This Project</h2>", unsafe_allow_html=True)
        st.markdown("""
        - 🐟 **Classifies multiple fish species** using AI-powered deep learning.  
        - 🎯 **Trained with MobileNet architecture** for fast and accurate predictions.  
        - 📷 **Allows users to upload images** and get real-time classification results.  
        - 📊 **Displays confidence scores** to show model certainty.  
        - 🚀 **User-friendly Streamlit interface** with sidebar navigation.  
        """)
        st.markdown("<h2 style='color: #DB7093;'>🛠 Technologies & Tools Used</h2>", unsafe_allow_html=True)
        st.markdown("""
        - **Programming Language:** Python 🐍  
        - **Framework:** TensorFlow & Keras 🔥  
        - **Web App:** Streamlit 🌐  
        - **Model Architecture:** MobileNet (Pre-trained CNN) 🧠  
        - **Dataset Processing:** NumPy, Pandas  
        - **Visualization:** Matplotlib, Seaborn  
        """)
        st.markdown("<h2 style='color: #DB7093;'>🔄 How This Project Was Developed</h2>", unsafe_allow_html=True)
        st.markdown("""
        1. 📥 **Collected & Preprocessed Fish Images**  
        2. 🔍 **Used Data Augmentation to Improve Model Generalization**  
        3. 🏋️ **Trained MobileNet Model on Fish Dataset**  
        4. 📊 **Evaluated Model Accuracy, Precision, and Recall**  
        5. 🌐 **Deployed the Model Using Streamlit for Real-Time Classification**  
        """)


# ---------------- UPLOAD IMAGE SECTION ----------------
if selected == "Upload Image":
    st.markdown("<h1 style='color: #C71585;'>📤 Upload an Image for Classification</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        model_path = r"E:/Guvi/Raja_Project5/mobilenet_fish_model.keras"
        model = tf.keras.models.load_model(model_path)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        class_labels = [
            "animal fish", "animal fish bass", "fish sea_food black_sea_sprat",
            "fish sea_food gilt_head_bream", "fish sea_food hourse_mackerel",
            "fish sea_food red_mullet", "fish sea_food red_sea_bream",
            "fish sea_food sea_bass", "fish sea_food shrimp",
            "fish sea_food striped_red_mullet", "fish sea_food trout"
        ]
        predicted_label = class_labels[predicted_class]
        st.subheader(f"🎯 Predicted Fish Species: **{predicted_label}**")
        st.write(f"🔍 Confidence Score: **{np.max(prediction) * 100:.2f}%**")

# ---------------- CLASSIFY SECTION ----------------
if selected == "Classify":
    st.markdown("<h1 style='color: #C71585;'>🔍 Classify Fish Species</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a fish image for classification...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        model_path = r"E:/Guvi/Raja_Project5/mobilenet_fish_model.keras"
        model = tf.keras.models.load_model(model_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        class_labels = [
            "animal fish", "animal fish bass", "fish sea_food black_sea_sprat",
            "fish sea_food gilt_head_bream", "fish sea_food hourse_mackerel",
            "fish sea_food red_mullet", "fish sea_food red_sea_bream",
            "fish sea_food sea_bass", "fish sea_food shrimp",
            "fish sea_food striped_red_mullet", "fish sea_food trout"
        ]
        predicted_label = class_labels[predicted_class]
        confidence_score = np.max(predictions) * 100
        st.subheader(f"🎯 Predicted Fish Species: **{predicted_label}**")
        st.write(f"🔍 Confidence Score: **{confidence_score:.2f}%**")
        st.subheader("📊 Confidence Scores for All Classes")
        for i, label in enumerate(class_labels):
            st.write(f"**{label}:** {predictions[0][i] * 100:.2f}%")

# ---------------- MODEL INSIGHTS SECTION ----------------
if selected == "Model Insights":
    st.markdown("<h1 style='color: #C71585;'>📊 Model Performance & Insights</h1>", unsafe_allow_html=True)
    st.subheader("🔹 Model Evaluation Metrics")
    st.markdown("""
    - **Final Validation Accuracy:** 98.7%  
    - **Final Validation Loss:** 0.05  
    - **Best Performing Model:** MobileNet  
    """)

