import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "skin_cancer_model.keras"
)

print("MODEL_PATH:", MODEL_PATH)
print("EXISTS:", os.path.exists(MODEL_PATH))

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

IMG_SIZE = 224

# Page title
st.title("Skin Cancer Classifier")

st.write(
    "Upload a skin lesion image and the model will predict the disease class."
)

# Image uploader
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.write("File uploaded successfully")

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.write("Image opened successfully")

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocessing
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    st.write("About to predict")

    # Prediction
    prediction = model.predict(image)

    st.write("Prediction completed")

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    # Display result
    st.subheader(
        f"Prediction: {class_names[predicted_class]}"
    )

    st.write(
        f"Confidence: {confidence:.2%}"
    )

    # Display all probabilities
    st.subheader("Class Probabilities")

    for i, class_name in enumerate(class_names):

        st.write(
            f"{class_name}: {prediction[0][i]:.4f}"
        )
st.markdown("---")

st.caption(
    "For educational purposes only. Not intended for medical diagnosis."
)