# Skin Disease Classifier using CNN

A deep learning-powered web application that classifies skin lesion images into seven skin disease categories using a Convolutional Neural Network (CNN). The model is trained on the HAM10000 dataset and deployed with Streamlit and Hugging Face Spaces for real-time predictions.

## Live Demo

🔗 Hugging Face Space: https://huggingface.co/spaces/Bibek360/Skin-Disease-Classifier

## Features

* Upload skin lesion images (JPG, JPEG, PNG)
* CNN-based image classification
* Real-time disease prediction
* Confidence score display
* Class probability visualization
* Interactive Streamlit web interface
* Deployed using Docker and Hugging Face Spaces

## Disease Categories

The model classifies images into the following categories:

* AKIEC – Actinic Keratoses and Intraepithelial Carcinoma
* BCC – Basal Cell Carcinoma
* BKL – Benign Keratosis-like Lesions
* DF – Dermatofibroma
* MEL – Melanoma
* NV – Melanocytic Nevi
* VASC – Vascular Lesions

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Pillow (PIL)
* Streamlit
* Docker
* Hugging Face Spaces

## Dataset

The model was trained on the HAM10000 (Human Against Machine with 10,000 Training Images) dataset, a widely used benchmark dataset for skin lesion classification.

## Project Workflow

1. Upload a skin lesion image.
2. Resize image to 224×224 pixels.
3. Normalize pixel values.
4. Perform prediction using the trained CNN model.
5. Display the predicted disease class and confidence score.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Skin-Disease-Classifier.git
cd Skin-Disease-Classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run src/streamlit_app.py
```

## Project Structure

```text
Skin-Disease-Classifier/
│
├── Dockerfile
├── README.md
├── requirements.txt
├── src/
│   ├── streamlit_app.py
│   └── skin_cancer_model.keras
└── .streamlit/
    └── config.toml
```

## Disclaimer

This project is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis or treatment.

## Author

Bibek Sahani

Computer Science & Engineering Student | AI/ML Enthusiast
