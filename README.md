# Suicide Risk Detection using NLP

Machine Learning project for detecting suicidal ideation in Reddit posts using Natural Language Processing (NLP) techniques.

This repository contains a complete pipeline for preprocessing text, extracting features using TF-IDF, training a machine learning model, and performing inference on new text data.

---

# Project Overview

Mental health monitoring through social media has become an important research area.  
This project aims to classify Reddit posts into two categories:

- **0 → Non-Suicide**
- **1 → Suicide Risk**

The system uses Natural Language Processing techniques and a Logistic Regression classifier to detect potentially suicidal content in text.

The goal of this project is to demonstrate a complete **NLP machine learning workflow**, including:

- Data preprocessing
- Feature extraction
- Model training
- Model evaluation
- Model persistence
- Inference pipeline

---

# Repository Structure
suicide-risk-detection-nlp
│
├── data
│ └── combined-set.csv
│
├── models
│ ├── suicide_classifier_model.pkl
│ ├── tfidf_vectorizer.pkl
│ └── metrics.json
│
├── reddit-suicide-risk-detection.ipynb
├── predict.py
├── README.md
└── .gitignore



### Folder Explanation

**data/**
- Contains the dataset used for training and evaluation.

**models/**
- Stores the trained machine learning model and vectorizer.

**reddit-suicide-risk-detection.ipynb**
- Jupyter Notebook containing the full training pipeline.

**predict.py**
- Script used for making predictions on new text inputs.

---

# Dataset

The dataset contains Reddit posts labeled as:

- **suicide**
- **non-suicide**

Important fields used:

- `selftext_clean` → cleaned post text
- `is_suicide` → target label

Example entry:
Text: "I feel hopeless and empty"
Label: 1 (suicide risk)


---

# Machine Learning Pipeline

The model training process includes the following steps:

### 1. Data Cleaning
Text preprocessing removes:

- URLs
- mentions
- hashtags
- numbers
- punctuation

### 2. Feature Extraction

TF-IDF Vectorization is used to convert text into numerical feature vectors.

TF-IDF → Term Frequency - Inverse Document Frequency


### 3. Model Training

The classification model used:

Logistic Regression

Configuration:

- `max_iter = 1000`
- `class_weight = balanced`
- `random_state = 42`

### 4. Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Metrics are stored in:

models/metrics.json

---

# Model Usage

The trained model and vectorizer are saved as:

models/suicide_classifier_model.pkl
models/tfidf_vectorizer.pkl


---

# Example Prediction

You can use the prediction script to classify new text.

-python
----------------------------------------
from predict import predict

text = "I feel hopeless and empty"

label, probability = predict(text)

print("Prediction:", label)
print("Probability:", probability) 
----------------------------------------
Example output:

Prediction: suicide risk
Probability: 0.87

Installation

Clone the repository:

git clone https://github.com/yourusername/suicide-risk-detection-nlp.git
cd suicide-risk-detection-nlp

Install dependencies:

pip install pandas scikit-learn numpy joblib

Technologies Used

Python

Scikit-learn

Pandas

NumPy

Natural Language Processing

TF-IDF

Logistic Regression

Future Improvements

Possible extensions of this project include:

Deep Learning models (CNN / LSTM / Transformers)

BERT-based text classification

Real-time prediction API

Streamlit web application

Model explainability (SHAP / LIME)

Disclaimer

This project is intended for research and educational purposes only.
It should not be used as a medical diagnostic tool.

Mental health assessments should always be conducted by qualified professionals.

Author

P. Tilemachos

Machine Learning & NLP Projects

GitHub:
https://github.com/P-Tilemachos
