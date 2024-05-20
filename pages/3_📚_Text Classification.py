import streamlit as st
import os
import pickle
from sklearn.datasets import fetch_20newsgroups
import numpy as np



# Define the directory where models are saved
model_dir = r'C:\Users\ganad\Desktop\Resume Work\NLP_HUB\models'

# Dictionary to map model names to file names
model_files = {
    'Naive Bayes': os.path.join(model_dir, 'naive_bayes_classifier.pkl'),
    'SVM': os.path.join(model_dir, 'svm_classifier.pkl'),
    'Logistic Regression': os.path.join(model_dir, 'logistic_regression_classifier.pkl')
}

# Function to load a model
def load_model(model_name):
    with open(model_files[model_name], 'rb') as f:
        model = pickle.load(f)
    return model

# Load all models
models = {name: load_model(name) for name in model_files.keys()}

# Load the dataset to fetch categories
data = fetch_20newsgroups(subset='train')
categories = data.target_names

st.title("Text Classification with Pre-trained Models")
st.markdown("""
## Text Classification Page

This page allows you to input text, select a classification model, and view the predicted category. The models available include Naive Bayes, SVM, and Logistic Regression. We'll also provide some mathematical explanation behind each model.
""")

# Text input
user_input = st.text_area("Enter text to classify:", "Natural Language Processing is fascinating and very useful.")

# Model selection
model_name = st.selectbox("Choose a classification model:", list(models.keys()))

# Submit button
if st.button('Classify'):
    if user_input:
        # Load the selected model
        model = models[model_name]
        
        # Predict the category
        prediction = model.predict([user_input])
        predicted_category = categories[prediction[0]]
        
        st.subheader(f"Prediction from {model_name}")
        st.write(f"The predicted category is: **{predicted_category}**")

        # Provide mathematical explanation for each model
        st.subheader("Mathematical Explanation")
        if model_name == 'Naive Bayes':
            st.markdown("""
            ### Naive Bayes Classifier
            The Naive Bayes classifier is based on Bayes' Theorem:
            \[
            P(C|X) = \\frac{P(X|C) \cdot P(C)}{P(X)}
            \]
            Where:
            - \(P(C|X)\) is the posterior probability of class \(C\) given the input \(X\).
            - \(P(X|C)\) is the likelihood of input \(X\) given class \(C\).
            - \(P(C)\) is the prior probability of class \(C\).
            - \(P(X)\) is the prior probability of input \(X\).

            The model assumes that the features (words in this case) are independent, which simplifies the computation of \(P(X|C)\).
            """)

        elif model_name == 'SVM':
            st.markdown("""
            ### Support Vector Machine (SVM)
            The Support Vector Machine classifier works by finding the hyperplane that best separates the classes in the feature space. The decision boundary is defined by:
            \[
            f(x) = w \cdot x + b
            \]
            Where:
            - \(w\) is the weight vector.
            - \(x\) is the input vector.
            - \(b\) is the bias term.

            The goal is to maximize the margin between the classes, which is the distance between the hyperplane and the nearest data points from each class.
            """)

        elif model_name == 'Logistic Regression':
            st.markdown("""
            ### Logistic Regression
            Logistic Regression predicts the probability that an input belongs to a particular class using the logistic function:
            \[
            P(C|X) = \\frac{1}{1 + e^{-(w \cdot x + b)}}
            \]
            Where:
            - \(P(C|X)\) is the probability of class \(C\) given the input \(X\).
            - \(w\) is the weight vector.
            - \(x\) is the input vector.
            - \(b\) is the bias term.

            The model is trained to minimize the logistic loss function, which is a measure of the difference between the predicted probabilities and the actual class labels.
            """)

st.markdown("""
### Summary
Text classification is a fundamental task in Natural Language Processing, allowing us to categorize text into predefined labels. The models showcased here—Naive Bayes, SVM, and Logistic Regression—are powerful tools for this purpose, each with its own mathematical foundation and strengths.
""")