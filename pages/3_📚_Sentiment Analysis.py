import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import time
st.title("Sentiment Analysis with Hugging Face Transformers")

st.image('data\sentimentanalysishotelgeneric-2048x803-1.jpg', caption="https://www.expressanalytics.com/blog/social-media-sentiment-analysis/")
st.markdown("""
## Sentiment Analysis Page

This page allows you to input text and view the predicted sentiment using a pre-trained model from Hugging Face's Transformers library.
""")

# Load the model and tokenizer from the local directory
model_dir = "experiments\models\distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Initialize the sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Text input
user_input = st.text_area("Enter text to analyze sentiment:", "Natural Language Processing is fascinating and very useful.")

# Submit button
if st.button('Analyze Sentiment'):
    with st.spinner('😃😐😠...'):
        time.sleep(3)
    if user_input:
        # Analyze the sentiment of the input text
        result = classifier(user_input)
        label = result[0]['label']
        score = result[0]['score']
        
        st.subheader("Sentiment Analysis Result")
        st.write(f"Label: **{label}**")
        st.write(f"Confidence Score: **{score:.4f}**")

# Add the explanation
st.markdown("""
### Explanation

The `transformers` library from Hugging Face provides pre-trained models that can be used for various NLP tasks, including sentiment analysis. The models are trained on large datasets and can classify text into different sentiment categories with high accuracy. The model name: **distilbert-base-uncased-finetuned-sst-2-english**

#### Example Usage

For example, if you input the text "I like to live my life.", the model will predict the sentiment based on the content of the text, such as "positive" or "negative".

#### How it Works

1. **Input Text**: The text to be analyzed is provided by the user.
2. **Model Inference**: The pre-trained model processes the text and generates a sentiment prediction.
3. **Output**: The predicted sentiment label and confidence score are displayed to the user.
""")