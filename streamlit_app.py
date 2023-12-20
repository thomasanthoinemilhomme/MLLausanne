import os
import streamlit as st
import pandas as pd
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import gdown
import sentencepiece
# Function to download and load the model
def load_model():
    url = 'https://drive.google.com/uc?id=1dyrnb4lsirtcFFDnuu6-wlmfmTKmJu0u'
    output = 'my_model.zip'
    gdown.download(url, output, quiet=False)

    # Extract the model
    import zipfile
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('my_model')

    # Load the model
    model = CamembertForSequenceClassification.from_pretrained('my_model')
    return model
advice_for_level = {
    'A1': "Try to use more complex sentences and richer vocabulary.",
    'A2': "Incorporate varied grammatical structures and verb tenses.",
    'B1': "Use more idiomatic expressions and expand your range of vocabulary.",
    'B2': "Focus on refining your sentence structure and using more nuanced expressions.",
    'C1': "Practice using subjunctive moods and more sophisticated conjunctions.",
    'C2': "Polish your language by mastering subtle linguistic nuances and advanced vocabulary."
}

# Streamlit page configuration
st.set_page_config(page_title="Language Level Detector", page_icon="🇫🇷")

# Title and project description
st.title("Detecting Language Level in French Sentences")
st.markdown("## Project conducted by Team Lausanne with Leo Andrade and Thomas Anthoine-Milhomme")

# User input
user_input = st.text_input("Enter a French sentence to analyze its language level:")

# Load model (this should be done only once, you might want to optimize this part)
model = load_model()

# Tokenizer
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

# Predict function
def predict(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions

# Display result
if user_input:
    prediction = predict(user_input)
    difficulty_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}
    predicted_level = difficulty_mapping[torch.argmax(prediction).item()]
    st.write(f"The predicted language level for this sentence is: **{predicted_level}**")
if predicted_level in advice_for_level:
        st.write(f"Advice to reach the next level: {advice_for_level[predicted_level]}")
