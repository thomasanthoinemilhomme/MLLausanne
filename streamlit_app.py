import os
import streamlit as st
import pandas as pd
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import gdown

# Function to download and load the model
def load_model():
    url = 'https://drive.google.com/file/d/1dyrnb4lsirtcFFDnuu6-wlmfmTKmJu0u/view?usp=drive_link'
    output = 'my_model.zip'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    # Check if the downloaded file is a zip file
    if not zipfile.is_zipfile(output):
        raise Exception("Downloaded file is not a zip file or download failed")

    # Extract the model
    try:
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('my_model')
    except zipfile.BadZipFile:
        raise Exception("Failed to unzip the model file. The file might be corrupted.")
    model = CamembertForSequenceClassification.from_pretrained('my_model')
    return model

# Streamlit page configuration
st.set_page_config(page_title="Language Level Detector", page_icon="🇫🇷")

# Title and project description
st.title("Detecting Language Level in French Sentences")
st.markdown("## Project conducted by Team Lausanne with Leo Andrad and Thomas Anthoine-Milhomme")

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
