!pip install accelerate -U
!pip install transformers[torch]
!pip install sentencepiece
import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertForSequenceClassification.from_pretrained('https://huggingface.co/tommilhomme/languagelevel/tree/main/my_model')

st.title('French Sentence Difficulty Predictor')

# User input for the sentence
sentence = st.text_input('Enter a French sentence:')

if sentence:
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)

    # Predict the difficulty level
    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert logits to probabilities and then to the corresponding class
    probs = torch.nn.functional.softmax(logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()

    # Define a mapping for difficulty levels
    difficulty_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

    # Display the predicted difficulty level
    st.write(f'Predicted difficulty level: {difficulty_mapping[prediction]}')
