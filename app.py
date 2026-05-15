import streamlit as st
import pickle
import re
import streamlit as st


# Load model
import streamlit as st
import os
import pickle

# load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# rest of your app code below
st.title("Sentiment Analysis App")
# Load vectorizer
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Title
st.title("Restaurant Review Sentiment Analysis")

# User Input
review = st.text_area("Enter Your Review")

# Prediction Button
if st.button("Predict"):

    # Clean text
    cleaned_review = clean_text(review)

    # Convert into vector
    vector_input = tfidf.transform([cleaned_review])

    # Predict
    prediction = model.predict(vector_input)

    # Output
    if prediction[0] == 1:
        st.success("Positive Review 😊")

    else:
        st.error("Negative Review 😞")
