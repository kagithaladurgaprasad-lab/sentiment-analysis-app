import streamlit as st
import pickle
import re
import streamlit as st
st.title("Debug Mode Active")

# Load model
model = pickle.load(open("model.pkl", "rb"))

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
