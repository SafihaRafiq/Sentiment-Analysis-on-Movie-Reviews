from pyexpat import model
import Scripts.streamlit as st
from numpy import vectorize

st.title("Movie Review Sentiment Analysis")

review_input = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment"):
    review_vector = vectorize.transform([review_input.lower()])
    prediction = model.predict(review_vector)
    st.write(f"Sentiment: {prediction[0].capitalize()}")
