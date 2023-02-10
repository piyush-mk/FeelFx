#streamlit app to input sentence and output the sentiment
from imp import load_module
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
st.set_page_config(page_title="Spotify Review Analyser", page_icon="🎵", layout="wide", initial_sidebar_state="auto", menu_items=None)
import pandas as pd

#title of the app
st.title("Spotify Review Analyser")
#text input
st.subheader("Enter your review")
text = st.text_input("")
st.write("")
#function to predict the sentiment using the model saved weights
def predict_sentiment(text):
    #load the model
    model = load_module('model.h5')
    from nltk.stem import PorterStemmer
    ps =PorterStemmer()
    def predict_sentiment(review):
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus = [review]
        X = tokenizer.texts_to_sequences(corpus)
        X = pad_sequences(X, maxlen=316)
        sentiment = model.predict(X,batch_size=1,verbose = 2)[0]
        if(np.argmax(sentiment) == 0):
            return("negative")
        elif (np.argmax(sentiment) == 1):
            return("neutral")
        elif (np.argmax(sentiment) == 2):
            return("positive")