#streamlit app to input sentence and output the sentiment
import streamlit as st
import numpy as np
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow import keras 
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import PorterStemmer
import re

st.set_page_config(page_title="Spotify Review Analyser", page_icon="🎵", layout="centered", initial_sidebar_state="auto", menu_items=None)
import pandas as pd


@st.cache_resource
def load_data():
    data = pd.read_csv('Dataset/reviews_cleaned.csv')
    return data

@st.cache_resource
def load_model():
    model2 = keras.models.load_model('model.h5')
    model2.load_weights('model.h5')
    return model2

data = load_data()

model2 = load_model()

@st.cache_resource
def load_tokenizer():
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    Sentiment_text=data['Review'].astype(str)
    tokenizer.fit_on_texts(Sentiment_text.values)
    return tokenizer
tokenizer = load_tokenizer()
def predict_sentiment2(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps =PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus = [review]
    X = tokenizer.texts_to_sequences(corpus)
    X = pad_sequences(X, maxlen=316)
    sentiment = model2.predict(X,batch_size=1,verbose = 2)[0]
    if(np.argmax(sentiment) == 0):
        return("negative")
    elif (np.argmax(sentiment) == 1):
        return("neutral")
    elif (np.argmax(sentiment) == 2):
        return("positive")

def page():
    st.title("Spotify Review Analyser")
    #text input
    st.markdown("<span style='font-size:25px'><span style='color:#ed5f55'>Enter a review to predict the sentiment</span></span>", unsafe_allow_html=True)
    text = st.text_input("")
    st.write("")
    #make enter key hit submit button
    if st.button("Submit"):
        st.write("")
        st.write("")
        st.markdown("<span style='font-size:20px'>Predicted Sentiment:</span>", unsafe_allow_html=True)
        st.write("")
        if predict_sentiment2(text) == "positive":
            st.markdown("<span style='font-size:20px'><span style='color:#00ff00'>Positive</span></span>", unsafe_allow_html=True)
        elif predict_sentiment2(text) == "negative":
            st.markdown("<span style='font-size:20px'><span style='color:#ff0000'>Negative</span></span>", unsafe_allow_html=True)
        elif predict_sentiment2(text) == "neutral":
            st.markdown("<span style='font-size:20px'><span style='color:#ffff00'>Neutral</span></span>", unsafe_allow_html=True)
page()

    
