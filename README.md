# Spotify_review_analyser

It will be a web app that allows you to input a comment about spotify and it will analyse the comment and give you a sentiment score.

## Dataset

The dataset is from kaggle and it is a csv file as provided here. [Dataset](/Dataset/reviews.csv)

## Notebook

The notebook is the file that contains the code for the project. [Notebook](/Notebook/Notebook.ipynb)

## Steps involved in the project

1. Importing the dataset
2. Picking the required columns
3. Cleaning the data
4. Preprocessing the text
   1. Removing the punctuations
   2. Changing the text to lower case
   3. Whitespaces removal
   4. Removing the numbers
   5. Removing urls and html tags
   6. Removing the stopwords
   7. Lemmatization
5. Exploratory Data Analysis
   1. Histogram of the rating
   2. Histogram after categorizing the rating

## Work to be done

1. More EDAs
   1. Wordcloud
2. Sentiment analysis
3. Deployment