import numpy as np
import pandas as pd

#Loading the dataset
df = pd.read_csv('AmazonAlexa_ReviewsDataset.tsv',sep='\t') 

dataset = df[['verified_reviews','rating']]
dataset.columns = ['Review', 'Sentiment']


# Creating a new column sentiment based on overall ratings
def compute_sentiments(labels):
  sentiments = []
  for label in labels:
    if label > 3.0:
      sentiment = 1
    elif label <= 3.0:
      sentiment = 0
    sentiments.append(sentiment)
  return sentiments

dataset['Sentiment'] = compute_sentiments(dataset.Sentiment)

# check distribution of sentiments
#dataset['Sentiment'].value_counts()

# check for null values
# dataset.isnull().sum()

dataset['Review'].fillna(' ', inplace=True)

#Data Transformation

x = dataset['Review']
y = dataset['Sentiment']

import spacy
nlp = spacy.load('en_core_web_sm')

import string
punct = string.punctuation
# punct

from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS) # list of stopwords

class CustomTokenizerExample():
    def __init__(self):
        pass

    def text_data_cleaning(self,sentence):
        doc = nlp(sentence)                         # spaCy tokenize text & call doc components, in order

        tokens = [] # list of tokens
        for token in doc:
            if token.lemma_ != "-PRON-":
                temp = token.lemma_.lower().strip()
            else:
              temp = token.lower_
            tokens.append(temp)

        cleaned_tokens = []
        for token in tokens:
            if token not in stopwords and token not in punct:
                cleaned_tokens.append(token)
        return cleaned_tokens
    

# if root form of that word is not proper noun then it is going to convert that into lower form
# and if that word is a proper noun, then we are directly taking lower form,
# because there is no lemma for proper noun
# stopwords and punctuations removedv

# Feature Engineering (TF-IDF)

from sklearn.feature_extraction.text import TfidfVectorizer

token = CustomTokenizerExample()
tfidf = TfidfVectorizer(tokenizer= token.text_data_cleaning)
# tokenizer=text_data_cleaning, tokenization will be done according to this function

#Train the model

# Train_Test_Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = dataset.Sentiment, random_state = 0)
# x_train.shape, x_test.shape

# 2520 samples in training dataset and 630 in test dataset

# print(x.shape, x_train.shape, x_test.shape,y_train.shape, y_test.shape)

#Fit x_train and y_train

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

classifier = LinearSVC()
pipeline = Pipeline([('tfidf',tfidf), ('clf',classifier)])

# it will first do vectorization and then it will do classification

pipeline.fit(x_train, y_train)

#Check Model Performance

y_pred = pipeline.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# With pipeline, we don't need to prepare the dataset for testing (x_test)

# confusion_matrix

# confusion_matrix(y_test, y_pred)

# we are getting almost 91% accuracy

# classification_report
# print(classification_report(y_test, y_pred))

#Predict Sentiments using Model

prediction = pipeline.predict(["Alexa is bad for adults but good for kids"])

if (prediction == 1):
  print("Result: This review is positive")
else:
  print("Result: This review is negative")
