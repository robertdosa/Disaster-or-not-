# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 18:49:10 2020

@author: Robert Dosa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Data preprocessing

print('Data preprocessing...')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Deleting 'location' column and filling the missing values

train = train.drop('location', axis=1)
train['keyword'].fillna(value='Unknown', inplace=True)

# Extracting metadata

train['length_tweet'] = train['text'].apply(len)
train['length_keyword'] = train['keyword'].apply(len)


def count_ht(tweet):
    """
    Function to count the hashtags in each tweet.
    """
    ht = 0
    for char in tweet:
        if char == '#':
            ht +=1
    return ht



def text_process(tw):
    """
    :tw: Text(string) 
    Returns a list for each tweet as a list of each word count in the tweet.
    """
    
    nopunc = [char for char in tw if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower()
            not in stopwords.words('english')]
    
# Adding metadata to the data frame
    
train['ht_number'] = train['text'].apply(count_ht)    

# Text preprocessing with BOW and TF-IDF

bowt = CountVectorizer(analyzer=text_process)
bowt.fit(train['text'])
tweet_bow = bowt.transform(train['text'])
tf_idf_tw = TfidfTransformer()
tf_idf_tw.fit(tweet_bow)

text_tfidf = tf_idf_tw.transform(tweet_bow) # Sparse matrix

# Creating a new DF from the sparse matrix and concatenating with the original
# data

tf_idf_df = pd.DataFrame(text_tfidf.toarray())
pr_train = pd.concat([train.drop(labels=['text'], axis=1), tf_idf_df], axis=1)

# Getting dummy variables for the 'keyword' column

pr_train = pd.get_dummies(data=pr_train, columns=['keyword'], drop_first=True)
pr_train = pr_train.drop(labels=['id'], axis=1)

# Scaling

sc = StandardScaler()
sc.fit(pr_train.drop(labels=['target'], axis=1))
train_sc = sc.transform(pr_train.drop('target', axis=1))
pr_train = pd.concat([pr_train['target'], pd.DataFrame(train_sc)], axis=1)

# Model building

X = pr_train.drop(labels=['target'], axis=1)
y = pr_train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creating a neural network

print('Initializing neural network...')

classifier = Sequential()
classifier.add(Dense(8000, kernel_initializer='uniform', activation='relu',
                     input_dim=26697))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='rmsprop', loss='binary_crossentropy',
                   metrics=['accuracy'])

print('Fitting the data...')
classifier.fit(X_train, y_train, batch_size=20, epochs=100)
classifier.save('with_text_ann.h5')


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

