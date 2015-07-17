#!/usr/bin/env python
#Note. Master thesis: Learning to Predict the Age Rating of Movies.
#Final model containing containing all features. Features can be gradually added or left out by commenting out in the concatenation process.
#the vectors for Word Embeddings and Posters are imported using Pickle.

import numpy as np
import matplotlib.pyplot as plt
import scipy
import csv
import re
import string
import nltk
import nltk.data

from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

def read_data(f):
    data = []
    for row in csv.reader(open(f), delimiter=';'):
        if row:
            plottext = row[8].decode('utf-8-sig')
            genre = { 'genre': row[3] }
            year = row[2]
            target = row[4]
            data.append((plottext,genre,target,year))
    (X, genre, Ycat,year) = zip(*data)
    year = np.array(year, dtype='float')
    global yearscale
    yearscale = preprocessing.StandardScaler()
    year = yearscale.fit_transform(year).reshape((-1,1))
    global le
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(Ycat)
    global labels
    labels = le.inverse_transform([0,1,2,3,4])
    global dv
    dv = DictVectorizer(sparse=False)
    genre = dv.fit_transform(genre)
    return (X, Y, genre, year)

def read_data_eval(f):
    data = []
    for row in csv.reader(open(f), delimiter=';'):
        if row:
            plottext = row[8].decode('utf-8-sig')
            genre = { 'genre': row[3] }
            year = row[2]
            target = row[4]
            data.append((plottext,genre,target,year))
    (X, genre, Ycat,year) = zip(*data)
    year = np.array(year, dtype='float')
    year = yearscale.transform(year).reshape((-1,1))
    Y = le.transform(Ycat)
    genre = dv.transform(genre)
    return (X, Y, genre, year)


X_train, Y_train, genre_train, year_train = read_data('../development2-complete.csv') #link to the train data
X_test, Y_test, genre_test, year_test = read_data_eval('final_dataset-EVALUATION.csv') #link to the test data

wordvec = joblib.load('../wordembeddings-dev.pkl')          #import Word Embeddings vectors
wordvec_test = joblib.load('../word_eval.pkl')
posters_train = joblib.load('../dev-posters.pkl')           #import Poster vectors
posters_test = joblib.load('../eval-posters.pkl')

#Parameter settings for N-grams
ngram = (1, 3)                                              #Range of N-grams
mdf = 1                                                     #Minimum document frequency
stop = 'english'                                            #Stop words are filtered from English language

tfidf = TfidfVectorizer(ngram_range=ngram, min_df=mdf, stop_words=stop)
tfidftrainvec = tfidf.fit_transform(X_train, Y_train)

#features are concatenated. 
X_train_conc = scipy.sparse.hstack((tfidftrainvec, wordvec, genre_train, year_train, posters_train)) #all features are enabled now, features can be left out or be added gradually.  

tfidftestvec = tfidf.transform(X_test)
X_test_conc = scipy.sparse.hstack((tfidftestvec, wordvec_test, genre_test, year_test, posters_test)) #all features are enabled now, features can be left out or be added gradually.

#note that different combinations of features require different optimalization parameters for the best performance. See chapter 4 section 7 for details.
print "Fitting SGDclassifier SVM labeled training data..."
clf = SGDClassifier(loss='hinge', n_jobs=-1, alpha=0.0001, penalty='elasticnet').fit( X_train_conc, Y_train )
predicted = clf.predict( X_test_conc )
print 'Evaluation metric SGD SVM:', metrics.classification_report(Y_test, predicted, target_names=labels)
print 'Accuracy score:', metrics.accuracy_score(Y_test, predicted)

print 'Fitting Ridgeclassifier labeled training data...'
clf = RidgeClassifier(alpha=0.01, solver='cholesky').fit( X_train_conc, Y_train )
predicted = clf.predict( X_test_conc )
print 'Evaluation metric Ridgeclassifier:', metrics.classification_report(Y_test, predicted, target_names=labels)
print 'Accuracy score:', metrics.accuracy_score(Y_test, predicted)
