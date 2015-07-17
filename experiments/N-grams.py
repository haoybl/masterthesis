#!/usr/bin/env python
#Note. Master thesis: Learning to Predict the Age Rating of Movies by Robbert van Hintum.
#Experiment containing N-grams and calculation of the baseline on Ridge regression and SVM.
#The implementation of Word Embeddings was partly inspired on the tutorial of https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors

import numpy as np
import csv

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from itertools import repeat

def read_data(f):
    data = []
    for row in csv.reader(open(f), delimiter=';'):
        if row:
            plottext = row[8]
            target = row[4]
            data.append((plottext, target))
    (X, Ycat) = zip(*data)
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(Ycat)
    global baseline
    baseline = le.transform(['R'])
    global labels
    labels = le.inverse_transform([0,1,2,3,4])
    return (X, Y)

X, Y = read_data('../development2-complete.csv')

#This can be used to import stemmed, lematized or POS-tag prepared plots.
#new_X = []
#with open('pos_tag.txt', 'r') as rf:
#    for line in rf:
#        new_X.append(line.strip('\n'))

X_test = X[:1900]
X_train = X[1900:]
Y_test = Y[:1900]
Y_train = Y[1900:]

#Calculate the baseline
maj_baseline = [4]*len(Y_test)
print 'Baseline:'
print 'Evaluation', metrics.classification_report(Y_test, maj_baseline, target_names=labels)
print 'Accuracy score:', accuracy_score(Y_test, maj_baseline)

#Parameter settings for N-grams
ngram = (1, 3)                                              #Range of N-grams
mdf = 1                                                     #Minimum document frequency
stop = 'english'                                            #Stop words are filtered from English language

#Parameters for classifiers
dictweight = {0:0.94, 1:0.996, 2:0.81, 3:0.82, 4:0.42}      #Example weighting for Classifier

tfidf = TfidfVectorizer(ngram_range=ngram, min_df=mdf, stop_words=stop)
tfidftrainvec = tfidf.fit_transform(X_train, Y_train)
tfidftestvec = tfidf.transform(X_test)  

#Classifiers are tuned for performance as can be read in chapter 4 section 7.
print "Fitting SGDclassifier SVM labeled training data..."
clf = SGDClassifier(loss='hinge', n_jobs=-1, class_weight=dictweight, alpha=0.0001, penalty='elasticnet').fit( tfidftrainvec, Y_train )
predicted = clf.predict( tfidftestvec )
print 'Evaluation metric SGD SVM:', metrics.classification_report(Y_test, predicted, target_names=labels)
print 'Accuracy score:', metrics.accuracy_score(Y_test, predicted)

print 'Fitting Ridgeclassifier labeled training data...'
clf = RidgeClassifier(alpha=0.01, solver='cholesky').fit( tfidftrainvec, Y_train )
predicted = clf.predict( tfidftestvec )
print 'Evaluation metric Ridgeclassifier:', metrics.classification_report(Y_test, predicted, target_names=labels)
print 'Accuracy score:', metrics.accuracy_score(Y_test, predicted)
