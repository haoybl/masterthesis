#!/usr/bin/env python
#Note. Master thesis: Learning to Predict the Age Rating of Movies by Robbert van Hintum.
#Experiment with Paragraph Vector. The vectors are saved as Pickle files.

# -\*- coding: utf-8 -\*-
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

import numpy as np
import scipy
import csv
import re
import string
import nltk
import nltk.data
import gensim
import random
from sklearn import metrics

from sklearn import preprocessing
from nltk.corpus import stopwords
from gensim.models import Doc2Vec
from sklearn.externals import joblib

def read_data(f):
    data = []
    for row in csv.reader(open(f), delimiter=';'):
        if row:
            plottext = row[8].decode('utf-8-sig')
            target = row[4]
            data.append((plottext, target))
    (X, Ycat) = zip(*data)
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(Ycat)
    global labels
    labels = le.inverse_transform([0,1,2,3,4])
    return (X, Y)

def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

def getVecs(model, corpus, size):
    vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

LabeledSentence = gensim.models.doc2vec.LabeledSentence

X, Y = read_data('development2-complete.csv')

X_test = X[:1900]
X_train = X[1900:]
Y_test = Y[:1900]
Y_train = Y[1900:]

X_train = cleanText(X_train)
X_test = cleanText(X_test)

X_train = labelizeReviews(X_train, 'TRAIN')
X_test = labelizeReviews(X_test, 'TEST')

size = 400 #amount of dimensions

model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=4)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=4)

model_dm.build_vocab(np.concatenate((X_train, X_test)))     #create vocabulary over train and test set
model_dbow.build_vocab(np.concatenate((X_train, X_test)))

all_train_reviews = np.array(X_train)
for epoch in range(10):
    perm = np.random.permutation(all_train_reviews.shape[0])
    model_dm.train(all_train_reviews[perm])
    model_dbow.train(all_train_reviews[perm])

train_vecs_dm = getVecs(model_dm, X_train, size)
train_vecs_dbow = getVecs(model_dbow, X_train, size)

train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow)) #concatenate train vectors of DM and DBOW

joblib.dump(train_vecs, 'pv_train.pkl' ) #save train vectors to Pickle file

#train over test set
X_test = np.array(X_test)

for epoch in range(10):
    perm = np.random.permutation(X_test.shape[0])
    model_dm.train(X_test[perm])
    model_dbow.train(X_test[perm])

#Construct vectors for test reviews
test_vecs_dm = getVecs(model_dm, X_test, size) 
test_vecs_dbow = getVecs(model_dbow, X_test, size)

test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow)) #concatenate test vectors of DM and DBOW

joblib.dump(test_vecs, 'pv_test.pkl' ) #save test vectors
