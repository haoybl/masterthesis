#!/usr/bin/env python
# -\*- coding: utf-8 -\*-
#Note. Master thesis: Learning to Predict the Age Rating of Movies by Robbert van Hintum.
#Experiment containing Word Embeddings. The vectors are saved as Pickle files.
#The implementation of Word Embeddings was partly inspired on the tutorial of https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors
print(__doc__)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

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
from nltk.corpus import stopwords
from gensim.models import Word2Vec

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

def review_to_wordslist( review, remove_stopwords=False ):    
    letters_only = re.sub("[^a-zA-Z]", " ", review) 
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))                  
        words = [w for w in words if not w in stops]   
    return(words)

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordslist( raw_sentence, remove_stopwords ))
    return sentences

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set: 
            nwords += 1
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
       if counter%1000 == 0:
           print "Review %d of %d" % (counter, len(reviews))
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       counter += 1
    return reviewFeatureVecs

X, Y = read_data('development2-complete.csv') #read data set

X_test = X[:1900]
X_train = X[1900:]
Y_test = Y[:1900]
Y_train = Y[1900:]

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []

print 'Parsing sentences from training set'
for plot in X:
    plot = plot.encode('ascii', 'ignore')
    sentences += review_to_sentences(plot, tokenizer)

num_features = 300    # Word vector dimensionality                      
min_word_count = 10   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 15          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

print "Training model..."
#model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling) #enabling this means that the model is trained on the corpus of movie plots.

model = Word2Vec.load_word2vec_format("../GoogleNews-vectors-negative300.bin", binary=True) #enabling this means that a pre-trained model is loaded.

clean_train_reviews = []
for review in X_train:
    clean_train_reviews.append( review_to_wordslist( review, remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

joblib.dump(trainDataVecs, 'word_train.pkl') #dump the train vectors in a Pickle file

clean_test_reviews = []
for review in X_test:
    clean_test_reviews.append( review_to_wordslist( review, remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

joblib.dump(testDataVecs, 'word_test.pkl') #dump the train vectors in a Pickle file
