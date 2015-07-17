#!/usr/bin/env python
#Note. Master thesis: Learning to Predict the Age Rating of Movies by Robbert van Hintum.
#Final model containing the features: N-grams, Word Embeddings, Genre and Year on SVM and Ridge regression.
#The implementation of Word Embeddings was partly inspired on the tutorial of https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors

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
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from gensim.models import Word2Vec

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
       if counter%1000. == 0:
           print "Review %d of %d" % (counter, len(reviews))
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       counter += 1
    return reviewFeatureVecs

X_train, Y_train, genre_train, year_train = read_data('../development2-complete.csv') #link to the train data
X_test, Y_test, genre_test, year_test = read_data_eval('../final_dataset-EVALUATION.csv') #link to the test data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#Parameter settings for Word Embeddings
num_features = 300                                          # Word vector dimensionality (always 300 when pre-trained model is loaded)                     

#Parameter settings for N-grams
ngram = (1, 3)                                              #Range of N-grams
mdf = 1                                                     #Minimum document frequency
stop = 'english'                                            #Stop words are filtered from English language

#Parameters for classifiers
dictweight = {0:0.94, 1:0.996, 2:0.81, 3:0.82, 4:0.42}      #Example weighting for Classifier

print "Training model..."
model = Word2Vec.load_word2vec_format("../../../../GoogleNews-vectors-negative300.bin", binary=True) #load the pre-trained model on GoogleNews.(download: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?pli=1)

clean_train_reviews = []
for review in X_train:
    clean_train_reviews.append( review_to_wordslist( review, remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features ) #Word Embedding train vectors are saved here.

tfidf = TfidfVectorizer(ngram_range=ngram, min_df=mdf, stop_words=stop)
tfidftrainvec = tfidf.fit_transform(X_train, Y_train)                       #N-gram train vectors are saved here

print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in X_test:
    clean_test_reviews.append( review_to_wordslist( review, remove_stopwords=True )) 

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features ) #Word Embeddings test vectors are saved here.
tfidftestvec = tfidf.transform(X_test)                                      #N-gram test vectors are saved here

X_train_conc = scipy.sparse.hstack((tfidftrainvec, trainDataVecs, genre_train, year_train)) #Train vectors of N-grams, Word Embeddings, Genre and Year are concatenated
X_test_conc = scipy.sparse.hstack((tfidftestvec, testDataVecs, genre_test, year_test)) #Test vectors of N-grams, Word Embeddings, Genre and Year are concatenated

#The concatenated features are fed to the classifiers.
#Classifiers are tuned for performance as can be read in chapter 4 section 7.
print "Fitting SGDclassifier SVM labeled training data..."
clf = SGDClassifier(loss='hinge', n_jobs=-1, class_weight=dictweight, alpha=0.0001, penalty='elasticnet').fit( X_train_conc, Y_train )
predicted = clf.predict( X_test_conc )
print 'Evaluation metric SGD SVM:', metrics.classification_report(Y_test, predicted, target_names=labels)
print 'Accuracy score:', metrics.accuracy_score(Y_test, predicted)

print 'Fitting Ridgeclassifier labeled training data...'
clf = RidgeClassifier(alpha=0.01, solver='cholesky').fit( X_train_conc, Y_train )
predicted = clf.predict( X_test_conc )
print 'Evaluation metric Ridgeclassifier:', metrics.classification_report(Y_test, predicted, target_names=labels)
print 'Accuracy score:', metrics.accuracy_score(Y_test, predicted)
