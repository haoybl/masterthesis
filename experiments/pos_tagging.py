#!/usr/bin/env python
# -\*- coding: utf-8 -\*-
#Note. Master thesis: Learning to Predict the Age Rating of Movies by Robbert van Hintum.
#Experiment where movie reviews are marked with a POS-tag and saved to a .txt file.
import numpy as np
import csv
import nltk
import string

from nltk import word_tokenize
from nltk.corpus import brown
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn import preprocessing
from sklearn import metrics


def read_data(f):
    data = []
    for row in csv.reader(open(f), delimiter=';'):
        if row:
            plottext = row[8].decode('utf-8-sig')
            target = row[4]
            data.append((plottext, target))
    (X, Ycat) = zip(*data)
    Y = preprocessing.LabelEncoder().fit_transform(Ycat)
    return (X, Y)

X, Y = read_data('development2-complete.csv')

#For stemming of the plots, this can be activated.
#X_stemmed = []
#for item in X:
#    sentence = ''
#    item = item.encode('ascii', 'ignore')
#    item = item.translate(string.maketrans("",""), string.punctuation)
#    for word in item.split(" "):
#        sentence += PorterStemmer().stem_word(word).lower()
#        sentence += ' '
#    X_stemmed.append(sentence)

#For lemmatizing of the plots, this can be activated.
#X_lemmatized = []
#for item in X:
#    sentence = ''
#    item = item.encode('ascii', 'ignore')
#    item = item.translate(string.maketrans("",""), string.punctuation)
#    for word in item.split(' '):
#        sentence += WordNetLemmatizer().lemmatize(word).lower()
#        sentence += ' '
#    X_lemmatized.append(sentence)

#this is the process of adding POS-tags to the words.
for i in X:                             #for i in X_lemmatized or X_stemmed to add lemmatizing or stemming.
    i = i.encode('ascii', 'ignore')
    tokenplot = ' '
    i = i.translate(string.maketrans("",""), string.punctuation)
    for z in nltk.pos_tag(word_tokenize(i)):
        tokenplot += '%s_%s ' % (z[0].lower(), z[1])
    new_X.append(tokenplot)

with open ('pos_tag.txt', 'w') as wf:   #file will be saved as .txt file and can be imported in the N-gram script.
    for item in new_X:
        wf.write(item)
        wf.write('\n')
