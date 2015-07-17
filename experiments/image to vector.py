#!/usr/bin/env python
#Note. Master thesis: Learning to Predict the Age Rating of Movies by Robbert van Hintum.
#Experiment containing Image features. The image features are matched with the movies in the dataset The vectors are saved as Pickle files.

import csv
import os.path
import shutil
import os
import cPickle as pickle
import numpy
import scipy

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.externals import joblib

def read_data(f):
    data = []
    for row in csv.reader(open(f), delimiter=';'):
        if row and row[5] is not '':
            plottext = row[8]
            poster = row[5].split('/')[-1]
            target = row[4]
            data.append((plottext, poster, target))
    (X, postername, Ycat) = zip(*data)
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(Ycat)
    global labels
    labels = le.inverse_transform([0,1,2,3,4])
    return (X, Y, postername)

def load_image_vectors(foldername):
    data_feats = []
    data_files = []
    for folder in os.listdir(foldername):
        if os.path.exists(foldername+'/'+folder+'/'+'vgg_feats.pkl'):
            current = pickle.load(open(foldername+'/'+folder+'/'+'vgg_feats.pkl'))
            for item in current['feats']:
                data_feats.append(item)
            for item in current['files']:
                data_files.append(item)
    for i in range(len(data_files)):
        data_files[i] = data_files[i].split('/')[-1]
    return data_feats, data_files

image_vectors, image_files = load_image_vectors('../filmposters_out') #poster vectors folder
X, Y, postername = read_data('final_dataset-EVALUATION.csv') #read data set

X_poster = []
for item in postername:
    if item and item in image_files:
        X_poster.append(image_vectors[image_files.index(item)])
    else:
        X_poster.append(numpy.zeros(shape=((100,))))

X_poster = StandardScaler().fit_transform(X_poster) #scale the features of the posters

joblib.dump(X_poster, 'eval-posters.pkl') #dump the vectors into a Pickle file.
print 'dump succes'
