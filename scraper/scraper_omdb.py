#Note. Master thesis: Learning to Predict the Age Rating of Movies by Robbert van Hintum.
#Scraper to extract the age rating, genre and poster from OMDb. Inputfile is a list of movies (.csv) extracted from the Bamman et al. (2013) dataset.
import omdb
import sys
import os
import os.path
import string
import re
import codecs

def scrape(t, y):
    x = omdb.get(title= t, year=y, fullplot=True, tomatoes=False)
    if x:
        plot = x['plot']
        age = x['rated']
        poster = x['poster']
        imdbid = x['imdb_id']
        genre = x['genre']
        language = x['language']
        return '%s; %s; %d; %s; %s; %s' % (imdbid, t, y, genre, age, poster)
    else:
        pass
    
inputfile = 'movie_input4.csv'
outputfile = 'output4.txt'

with codecs.open(inputfile, 'r', encoding='utf-8') as rf:
    inputmovie = rf.readlines()
rf.close()

outputtext = ''
count = 0
writecount = 0
#new_array = inputmovie[0].split('\r')

with codecs.open(outputfile, 'w', encoding='utf-8') as wf:
    for line in inputmovie:
        f = line.split(';')
        #inputdate = f[2].split('-')[2]
        #inputdate = '20%s' % (inputdate)
        s = scrape(unicode(f[1]),int(f[2]))
        outputtext += '%d;%s\n' % (int(f[0]), s)
        writecount += 1
        if writecount == 200 or writecount == len(inputmovie):
            wf.writelines(outputtext)
            writecount = 0
            outputtext = ''
        count += 1
        print count

wf.close()

print '%d lines have been processed' % (count)
