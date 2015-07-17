# masterthesis
July 16th, 2015.
Master thesis: Learning to Predict the Age Rating of Movies by Robbert van Hintum.

Abstract:
Only few studies have conducted research on the prediction of age ratings of movies. Prior research focussed mainly on extracting material of the movie itself, such as stills and subtitles to determine the age rating. Additionally in those prior studies, a rule-driven approach was used instead of a more modern supervised learning task.

This study examines the possibility to predict the age rating of movies in a supervised learning task, using auxiliary materials of movies such as the movie plot, poster, genre and year of release. The experiments in this study demonstrated that the age ratings of movies can be predicted to some degree. The best model, containing of the movie plot, genre and year was able to improve the performance with 24% over the majority baseline.

Information:
All data and code that is used in this experiment can be found in this repository. An overview of what is located where:

OVERVIEW OF THE FOLDERS
- dataset [Contains the dataset, split test and train and development]
- experiments [Contains all experiments conducted in this research. These scripts save the vectors into Pickle files]
- final models [‘final.py’ contains the best performing model fully written out. ‘Other.py’ all final models. Different models can be made using ‘Other.py’ by adding or commenting the Pickle files from the features.]
- scraper [Contains the script that has been used to fetch the data set using the OMDb API]
- vectors [Contains the vectors that are created in the experiments by the feature extraction methods. These vectors can be imported in ‘Other.py’ to construct the final models.].

OVERVIEW OF THE FILES
- /DATASET
-   ‘test.csv’ [test set]
-   ‘train-development.cv’ [train and development set]
- /EXPERIMENTS
-   ‘Image to vector.py’ [script to match movie poster vectors to dataset]
-   ’N-grams.py’
-   ‘Paragraph Vector.py’
-   ‘pos_tagging.py’
-   ‘stemming+lemmatization.py’
-   ‘Word Embeddings.py’
-   final models
-   ‘final.py’ [best performing model fully written out]
-   ‘other.py’ [all models]
- /SCRAPER
-   ‘scraper_omdb.py’
-   ‘movie_input.csv’
- /VECTORS
-   ‘dev-posters.pkl’ [train and development set posters]
-   ‘eval-posters.pkl’ [test set posters]
-   ‘pv_test.pkl’	[test set Paragraph Vector]
-   ‘pv_train.pkl’	[train set Paragraph Vector]
-   ‘wordembeddings-dev.pkl’ [train and development set Word Embeddings]
-   ‘word_eval.pkl’ [test set Word Embeddings]

REQUIRED LIBRARIES
- Scikit-learn
- NLTK
- Gensim

REFERENCES
- The implementation of Word Embeddings was partly inspired on the tutorial of https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors
