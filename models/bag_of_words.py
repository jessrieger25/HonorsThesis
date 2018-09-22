import matplotlib as mpl

mpl.use('GTK')
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import json
import re as re
import time
import progressbar
import sqlalchemy
import warnings
import matplotlib.pyplot as plt
from six.moves import zip, range
import sklearn

import os

warnings.filterwarnings('ignore')



print(sklearn.__version__)
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
from sklearn import preprocessing
from collections import Counter, OrderedDict
from nltk.corpus import stopwords
from nltk import SnowballStemmer
from nltk.tokenize import sent_tokenize

from wordcloud import WordCloud

class BagOfWords:

    def create_bag_of_words(self, sentences,
                            NGRAM_RANGE=(0, 1),
                            stop_words=None,
                            stem=False,
                            MIN_DF=0.05,
                            MAX_DF=0.95,
                            USE_IDF=False):
        ANALYZER = "word"  # unit of features are single words rather then phrases of words
        STRIP_ACCENTS = 'unicode'
        stemmer = nltk.SnowballStemmer("english")

        if stem:
            tokenize = lambda x: [stemmer.stem(i) for i in x.split()]
        else:
            tokenize = None
        vectorizer = CountVectorizer(analyzer=ANALYZER,
                                     tokenizer=tokenize,
                                     ngram_range=NGRAM_RANGE,
                                     stop_words=stop_words,
                                     strip_accents=STRIP_ACCENTS,
                                     min_df=MIN_DF,
                                     max_df=MAX_DF)

        bag_of_words = vectorizer.fit_transform(sentences)  # transform our corpus is a bag of words
        features = vectorizer.get_feature_names()

        # if you set IDF to true, it creates all of the following parameters:

        if USE_IDF:
            NORM = None  # turn on normalization flag
            SMOOTH_IDF = True  # prvents division by zero errors
            SUBLINEAR_IDF = True  # replace TF with 1 + log(TF)
            transformer = TfidfTransformer(norm=NORM, smooth_idf=SMOOTH_IDF, sublinear_tf=True)
            # get the bag-of-words from the vectorizer and
            # then use TFIDF to limit the tokens found throughout the text
            tfidf = transformer.fit_transform(bag_of_words)

            # if you set IDF to true this is what you get
            return tfidf, features
        else:
            return bag_of_words, features
            # what you get if it's false

    def get_word_counts(self, bag_of_words, feature_names):
        """
        Get the ordered word counts from a bag_of_words

        Parameters
        ----------
        bag_of_words: obj
            scipy sparse matrix from CounterVectorizer
        feature_names: ls
            list of words

        Returns
        -------
        word_counts: dict
            Dictionary of word counts
        """

        # convert bag of words to array
        np_bag_of_words = bag_of_words.toarray()

        # calculate word count.
        word_count = np.sum(np_bag_of_words, axis=0)

        # convert to flattened array.
        np_word_count = np.asarray(word_count).ravel()

        # create dict of words mapped to count of occurrences of each word.
        dict_word_counts = dict(zip(feature_names, np_word_count))

        # Create ordered dictionary
        orddict_word_counts = OrderedDict(sorted(dict_word_counts.items(), key=lambda x: x[1], reverse=True), )

        return orddict_word_counts

    def create_topics(self, tfidf, features, N_TOPICS=3, N_TOP_WORDS=5, ):
        """
        Given a matrix of features of text data generate topics

        Parameters
        -----------
        tfidf: scipy sparse matrix (this is a python package!)
            sparse matrix of text features
        N_TOPICS: int
            number of topics (default 10)
        N_TOP_WORDS: int
            number of top words to display in each topic (default 10)

        Returns
        -------
        ls_keywords: ls
            list of keywords for each topics
        doctopic: array
            numpy array with percentages of topic that fit each category
        N_TOPICS: int
            number of assumed topics
        N_TOP_WORDS: int
            Number of top words in a given topic.
        """

        with progressbar.ProgressBar(max_value=progressbar.UnknownLength) as bar:
            i = 0
            lda = LatentDirichletAllocation(n_topics=N_TOPICS,
                                            learning_method='online')  # create an object that will create 5 topics
            bar.update(i)
            i += 1
            doctopic = lda.fit_transform(tfidf)
            bar.update(i)
            i += 1
            ls_keywords = []
            for i, topic in enumerate(lda.components_):
                word_idx = np.argsort(topic)[::-1][:N_TOP_WORDS]
                keywords = ', '.join(features[i] for i in word_idx)
                ls_keywords.append(keywords)
                print(i, keywords)
                bar.update(i)
                i += 1

        return ls_keywords, doctopic
