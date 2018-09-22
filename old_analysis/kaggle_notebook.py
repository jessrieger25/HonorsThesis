# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


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

print(os.listdir("../input"))

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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

""" Loading the Data:

    This loads the data into the variable sentences and tokenizes it by sentence. 

    """
filename = '../input/TimeMachine1.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
sentences = sent_tokenize(text)
print(sentences)

# This is from the kaggle notebook
# filename = '../input/tester-page1/sample1.txt'
# file = open(filename, 'rt')
# text = file.read()
# file.close()

# sentences = sent_tokenize(text)
# sentences

""" Preprocessing the Data:

    This removes unnecessary characters and preps the data for analysis.

    """

""" Additional Prep for OCR'd Text

    This must be done to fix common errors made during OCR, such as fixing the continuation of words onto
    the next line.

    """

RE_PREPROCESS = re.compile(r"-\s|-\n|\’\s")  # the regular expressions that matches all non-characters
sentences = np.array([re.sub(RE_PREPROCESS, '', sen) for sen in sentences])
RE_PREPROCESS = re.compile(r"~")  # the regular expressions that matches all non-characters
sentences = np.array([re.sub(RE_PREPROCESS, '-', sen) for sen in sentences])
sentences = [sen.lower() for sen in sentences]
RE_PREPROCESS = re.compile(r"\n")  # the regular expressions that matches all non-characters
sentences = np.array([re.sub(RE_PREPROCESS, ' ', sen) for sen in sentences])
RE_PREPROCESS = re.compile(r'\.+|\'+|,+|\(+|\)+|\!+|\?+')  # the regular expressions that matches all non-characters
sentences = np.array([re.sub(RE_PREPROCESS, '', sen) for sen in sentences])

""" Basic Character Removal:

    This removes all unwanted characters, such as punctuation and apostropes.
    """

RE_PREPROCESS = re.compile(r'\W+|\d+')
sentences = np.array([re.sub(RE_PREPROCESS, ' ', sentence).lower() for sentence in sentences])

""" Removing Stop Words:

    Accessing NLTK english stopwords. We can always add custom words
    to this list to improve its effectiveness.
    """
eng_stopwords = stopwords.words('english')

""" Functions for Analysis:

    These functions are used to perform the actual analysis such as creating bags of words and getting 
    word counts.
    """


def create_bag_of_words(sentences,
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


def get_word_counts(bag_of_words, feature_names):
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


def create_topics(tfidf, features, N_TOPICS=3, N_TOP_WORDS=5, ):
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


""" Testing the Functions:

    Examples of how to use the functions and displaying their results. 
    """
# Basic call to create bag of words with english stop words removed.
bag_of_words, features = create_bag_of_words(sentences, stop_words=eng_stopwords, stem=True)

# Diplaying the bag of words and the features
print(bag_of_words)
print(features)

# Basic call to get word counts without additional parameters.
word_counts = get_word_counts(bag_of_words, features)

# Displaying the word counts
print(word_counts)

# Call to create topics from the words present in the document.
processed_keywords, processed_doctopic = create_topics(bag_of_words, features)

# Displaying results:
print(processed_doctopic.shape)
print(processed_keywords)

# Call with all parameters
# look for 10 topics, include 4 words in each. We can play with this to see where we get the best
# results.
processed_keywords, processed_doctopic = create_topics(bag_of_words,
                                                       features,
                                                       N_TOPICS=10,
                                                       N_TOP_WORDS=4)

""" Aside For Fun:

    This is a cool way to display bags of words that we could maybe use for presentations or papers.
    """
# Creating a cloud
list_from_sentences = str(sentences)
word_cloud = WordCloud().generate(list_from_sentences)

# Display the generated image (this uses matplotlib)
plt.imshow(word_cloud)
plt.axis("off")

""" Interesting Visualizations:

    This is a section for new and different ways we find to display the data.
    """

# Pie Chart Example

labels = []
sizes = []
for key, value in word_counts.items():
    sizes.append((value / 51) * 100)
    labels.append(key)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:

# Creating the list for the pie chart

explode_list = list()
for i in range(0, len(labels)):
    if i == 2:
        explode_list.append(0.1)
    else:
        explode_list.append(0)

explode = (explode_list)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

""" Part of Speech Tagging:

    Tokenizer - PunktSentence Tokenizer - unsupervised machine learning training algorithm.

    Chunking: This allows you to parse out common sequences of parts of speech and isolate them.
    """

from nltk.tokenize import PunktSentenceTokenizer


def pos_tagging():
    # eng_stopwords = stopwords.words('english')

    # Declared variables
    text_parsed = ""
    text_test_parsed = ""

    # Sample text 1: For Training
    filename = '../input/TimeMachine1.txt'
    file = open(filename, 'rt')
    text = file.read()
    file.close()

    for i in text:
        if i not in eng_stopwords:
            text_parsed += i
    sentences_training = text_parsed

    # Sample text 2: For Testing
    filename = '../input/TimeMachine2.txt'
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    for i in text:
        if i not in eng_stopwords:
            text_test_parsed += i
    sentences_testing = text_test_parsed

    custom_sent_tokenizer = PunktSentenceTokenizer(sentences_training)

    tokenized = custom_sent_tokenizer.tokenize(sentences_testing)

    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunk_gram = r"""Chunk: {<RB.?>*<VB.?>*<NNP><NN>?}
                                    }<VB.?|IN|DT|TO>+{"""

            chunk_parser = nltk.RegexpParser(chunk_gram)
            chunked = chunk_parser.parse(tagged)
            print(chunked)
    except Exception as e:
        print(str(e))


pos_tagging()

""" WORDNET: for finding synonmys, antonyms, etc...
    lemmas = syns"""
from nltk.corpus import wordnet

syns = wordnet.synsets("program")
print(syns[0])
print(syns[0].lemmas)
print(syns[0].lemmas()[0].name())
print(syns[0].definition())

synonyms = []
antonyms = []

for syn in wordnet.synsets('good'):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(synonyms)
print(antonyms)





