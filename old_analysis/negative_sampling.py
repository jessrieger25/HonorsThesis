import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
from sklearn import preprocessing
from sklearn.manifold import TSNE
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import euclidean_distances
import nltk
from collections import Counter
import random

# Lemmetization?

# Load data
eng_stopwords = stopwords.words('english')
corpus_raw = ""
with open("/Users/Jess/PycharmProjects/Honors_Thesis_2/time_machine_skip_gram.txt", "r") as time:
    for line in time.readlines():
        corpus_raw += line.replace('\n', " ")

# Convert to Lower
corpus_raw = corpus_raw.lower()

# Data prep
custom_sent_tokenizer = PunktSentenceTokenizer(corpus_raw)
tokenized_sentences = custom_sent_tokenizer.tokenize(corpus_raw)

sen_word_token = []
for sen in tokenized_sentences:
    temp = nltk.word_tokenize(sen)
    new_temp = []
    for one in temp:
        if one not in eng_stopwords and one not in [".", "!", "?"]:
            new_temp.append(one)
    sen_word_token.append(new_temp)

words = set()
print(sen_word_token)
for one in sen_word_token:
    for internal in one:
        words.add(internal)
# words = set(sen_word_token) # so that all duplicate words are removed
word2int = {}
int2word = {}
vocab_size = len(words) # gives the total number of unique words
for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# Vectorization
window_tuples = []
WINDOW_SIZE = 5
for sentence in sen_word_token:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                window_tuples.append([word, nb_word])


# Start Tutorial
threshold = 1e-5
word_counts = Counter(words)
total_count = len(words)
freqs = {word: count/total_count for word, count in word_counts.items()}
p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}
train_words = [word for word in words if random.random() < (1 - p_drop[word])]

inputs = tf.placeholder(tf.int32, [None], name='inputs')
labels = tf.placeholder(tf.int32, [None, None], name='labels')

n_vocab = len(words)
n_embedding =  300
embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
embed = tf.nn.embedding_lookup(embedding, inputs)

# Number of negative labels to sample
n_sampled = 100
softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding)))
softmax_b = tf.Variable(tf.zeros(n_vocab), name="softmax_bias")

# Calculate the loss using negative sampling
loss = tf.nn.sampled_softmax_loss(
    weights=softmax_w,
    biases=softmax_b,
    labels=labels,
    inputs=embed,
    num_sampled=n_sampled,
    num_classes=n_vocab)

cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer().minimize(cost)

sess = tf.Session()
batches = np.random.choice(len(train_words), size=200)
print(batches)
rand_x = batches
rand_y = batches
# batches = np.get_batches(train_words, batch_size, window_size)
for x in range(0, len(batches)):
    print(x)
    feed = {inputs: rand_x[x], labels: np.array(rand_y[x])[:, None]}
    train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
embed_mat = sess.run(embedding)
viz_words = 500
tsne = TSNE()
embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])
fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)