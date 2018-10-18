import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
from .word_prep import WordPrep
import objgraph
import random
import nltk


class SkipGram:

    def __init__(self, words, word2int, keywords):
        self.keywords = self.adjust_keywords(keywords)
        self.word2int = self.convert_phrases(word2int)
        self.words = self.group_phrases(words)
        self.vocab_size = len(self.word2int)
        self.vectors = []

        # Training variables
        self.window_tuples = []
        self.WINDOW_SIZE = 3
        self.x_train = []  # input word
        self.y_train = []  # output word

        # Model Variables
        self.EMBEDDING_DIM = 3  # you can choose your own number

    def group_phrases(self, word_list):
        for keyword in self.keywords:
            if "_" in keyword:
                keyword_split = keyword.split("_")
                first_word = keyword_split[0]
                num_words = len(keyword_split)
                index = 0
                while index < len(word_list):
                    if index != 0:
                        index = index + 1

                    try:
                        found_index = word_list[index:].index(first_word)
                        index = index + found_index
                        new_string = word_list[index]
                        found = True
                        if index + num_words < len(word_list):
                            for num in range(0, num_words):
                                if keyword_split[num] != word_list[index + num]:
                                    found = False
                                    break
                                elif num != 0:
                                    new_string += "_" + word_list[index + num]
                            if found is True:
                                word_list[index] = new_string
                                for other_indices in range(index + 1, index + num_words):
                                    word_list.pop(other_indices)
                    except ValueError as e:
                        index = len(word_list) + 1

        return word_list

    def convert_phrases(self, word2int_dict):
        new_dictionary = {}
        for keyword in self.keywords:
            if "_" in keyword:
                word2int_dict[keyword] = len(word2int_dict)
        return word2int_dict

    def adjust_keywords(self, keywords):
        new_keywords = {}
        for keyword, index in keywords.items():
            new_keyword = keyword
            if " " in keyword:
                new_keyword = keyword.replace(" ", "_")
            new_keywords[new_keyword] = keywords[keyword]
        return new_keywords

    def make_training_window_tuples(self):
        # Vectorization
        for word_index in range(0, len(self.words)):
            for nb_word in self.words[
                           max(word_index - self.WINDOW_SIZE, 0): min(word_index + self.WINDOW_SIZE, len(self.words)) + 1]:
                if nb_word != self.words[word_index]:
                    self.window_tuples.append([self.words[word_index], nb_word])

    # function to convert numbers to one hot vectors
    def to_one_hot(self, data_point_index):
        temp = np.zeros(self.vocab_size)
        temp[data_point_index] = 1
        return temp

    def prepare_training_data_skipgram(self, tuple_group):

        for data_word in tuple_group:
            self.x_train = []  # input word
            self.y_train = []  # output word
            self.x_train.append(self.to_one_hot(self.word2int[data_word[0].lower().strip()]))
            self.y_train.append(self.to_one_hot(self.word2int[data_word[1].lower().strip()]))

        # convert them to numpy arrays
        self.x_train = np.asarray(self.x_train)
        self.y_train = np.asarray(self.y_train)

    def make_skipgram(self, sess, train_step, cross_entropy_loss):
        n_iters = 1000
        # train for n_iter iterations
        for _ in range(n_iters):
            sess.run(train_step, feed_dict={self.x: self.x_train, self.y_label: self.y_train})
            print('loss is : ', sess.run(cross_entropy_loss, feed_dict={self.x: self.x_train, self.y_label: self.y_train}))

        print(sess.run(self.W1))
        print('----------')
        print(sess.run(self.b1))
        print('----------')

        self.vectors = sess.run(self.W1 + self.b1)
        print(len(self.vectors))

    def random_analysis(self):
        print("Hi")
        # print(self.int2word[self.find_closest(self.word2int['time'])])
        # print(self.int2word[self.find_closest(self.word2int['fire'])])
        #
        # weights = self.vectors
        #
        # distance_matrix = euclidean_distances(weights)
        # print(distance_matrix.shape)
        #
        # similar_words = {
        # search_term: [self.int2word[idx] for idx in distance_matrix[self.word2int[search_term] - 1].argsort()[1:6] + 1]
        # for search_term in ['fire', 'traveller', 'time']}
        #
        # print(similar_words)

    def euclidean_dist(self, vec1, vec2):
        return np.sqrt(np.sum((vec1-vec2)**2))


    def find_closest(self, word_index):
        min_dist = 10000 # to act like positive infinity
        min_index = -1
        query_vector = self.vectors[word_index]
        for index, vector in enumerate(self.vectors):
            if self.euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
                min_dist = self.euclidean_dist(vector, query_vector)
                min_index = index
        return min_index

    def run(self):
        self.make_training_window_tuples()
        self.x = tf.placeholder(tf.float32, shape=(None, self.vocab_size))
        self.y_label = tf.placeholder(tf.float32, shape=(None, self.vocab_size))
        self.W1 = tf.Variable(tf.random_normal([self.vocab_size, self.EMBEDDING_DIM]))
        self.b1 = tf.Variable(tf.random_normal([self.EMBEDDING_DIM]))  # bias
        hidden_representation = tf.add(tf.matmul(self.x, self.W1), self.b1)

        W2 = tf.Variable(tf.random_normal([self.EMBEDDING_DIM, self.vocab_size]))
        b2 = tf.Variable(tf.random_normal([self.vocab_size]))
        prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, W2), b2))

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)  # make sure you do this!
        # define the loss function:
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(self.y_label * tf.log(prediction), reduction_indices=[1]))
        # define the training step:
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
        for index in range(0, len(self.window_tuples), 1024):
            self.prepare_training_data_skipgram(self.window_tuples[index:index+1023])
            self.make_skipgram(sess, train_step, cross_entropy_loss)
        self.random_analysis()




