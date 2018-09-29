import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
from .word_prep import WordPrep
import nltk


class SkipGram:

    def __init__(self, sen_word, words, word2int, int2word):

        self.sen_word_token = sen_word
        self.words = words
        self.int2word = int2word
        self.word2int = word2int

        # Training variables
        self.window_tuples = []
        self.WINDOW_SIZE = 5
        self.x_train = []  # input word
        self.y_train = []  # output word

        # Model Variables
        self.EMBEDDING_DIM = 3  # you can choose your own number

    def make_training_window_tuples(self):
        # Vectorization
        for word_index in range(0, len(self.words)):
            for nb_word in self.words[
                           max(word_index - self.WINDOW_SIZE, 0): min(word_index + self.WINDOW_SIZE, len(self.words)) + 1]:
                if nb_word != self.words[word_index]:
                    self.window_tuples.append([self.words[word_index], nb_word])

    # function to convert numbers to one hot vectors
    def to_one_hot(self, data_point_index, vocab_size):
        temp = np.zeros(len(self.words))
        temp[data_point_index] = 1
        return temp

    def prepare_training_data_skipgram(self):
        vocab_size = len(self.words)
        for data_word in self.window_tuples:
            self.x_train.append(self.to_one_hot(self.word2int[data_word[0]], vocab_size))
            self.y_train.append(self.to_one_hot(self.word2int[data_word[1]], vocab_size))

        # convert them to numpy arrays
        self.x_train = np.asarray(self.x_train)
        self.y_train = np.asarray(self.y_train)

        self.x = tf.placeholder(tf.float32, shape=(None, vocab_size))
        self.y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

    def make_skipgram(self):
        vocab_size = len(self.words)
        W1 = tf.Variable(tf.random_normal([vocab_size, self.EMBEDDING_DIM]))
        b1 = tf.Variable(tf.random_normal([self.EMBEDDING_DIM]))  # bias
        hidden_representation = tf.add(tf.matmul(self.x, W1), b1)

        W2 = tf.Variable(tf.random_normal([self.EMBEDDING_DIM, vocab_size]))
        b2 = tf.Variable(tf.random_normal([vocab_size]))
        prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, W2), b2))

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)  # make sure you do this!
        # define the loss function:
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(self.y_label * tf.log(prediction), reduction_indices=[1]))
        # define the training step:
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
        n_iters = 1000
        # train for n_iter iterations
        for _ in range(n_iters):
            sess.run(train_step, feed_dict={self.x: self.x_train, self.y_label: self.y_train})
            print('loss is : ', sess.run(cross_entropy_loss, feed_dict={self.x: self.x_train, self.y_label: self.y_train}))

        print(sess.run(W1))
        print('----------')
        print(sess.run(b1))
        print('----------')

        self.vectors = sess.run(W1 + b1)
        print(len(self.vectors))
        print(len(self.words))
        print(self.vectors)

    def random_analysis(self):
        print(self.int2word[self.find_closest(self.word2int['time'])])
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
        self.prepare_training_data_skipgram()
        self.make_skipgram()
        self.random_analysis()




