from models.word_prep import WordPrep
import nltk
import tensorflow as tf
import numpy as np
import random
import collections
import time
# Source: https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537


class LSTMModel:

    def __init__(self, corpus_sent, word2int, plain_corpus, vectors):
        self.token_sent = corpus_sent
        self.word2int = word2int
        self.corpus = nltk.word_tokenize(plain_corpus)
        self.vocab_size = len(self.word2int.items())
        self.vectors = vectors
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.vocab_size]))
        }
        print("vocab")
        print(self.vocab_size)
        self.n_input = self.vocab_size
        self.n_hidden = 512
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.vocab_size]))
        }
        self.learning_rate = 0.001
        self.training_iters = 50000
        self.display_step = 1000
        self.x = tf.placeholder("float", [None, self.n_input, 1])
        self.y = tf.placeholder("float", [None, self.vocab_size])
        self.dictionary, self.reverse_dictionary = self.build_dataset(self.corpus)
        self.start_time = time.time()

    def make_multi_hot_vecs(self):
        new_hot_vecs = []
        new_bland_vecs = []

        for sen in self.token_sent:
            temp_vec = np.zeros([self.vocab_size], dtype=float)
            temp_word_vec = []
            temp = nltk.word_tokenize(sen)
            for word in temp:
                temp_word_vec.append(word)
                temp_vec[self.word2int[word]] += 1
            new_hot_vecs.append(temp_vec)
            new_bland_vecs.append(temp_word_vec)
        return new_hot_vecs, new_bland_vecs

    def build_dataset(self, words):
        count = collections.Counter(words).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary

    def RNN(self, x, weights, biases):

        # reshape to [1, n_input]

        x = tf.reshape(x, [-1, self.n_input])

        # Generate a n_input-element sequence of inputs
        # (eg. [had] [a] [general] -> [20] [6] [33])
        x = tf.split(x, self.n_input, 1)

        rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.n_hidden), tf.contrib.rnn.BasicLSTMCell(self.n_hidden)])

        # 1-layer LSTM with n_hidden units.
        # rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

        # generate prediction
        outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # there are n_input outputs but
        # we only want the last output
        print(outputs)
        return tf.matmul(outputs[-13], weights['out']) + biases['out']

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def run(self):

        pred = self.RNN(self.x, self.weights, self.biases)

        # Loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(cost)

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as session:
            session.run(init)
            step = 0
            offset = random.randint(0, self.n_input + 1)
            end_offset = self.n_input + 1
            acc_total = 0
            loss_total = 0

            while step < self.training_iters:
                # Generate a minibatch. Add some randomness on selection process.
                if offset > (len(self.corpus) - end_offset):
                    offset = random.randint(0, self.n_input + 1)

                symbols_in_keys = [[self.dictionary[str(self.corpus[i])]] for i in range(offset, offset + self.n_input)]
                symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, self.n_input, 1])

                symbols_out_onehot = np.zeros([self.vocab_size], dtype=float)
                symbols_out_onehot[self.dictionary[str(self.corpus[offset + self.n_input])]] = 1.0
                symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

                _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                        feed_dict={self.x: symbols_in_keys, self.y: symbols_out_onehot})
                loss_total += loss
                acc_total += acc
                if (step + 1) % self.display_step == 0:
                    print("Iter= " + str(step + 1) + ", Average Loss= " + \
                          "{:.6f}".format(loss_total / self.display_step) + ", Average Accuracy= " + \
                          "{:.2f}%".format(100 * acc_total / self.display_step))
                    acc_total = 0
                    loss_total = 0
                    symbols_in = [self.corpus[i] for i in range(offset, offset + self.n_input)]
                    symbols_out = self.vectors[offset + self.n_input]
                    symbols_out_pred = onehot_pred
                    print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
                step += 1
                offset += (self.n_input + 1)
            print("Optimization Finished!")
            print("Elapsed time: ", self.elapsed(time.time() - self.start_time))
            print("Run on command line.")
            print("Point your web browser to: http://localhost:6006/")
            while True:
                prompt = "%s words: " % self.n_input
                sentence = input(prompt)
                sentence = sentence.strip()
                words = sentence.split(' ')
                if len(words) != self.n_input:
                    continue
                try:
                    symbols_in_keys = [self.dictionary[str(words[i])] for i in range(len(words))]
                    for i in range(32):
                        keys = np.reshape(np.array(symbols_in_keys), [-1, self.n_input, 1])
                        onehot_pred = session.run(pred, feed_dict={self.x: keys})
                        onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                        sentence = "%s %s" % (sentence, self.reverse_dictionary[onehot_pred_index])
                        symbols_in_keys = symbols_in_keys[1:]
                        symbols_in_keys.append(onehot_pred_index)
                    print(sentence)
                except:
                    print("Word not in dictionary")