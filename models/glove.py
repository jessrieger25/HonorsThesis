from numpy import asarray
from .word_prep import WordPrep
from numpy import zeros


class Glove():
    def __init__(self, sen_word, vocab, words, word2int, int2word):

        self.words = words
        self.vocab = vocab
        self.word2int = word2int
        self.vocab_size = len(self.vocab)
        self.embeddings_index = self.load_vecs()
        self.embedding_matrix = zeros((len(self.words), 100))

    def load_vecs(self):

        # load the whole embedding into memory
        embeddings_index = dict()
        f = open('/Users/Jess/PycharmProjects/Honors_Thesis_2/models/glove_text/glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        return embeddings_index

    def make_embedding_matrix(self):

        # create a weight matrix for words in training docs
        for ind in range(0, len(self.words)):
            embedding_vector = self.embeddings_index.get(self.words[ind])
            if embedding_vector is not None:
                self.embedding_matrix[ind] = embedding_vector

    def run(self):
        self.load_vecs()
        self.make_embedding_matrix()
