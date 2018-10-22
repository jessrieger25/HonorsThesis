from numpy import asarray
from numpy import zeros
import os


class Glove():
    def __init__(self, sen_word, vocab, words, word2int, int2word, keywords):

        self.words = words
        self.vocab = vocab
        self.word2int = word2int
        self.vocab_size = len(self.vocab)
        self.embeddings_index = self.load_vecs()
        self.embedding_matrix = zeros((len(self.words), 100))
        self.keyword_list = keywords
        self.keyword_embedding = {}
        self.make_keyword_embedding()

    def load_vecs(self):

        # load the whole embedding into memory
        embeddings_index = dict()
        f = open(os.path.abspath("./glove_text/glove.6B.100d.txt"))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        return embeddings_index

    def make_keyword_embedding(self):
        print("making keyword embedding")
        print(self.keyword_list)
        for word in self.keyword_list.items():
            print(word)
            embedding_vector = self.embeddings_index.get(word[0])
            if embedding_vector is not None:
                self.keyword_embedding[word[0]] = embedding_vector

        print('Loaded %s word vectors.' % len(self.keyword_embedding))
        return self.keyword_embedding

    def make_embedding_matrix(self):

        # create a weight matrix for words in training docs
        for ind in range(0, len(self.words)):
            print(self.words)
            print(self.keyword_embedding)
            if self.words[ind] in self.keyword_embedding:

                embedding_vector = self.keyword_embedding[self.words[ind]]
                self.embedding_matrix[ind] = embedding_vector

    def run(self):
        self.load_vecs()
        self.make_embedding_matrix()
