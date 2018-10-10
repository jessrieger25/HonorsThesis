
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers import LSTM


class LSTMKeras:

    def __init__(self, sentence_tokens, sentiment_vecs):
        self.sent = sentence_tokens # docs in the tutorial.
        self.labels = sentiment_vecs
        self.vocab_size = 50
        self.max_len = self.max_length()

    # Must get the max length of a sentence!
    def max_length(self):
        max_sen_length = 0
        for sen in self.sent:
            if len(sen) > max_sen_length:
                max_sen_length = len(sen)

        return max_sen_length

    def run(self):
        # integer encode the documents

        encoded_docs = [one_hot(sen, self.vocab_size) for sen in self.sent]
        print(encoded_docs)

        # pad documents
        padded_docs = pad_sequences(encoded_docs, maxlen=self.max_len, padding='post')
        print(padded_docs)

        # define the model
        model = Sequential()
        model.add(Embedding(self.vocab_size, 8, input_length=self.max_len))
        model.add(LSTM(13))
        model.add(Dense(13, activation='sigmoid'))
        # compile the model
        model.compile(optimizer='sgd', loss='mean_absolute_error', metrics=['acc', 'mae'])
        # summarize the model
        print(model.summary())

        # fit the model
        model.fit(padded_docs, self.labels, epochs=100, verbose=0)
        print(padded_docs)

        # evaluate the model
        loss, accuracy, mean_absolute_error = model.evaluate(padded_docs, self.labels, verbose=0)

        # WILL HAVE TO ADD TSNE ONCE HAVE ACTUAL DATA
        print(model.layers)
        print("Loss: ", loss, ", accuracy: ", accuracy, ", mean_absolute_error: ", mean_absolute_error)
        return model.layers[0].get_weights()


