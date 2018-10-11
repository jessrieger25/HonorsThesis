from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
import nltk
import os
from nltk.corpus import stopwords


class WordPrep():

    def __init__(self, corpus_raw):

        # blank variables
        self.vocab_list = set()
        self.word_list = []
        self.word2int = {}
        self.int2word = {}
        self.keywords = {}

        self.corpus_raw = corpus_raw
        self.categories_num = 1
        self.eng_stopwords = stopwords.words('english')
        self.sen_word_token = self.create_token_sentences()
        self.word_list = self.create_word_list()
        self.vocab_list = self.create_vocab_set()
        dictionaries = self.make_int_dictionary()
        self.word2int = dictionaries[0]
        self.int2word = dictionaries[1]
        self.keywords = self.create_keyword_list()

        self.list_of_list = []
        for single_keyword in range(0, self.categories_num):
            self.list_of_list.append([])
        self.list_of_list.append([])

    def create_keyword_list(self):
        with open(os.path.abspath("../text_parsing/word_lists/keywords.txt"), 'r') as words:
            keywords_file = words.readlines()

        for word in keywords_file:
            print('going through', word)
            if word == "\n":
                self.categories_num += 1
                print("increment")
            else:
                self.keywords[word.strip().lower()] = self.categories_num
                print(self.keywords)
        return self.keywords

    def create_token_sentences(self):
        # Data prep
        custom_sent_tokenizer = PunktSentenceTokenizer(self.corpus_raw)
        tokenized_sentences = custom_sent_tokenizer.tokenize(self.corpus_raw)

        return tokenized_sentences

    def create_vocab_set(self):

        for one in self.word_list:
            self.vocab_list.add(one)

        return self.vocab_list

    def create_word_list(self):
        for sentence in self.sen_word_token:
            temp = nltk.word_tokenize(sentence)
            for one in temp:
                # if one not in self.eng_stopwords and one not in [".", "!", "?"]:
                self.word_list.append(one)
        return self.word_list

    def make_int_dictionary(self):
        for i, word in enumerate(self.vocab_list):
            self.word2int[word] = i
            self.int2word[i] = word
        print(self.word2int)
        print(self.vocab_list)
        print(self.int2word)
        return self.word2int, self.int2word

    def word_count(self):
        counts = {}
        for word in self.word_list:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] = counts[word] + 1
        return counts