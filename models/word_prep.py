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
        print("Creating sentences")
        self.sen_word_token = self.create_token_sentences()
        print("Creating wordlist")
        self.word_list = self.create_word_list()
        print("Creating vocabset")
        self.vocab_list = self.create_vocab_set()
        print("Creating int dic")
        dictionaries = self.make_int_dictionary()
        self.word2int = dictionaries[0]
        self.int2word = dictionaries[1]

        self.keyword_categories = []
        print("Creating keyword")
        self.keywords = self.create_keyword_list()

        self.list_of_list = []
        for single_keyword in range(0, self.categories_num):
            self.list_of_list.append([])
        self.list_of_list.append([])

    def create_keyword_list(self):
        with open(os.path.abspath("../text_parsing/word_lists/keywords.txt"), 'r') as words:
            keywords_file = words.readlines()

        for word in keywords_file:
            if word == "\n":
                self.categories_num += 1
            elif 'Category: ' in word:
                category = word.split(':')
                self.keyword_categories.append(category[1].strip())
            else:
                self.keywords[word.strip().lower()] = self.categories_num

        self.keyword_categories.append('uncategorized')
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
                self.word_list.append(one.lower().strip())
        return self.word_list

    def make_int_dictionary(self):
        for i, word in enumerate(self.vocab_list):
            cleaned_word = word.lower().strip()
            self.word2int[cleaned_word] = i
            self.int2word[i] = cleaned_word
        return self.word2int, self.int2word

    def word_count(self):
        counts = {}
        for word in self.word_list:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] = counts[word] + 1
        return counts