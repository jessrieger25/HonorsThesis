from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords


class WordPrep():

    def __init__(self, corpus_raw):

        # blank variables
        self.vocab_list = []
        self.word_list = []
        self.word2int = {}
        self.int2word = {}
        self.keywords = {}

        self.corpus_raw = corpus_raw
        self.categories_num = 1
        self.eng_stopwords = stopwords.words('english')
        self.sen_word_token = self.create_token_sentences()
        self.vocab_list = self.create_vocab_set()
        self.word_list = self.create_word_list()
        dictionaries = self.make_int_dictionary()
        self.word2int = dictionaries[0]
        self.int2word = dictionaries[1]
        self.keywords = self.create_keyword_list()

    def create_keyword_list(self):
        with open('/Users/Jess/PycharmProjects/Honors_Thesis_2/text_parsing/source_text/keywords.txt', 'r') as words:
            keywords_file = words.readlines()

        for word in keywords_file:
            if word == "\n":
                self.categories_num += 1
            else:
                self.keywords[word.strip().lower()] = self.categories_num
        return self.keywords

    def create_token_sentences(self):
        sen_word_token = []
        # Data prep
        custom_sent_tokenizer = PunktSentenceTokenizer(self.corpus_raw)
        tokenized_sentences = custom_sent_tokenizer.tokenize(self.corpus_raw)

        for sen in tokenized_sentences:
            temp = nltk.word_tokenize(sen)
            new_temp = []
            for one in temp:
                if one not in self.eng_stopwords and one not in [".", "!", "?"]:
                    new_temp.append(one)
            sen_word_token.append(new_temp)
        return sen_word_token

    def create_vocab_set(self):

        for one in self.sen_word_token:
            for internal in one:
                self.vocab_list.append(internal)

        return self.vocab_list

    def create_word_list(self):
        for sentence in self.sen_word_token:
            for word_index, word in enumerate(sentence):
                self.word_list.append(word)
        return self.word_list

    def make_int_dictionary(self):
        for i, word in enumerate(self.vocab_list):
            self.word2int[word] = i
            self.int2word[i] = word

        return self.word2int, self.int2word

    def word_count(self):
        counts = {}
        for word in self.word_list:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] = counts[word] + 1
        return counts