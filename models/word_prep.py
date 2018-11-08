from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
import nltk
import os
from nltk.corpus import stopwords


class WordPrep():

    def __init__(self, file_list):

        self.corpus_raw = self.load_text(file_list)
        self.tokenized_sentences = self.create_token_sentences(self.corpus_raw)
        self.word_list = self.create_word_list(self.tokenized_sentences)
        self.vocab_list = self.create_vocab_set(self.word_list)
        dictionaries = self.make_int_dictionary(self.vocab_list)
        self.word2int = dictionaries[0]
        self.int2word = dictionaries[1]

        self.list_of_list = []

        keyword_info = self.create_keyword_list()
        self.keywords = keyword_info[0]
        self.keyword_categories = keyword_info[1]

    def load_text(self, file_list):
        # Preparing the text:
        corpus_raw = ""

        for file in file_list:
            # Load data
            with open(file, "r", encoding="utf8", errors='ignore') as time:
                for line in time.readlines():
                    corpus_raw += line.replace('\n', " ")
        return corpus_raw

    def create_token_sentences(self, text):
        '''

        :param text: The raw text that was passed to WordPrep
        :return: A list of tokenized sentences.
        '''
        custom_sent_tokenizer = PunktSentenceTokenizer(text)
        tokenized_sentences = custom_sent_tokenizer.tokenize(text)

        return tokenized_sentences

    def create_word_list(self, tokenized_sentences):
        '''

        :param tokenized_sentences: List of tokenized sentences created above.
        :return: List of tokenized words.
        '''
        word_list = []

        for sentence in tokenized_sentences:
            temp = nltk.word_tokenize(sentence)
            for one in temp:
                word_list.append(one.lower().strip())
        return word_list

    def create_vocab_set(self, word_list):
        '''

        :param word_list: List of tokenized words.
        :return: Set of all the words present in the list.
        '''

        vocab_list = set()

        for one in word_list:
            vocab_list.add(one)

        return vocab_list

    def make_int_dictionary(self, vocab_list):
        """

        :param vocab_list: Set of all the words in the text.
        :return: Two dictionaries, one with words as the keys and integers as the values, and the other
        as the exact opposite. Facilitates word to integer conversion for analysis.
        """

        word2int = {}
        int2word = {}

        for i, word in enumerate(vocab_list):
            word2int[word] = i
            int2word[i] = word
        return word2int, int2word

    def create_keyword_list(self):
        """

        :return: Created dictionary of keywords (grouped by cateogry - represented as the value.) As
        well as list of category names.
        """
        categories_num = 0
        keyword_categories = []
        keywords = {}

        with open(os.path.abspath("../text_parsing/word_lists/keywords.txt"), 'r') as words:
            keywords_file = words.readlines()

        for word in keywords_file:
            if word == "\n":
                categories_num += 1
            elif 'Category: ' in word:
                category = word.split(':')
                keyword_categories.append(category[1].strip())
                self.list_of_list.append([])
            else:
                keywords[word.strip().lower()] = categories_num

        keyword_categories.append('uncategorized')
        self.list_of_list.append([])
        return keywords, keyword_categories

    def word_count(self, word_list):
        """

        :return: Dictionary with all the words used, and the number of times they appear.
        """
        counts = {}
        for word in word_list:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] = counts[word] + 1
        return counts