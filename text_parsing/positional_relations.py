from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
import nltk
import os
import numpy as np
from text_parsing.creating_graph import CreateGraph
from models.word_prep import WordPrep
import matplotlib.pyplot as plt
from datetime import datetime


class PositionalRelations:

    def __init__(self):

        self.ignored = []
        with open(os.path.abspath("./word_lists/ignored.txt"), 'r') as ignored:
            ignored_list = ignored.readlines()

        self.eng_stopwords = stopwords.words('english')
        self.eng_stopwords.extend(ignored_list)

        self.source_files = [os.path.abspath("../ficino/book_1-4.txt")]

        self.corpus_raw = ''
        for file in self.source_files:
            # Load data
            with open(file, "r",  encoding="utf8", errors='ignore') as time:
                for line in time.readlines():
                    self.corpus_raw += line.replace('\n', " ")
        self.wp = WordPrep(self.corpus_raw)
        self.count = self.wp.word_count()
        self.keywords = self.wp.keywords

        self.average_distances = {}
        self.surrounding_words = []

    def find_average_dist(self, tracked):
        distances = []
        for other_word, category in self.keywords.items():
            for ind in range(0, len(self.wp.word_list)):
                if self.wp.word_list[ind] == tracked:
                    num = ind
                    while num >= 0 and self.wp.word_list[num] != other_word:
                        num -= 1
                    if num >= 0:
                        distances.append({tracked: ind, other_word: num})
                    num = ind
                    while num < len(self.wp.word_list) and self.wp.word_list[num] != other_word:
                        num += 1
                    if num < len(self.wp.word_list) :
                        distances.append({tracked: ind, other_word: num})

            if len(distances) != 0:
                sum = 0
                for ind in range(0, len(distances)):
                    if other_word in distances[ind]:
                        sum += abs(distances[ind][tracked] - distances[ind][other_word])
                self.average_distances[other_word] = sum / len(distances)
            else:
                self.average_distances[other_word] = 0
        print("THis is average distances")
        print(self.average_distances)
        CreateGraph().average_distances(self.average_distances, tracked)

        return self.average_distances

    def within_range(self, tracked, num):
        found_in_range = []
        for ind in range(0, len(self.wp.word_list)):
            if self.wp.word_list[ind] == tracked:

                start = int(ind) - int(num)
                end = int(ind) + int(num)
                if start < 0:
                    start = 0
                if end >= len(self.wp.word_list):
                    end = len(self.wp.word_list) - 1
                for surrounding_ind in range(start, end + 1):

                    if self.wp.word_list[surrounding_ind] in self.keywords and self.wp.word_list[surrounding_ind] != tracked:
                        temp = {tracked: ind}
                        temp[self.wp.word_list[surrounding_ind]] = surrounding_ind
                        found_in_range.append(temp)

        CreateGraph().within_range_graph(found_in_range, tracked, num)

        return found_in_range

    def print_surrounding_window(self, word, number):
        occurances = []
        for ind in range(0, len(self.wp.word_list)):
            if self.wp.word_list[ind] == word:
                start = ind - number
                if start < 0:
                    start = 0
                end = ind + number
                if end >= len(self.wp.word_list):
                    end = len(self.wp.word_list) - 1
                occurances.append({ind: self.wp.word_list[start:end]})
        return occurances

    def structural_relations(self, main_word, other_word, book):
        distances = []
        for ind in range(0, len(book)):
            if main_word in book[ind]['sentence']:
                num = ind
                while num >= 0 and other_word not in book[num]['sentence']:
                    num -= 1
                if num >= 0:
                    temp = {main_word: book[ind], other_word: book[num]}
                    if temp not in distances:
                        distances.append(temp)
                num = ind
                while num < len(book) and other_word not in book[num]['sentence']:
                    num += 1
                if num < len(book):
                    temp = {main_word: book[ind], other_word: book[num]}
                    if temp not in distances:
                        distances.append(temp)

        return distances

    def get_counts(self):
        counts = []
        used_keywords = []

        print(self.wp.keyword_categories)
        print(self.wp.keywords)
        print(self.count)

        current_category = 0
        for word, category in self.keywords.items():
            if word in self.count:
                if current_category != category:
                    print("Changing category")
                    self.plot_category(used_keywords, current_category, counts)
                    counts = []
                    used_keywords = []
                    counts.append(self.count[word])
                    used_keywords.append(word)
                    current_category += 1

                else:
                    counts.append(self.count[word])
                    used_keywords.append(word)
        self.plot_category(used_keywords, current_category, counts)


        # y_pos = np.arange(len(used_keywords))
        # print(y_pos)
        # plt.bar(y_pos, counts, align='center', alpha=0.5)
        # plt.xticks(y_pos, self.keywords)
        # plt.ylabel('Usage')
        # plt.title('Keyword Counts')
        #
        # plt.show()

    def plot_category(self, used_keywords, current_category, counts):
        y_pos = np.arange(len(used_keywords))
        plt.bar(y_pos, counts, align='center', alpha=0.5)
        plt.xticks(y_pos, used_keywords, rotation='vertical')
        plt.ylabel('Usage')
        plt.title('Keyword Counts for Category: ' + str(current_category))
        plt.savefig(
            os.path.abspath("../graphics_ficino/word_count" + str(current_category) + "_" + datetime.utcnow().isoformat(
                'T') + '.png'))
        plt.show()



