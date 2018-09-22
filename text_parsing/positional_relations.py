from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
import nltk
from text_parsing.creating_graph import CreateGraph


class PositionalRelations:

    def __init__(self):

        self.keywords = []
        with open('/Users/Jess/PycharmProjects/Honors_Thesis_2/text_parsing/source_text/keywords.txt', 'r') as words:
            keywords_file = words.readlines()

        for word in keywords_file:
            self.keywords.append(word.lower())

        self.ignored = []
        with open('/Users/Jess/PycharmProjects/Honors_Thesis_2/text_parsing/source_text/ignored.txt', 'r') as ignored:
            ignored_list = ignored.readlines()

        self.eng_stopwords = stopwords.words('english')
        self.eng_stopwords.extend(ignored_list)

        self.source_files = ['/Users/Jess/PycharmProjects/Honors_Thesis_2/TimeMachine1.txt']

        self.text = []
        self.text = self.text_to_wordlist(self.source_files)

        self.average_distances = {}
        self.surrounding_words = []

    def text_to_wordlist(self, file_list):
        for file in file_list:

            unmodified_string = ""
            with open(file, "r") as single_chapter:
                for line in single_chapter.readlines():
                    unmodified_string += line.replace('\n', " ")

            custom_sent_tokenizer = PunktSentenceTokenizer(unmodified_string)
            tokenized_sentences = custom_sent_tokenizer.tokenize(unmodified_string)

            words = []
            for i in tokenized_sentences:
                words += nltk.word_tokenize(i)

            text_as_words = []
            for i in words:
                temp = i.lower()
                if temp not in self.eng_stopwords:
                    text_as_words.append(temp)
            return text_as_words

    def find_average_dist(self, tracked):
        distances = []
        for other_word in self.keywords:
            for ind in range(0, len(self.text)):
                if self.text[ind] == tracked:
                    num = ind
                    while num >= 0 and self.text[num] != other_word:
                        num -= 1
                    if num >= 0:
                        distances.append({tracked: ind, other_word: num})
                    num = ind
                    while num < len(self.text) and self.text[num] != other_word:
                        num += 1
                    if num < len(self.text) :
                        distances.append({tracked: ind, other_word: num})

            if len(distances) != 0:
                sum = 0
                for ind in range(0, len(distances)):
                    sum += abs(distances[ind][tracked] - distances[ind][other_word])
                self.average_distances[other_word] = sum / len(distances)
            else:
                self.average_distances[other_word] = 0

        CreateGraph().average_distances(self.average_distances, tracked)

        return self.average_distances

    def within_range(self, tracked, num):
        found_in_range = []
        for ind in range(0, len(self.text)):
            if self.text[ind] == tracked:

                start = int(ind) - int(num)
                end = int(ind) + int(num)
                if start < 0:
                    start = 0
                if end >= len(self.text):
                    end = len(self.text) - 1
                for surrounding_ind in range(start, end + 1):

                    if self.text[surrounding_ind] in self.keywords and self.text[surrounding_ind] != tracked:
                        temp = {tracked: ind}
                        temp[self.text[surrounding_ind]] = surrounding_ind
                        found_in_range.append(temp)

        CreateGraph().within_range_graph(found_in_range, tracked)

        return found_in_range

    def print_surrounding_window(self, word, number):
        occurances = []
        for ind in range(0, len(self.text)):
            if self.text[ind] == word:
                start = ind - number
                if start < 0:
                    start = 0
                end = ind + number
                if end >= len(self.text):
                    end = len(self.text) - 1
                occurances.append({ind: self.text[start:end]})

                # Clusters with time_1
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




