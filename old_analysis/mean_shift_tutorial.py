""" Mean shift = an algorithm that determines how many clusters there should be for a given dataset.

Radius = a perfect circle around a cluster center. To begin, each data point (feature set) is its own cluster.

bandwidth = everything in the circle.

1. Take radius of cluster center
2. Take mean of all bandwidth - mean of featuresets in the circle.
3. move cluster center.

When not move any more, optimized, done. Convergence of cluster centers to form your final clusters.


"""

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")
from sklearn import preprocessing
import pandas as pd
import re
import nltk
from sklearn.cluster.k_means_ import KMeans
from nltk.corpus import stopwords


# Create starting centers - used to generate random data from it, so then they should be close to the cluster centers
# that the algo finds otherwise something went wrong.


class ThreeDPlotting:

    @staticmethod
    def main():
        centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]
        X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1)

        ms = MeanShift()
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        print(cluster_centers)

        n_clusters = len(np.unique(labels))

        colors = 10*['r', 'g', 'b', 'c', 'k', 'y', 'm']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(X)):
            ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='x', c='k', s=150, linewidths=5, zorder=10)
        plt.show()

""" Starting part 40 of the mean shift tutorial """

class TitanicUtilityMethods:

    def __init__(self):
        self.df = pd.read_excel("titanic.xls")
        self.df.drop(['body', 'name'], 1, inplace=True)
        self.df.convert_objects(convert_numeric=True)
        self.df.fillna(0, inplace=True)
        self.df.drop(['boat', 'sex'], 1, inplace=True)

        self.original_df = pd.DataFrame.copy(self.df)

    """ Problem: most of the columns are not numerical, thus it makes
    it impossible to use kmeans on that data. So we solve this by taking
    the set of a column and assigning each a unique ID. (number)

    Fill in missing data as well.
    """
    def handle_non_numeric_data(self):

        columns = self.df.columns.values
        for column in columns:
            text_digit_values = {}

            def convert_to_int_val(val):
                return text_digit_values[val]

            if self.df[column].dtype != np.int64 and self.df[column].dtype != np.float64:
                column_contents = self.df[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:
                    if unique not in text_digit_values:
                        text_digit_values[unique] = x
                        x += 1
                self.df[column] = list(map(convert_to_int_val, self.df[column]))
        return self.df

    def analyze_and_print(self):

        # Can remove other columns to see how they affect the results
        X = np.array(self.df.drop(['survived'], 1).astype(float))
        X = preprocessing.scale(X)  # Adjusts the mean of the data so that it is more standardized. Helps account for
                                    # Different factors having different ranges and not allowing outliers to mess
                                    # with learning.
        y = np.array(self.df['survived'])
        clf = KMeans(n_clusters=2)
        clf.fit(X)

        correct = 0
        for i in range(0, len(X)):
            predict_me = np.array(X[i].astype(float))
            predict_me = predict_me.reshape(-1, len(predict_me))
            prediction = clf.predict(predict_me)
            if prediction[0] == y[i]:
                correct += 1

        print(correct/len(X))


class TimeMachineLoad:

    def __init__(self):
        self.file_prefix = '/Users/Jess/Desktop/TimeMachine/TimeMachine'
        self.file_suffix = ".txt"
        self.paragraph_denoting_string = '\n'
        self.page_length = 30
        self.titles = ['Introduction', 'The Machine', 'The Time Traveller Returns', 'Time Travelling', 'In the Golden Age',
                  'The Sunset of Mankind', 'A Sudden Shock', 'Explanation', 'The Morlocks', 'When Night Came',
                  'The Palace of Green Porcelain', 'In the Darkness', 'The Trap of the White Sphinx',
                  'The Further Vision',
                  'The Time Traveller’s Return', 'After the Story']
        self.word_to_num_dict = dict()
        self.words_array = []
        self.words = []
        self.eng_stopwords = stopwords.words('english')
        self.page_counter = 1
        self.paragraph_counter = 1
        self.paragraph_start_line = 0
        self.continued_paragraph = False

        self.sentence_counter = 1
        self.sentence_string = ""
        self.continued_sentence = False

        self.punctuation_characters = [".", "?", "!"]
        self.removed_characters = ["(", ")", "“", "”", ","]
        self.punctuation = re.compile('[\.?!]')
        self.removed = re.compile('.*[)(,“”].*')
        self.pos_dict = {}
        self.index = 0

    def make_word_array(self):
        """ Preparing the text: This removes uppercase letters, newlines, double spaces. Could add additional parsing - punctuation removal, etc."""
        unmodified_string = ""
        with open("/Users/Jess/PycharmProjects/HonorsThesis/TimeMachine1.txt", "r") as time:
            for line in time.readlines():
                unmodified_string += line

        # Number of lines on a page - randomly decided


        # Input data files are available in the "../input/" directory.
        # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

        book_id = 1
        row_counter = 0
        index = 0
        # self.words_array.append(['length', 'sentence', 'word', 'POS'])

        # Adding more data:
        for chapter_counter in range(1, 4):
            print(chapter_counter)
            filename = self.file_prefix + str(chapter_counter) + self.file_suffix
            file = open(filename, 'rt')
            lines = file.readlines()

            # words_array.append(["Book", "page", "line_num", "paragraph", "sentence", "word_index"])
            # should be len lines
            for line_num in range(0, len(lines)):

                line = lines[line_num].lower()
                print(line)

                if line == self.paragraph_denoting_string:
                    self.paragraph_counter += 1
                    continue

                if line_num % self.page_length == 0 and line_num > 0:
                    self.page_counter += 1
                    self.paragraph_counter = 1

                parts = []
                for character in self.punctuation_characters:
                    if character in line:
                        parts = line.split(character)
                        break
                    else:
                        parts = [line]

                for num in range(len(parts)):
                    words = parts[num].split(" ")

                    for i in range(len(words)):
                        if words[i] not in self.eng_stopwords:
                            words[i] = words[i].strip()

                            if self.removed.match(words[i]):
                                for char in self.removed_characters:
                                    if char in words[i]:
                                        words[i] = words[i].replace(char, "")

                            if words[i] not in self.word_to_num_dict:
                                self.word_to_num_dict[words[i]] = index
                                current_index = index
                                index += 1
                            else:
                                current_index = self.word_to_num_dict[words[i]]
                            if words[i] != "":
                                tagged = nltk.pos_tag([words[i]])
                                if tagged[0][1] not in self.pos_dict:
                                    self.pos_dict[tagged[0][1]] = self.index
                                    tagged_index = self.index
                                    self.index += 1
                                else:
                                    tagged_index = self.pos_dict[tagged[0][1]]

                            else:
                                tagged_index = float(0)

                            # Format = Book, page, line_num, paragraph, sentence, word_index
                            # This is the key part: the parts of the array are crucial as they determine
                            # what is found or not found.
                            self.words_array.append(
                                [float(len(words[i])), float(current_index), float(tagged_index)])
                            # self.words_array.append(
                            #     [float(len(words[i])), float(chapter_counter), float(self.page_counter), float(line_num % self.page_length), float(self.paragraph_counter),
                            #      float(self.sentence_counter), float(current_index)])
                            self.words.append(words[i])
                            row_counter += 1
                    self.sentence_counter += 1

        return self.words_array


class BasicUtility:

    def __init__(self):
        self.colors = ["g", "r", "b", "c", "k", "o"]

    def analyze(self, passed_clf, passed_original_df, np_array):
        labels = passed_clf.labels_
        cluster_centers = passed_clf.cluster_centers_
        print(passed_original_df)
        passed_original_df['cluster_group'] = np.nan

        for i in range(len(np_array)):
            passed_original_df['cluster_group'].iloc[i] = labels[i]

        print(passed_original_df)
        n_clusters_ = len(np.unique(labels))
        survival_rates = {}

        # for i in range(n_clusters_):
        #     temp_df = passed_original_df[(passed_original_df['cluster_group'] == float(i))]
        #     survival_cluster = temp_df[(temp_df['survived'] == 1)]
        #     survival_rate = len(survival_cluster) / len(temp_df)
        #     survival_rates[i] = survival_rate
        # print(survival_rates)

    def print_clf(self, passed_clf):

        for centroid in passed_clf.centroids:
            plt.scatter(passed_clf.centroids[centroid][0], passed_clf.centroids[centroid][1], marker="o", color="k", s=150,
                        linewidths=5)

        for classification in passed_clf.classifications:
            color = self.colors[classification]
            for featureset in passed_clf.classifications[classification]:
                plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

    def using_k_means_predict(self, np_array, passed_clf):
        # This uses the custom k_means algorithm, only works for 2 column datasets.
        for unknown in np_array:
            classification = passed_clf.predict(unknown)
            plt.scatter(unknown[0], unknown[1], marker="*", color=self.colors[classification], s=150, linewidths=5)
        plt.show()

""" Clustering: given just feature sets and machine searches for groups and custers. Maybe make data with features of page
sentence, paragraph, etc.

Flat clustering - you decide groups
clustering - machine decides groups.
"""

""" K-means clustering: k = number of groups or clusters.
What if we classify the words based on their sentiment, 
then look for their relative positions based on paragraph, line, 
etc using this tactic. Could classify topic with color and see the
different clusters.
"""

""" Making our own K means algorithm and then applying it to the titanic dataset."""

class K_Means:

    # tol = how much will centroid move.

    # Change num iterations to see how the classification evolves.
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = dict()

        for i in range(self.k):

             self.centroids[i] = data[i]

        for i in range(self.max_iter):

            self.classifications = dict()

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:

                dist = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = dist.index(min(dist))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        dist = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = dist.index(min(dist))
        return classification

""" Basics of KMeans clustering """

class KMeansBasics:

    def __init__(self):
        self.X = np.array([[1, 2],
                          [1.5, 1.8],
                          [8, 8],
                          [5, 8],
                          [1, .6],
                          [9, 11]])

    def plot(self, array_to_plot):
        plt.scatter(array_to_plot[:, 0], array_to_plot[:, 1], s=150, linewidths=5)
        plt.show()

    def apply_KMeans(self, array_to_fit, num_clusters):
        clf = KMeans(n_clusters=num_clusters)
        clf.fit(self.X)

    def plot_fitted_algorithm(self, clf_to_map):
        centroids = clf_to_map.cluster_centers_

        labels = clf_to_map.labels_

        colors = ["g.", "r.", "b.", "c.", "k.", "o."]

        for i in range(0, len(self.X)):
            plt.plot(self.X[i][0], self.X[i][1], colors[labels[i]], markersize=25)

        plt.scatter(centroids[:, 0], centroids[:, 0], marker="x", s=100, linewidths=5)
        plt.show()

""" Titanic Data

Working with the titanic data - preparing the dataset, fitting the meanshift algorithm to it. 
"""
# titanic = TitanicUtilityMethods()
# titanic.handle_non_numeric_data()
#
# X = np.array(titanic.df.drop(['survived'], 1).astype(float))
# X = preprocessing.scale(X)  # Adjusts the mean of the data so that it is more standardized. Helps account for
#                             # Different factors having different ranges and not allowing outliers to mess
#                             # with learning.
# y = np.array(titanic.df['survived'])
#
# clf = MeanShift()
# clf.fit(X)
# BasicUtility().analyze(clf, titanic.original_df)

""" Time Machine Data

Working with the Time Machine Data : Creating an array, making it into an np.array then a dataframe, then putting
it through the algorithm. 

The Dataframe is good if you want to call the first part of analyze on it so that you can see the labels.

At the bottom we can also see the application of kmeans to the data. 
"""

book = TimeMachineLoad()

returned_array = book.make_word_array()

T = np.array(book.words_array, dtype=float)

# # Version 1:
# word_df = pd.DataFrame({"length": T[:, 0], "sentence": T[:, 1], "word": T[:, 2], "POS": T[:, 3]})

# Version 2:
word_df = pd.DataFrame({"length": T[:, 0], "word": T[:, 1], "POS": T[:, 2]})

original_word_df = pd.DataFrame.copy(word_df)
# print(word_df)
#
# T = np.array(word_df.astype(float))
# # X = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [10.0, 10.0, 10.0]])
# print("after array")
new_clf = MeanShift()
new_clf.fit(T)

print(set(new_clf.labels_))
#s
print("stuck on labels")
new_labels = new_clf.labels_

print(set(new_labels))
BasicUtility().analyze(new_clf, original_word_df, T)

plotted_df = pd.DataFrame.copy(original_word_df)

plotted_df.drop(['length', 'POS'], 1, inplace=True)
array_of_words = np.array(plotted_df, dtype=float)
print(array_of_words)

KMeansBasics().plot(array_of_words)
print(original_word_df[(original_word_df['cluster_group'] == 1)])

# # Applying KMeans to data
# # Note: works well when you do not tell how many clusters.
# new_clf = KMeans()
# new_clf.fit(T)
#
# print(new_clf.labels_)
# print(set(new_clf.labels_))
#
# BasicUtility().analyze(new_clf, original_word_df, np_array=T)

""" Things to add from the Kaggle notebook: wordcounts, synonyms, topics"""

""" Random: can take the labeling column of the dataframe when it equals a particular group, and say .describe() for stats."""