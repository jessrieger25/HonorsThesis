
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn import preprocessing
from datetime import datetime


class TSNEVisualizations():

    def __init__(self):
        self.model = TSNE(n_components=2, random_state=0, n_iter=10000, init='pca')

        """
        t-SNE [1] is a tool to visualize high-dimensional data. It converts similarities between data points to joint 
        probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the 
        low-dimensional embedding and the high-dimensional data. t-SNE has a cost function that is not convex, i.e. 
        with different initializations we can get different results.

        In probability theory and statistics, a probability distribution is a mathematical function that provides the 
        probabilities of occurrence of different possible outcomes in an experiment.

        A joint probability is a statistical measure that calculates the likelihood of two events occurring together and 
        at the same point in time.

        In mathematical statistics, the Kullback–Leibler divergence is a measure of how one probability distribution is 
        different from a second, reference probability distribution
        """

    def run(self, vectors, words, word2int, sizes={}, separates=[], keywords={}):

        np.set_printoptions(suppress=True)
        vectors = self.model.fit_transform(vectors)

        print(vectors)
        max = 0
        min = 0

        for word in words:
            if vectors[word2int[word]][1] > max:
                max = vectors[word2int[word]][1]
            elif vectors[word2int[word]][1] < min:
                min = vectors[word2int[word]][1]

        fig, ax = plt.subplots()
        plt.axis([min, max, min, max])

        size_list = list()
        for word in words:
            print(vectors[word2int[word]][1])

            if word in keywords:

                separates[keywords[word]-1].append(vectors[word2int[word]])
            else:
                separates[len(separates)-1].append(vectors[word2int[word]])

            size_list.append(sizes[word])
            ax.annotate(word, (vectors[word2int[word]][0], vectors[word2int[word]][1]))

        plt.show()
        fig.savefig('/Users/Jess/PycharmProjects/Honors_Thesis_2/graphics/tsne_' + datetime.utcnow().isoformat('T') + '.png', dpi=350)

        # Help to display later
        # from IPython.display import Image
        # Image('my_figure.png')

        self.scatterplot(vectors[:, 0], vectors[:, 1], x_label='x', y_label='y', title='Scatter Plot of Time Machine Text', sizes=size_list, lists=separates)

    def scatterplot(self, x_data, y_data, x_label="", y_label="", title="", color="r", yscale_log=False, sizes=[], lists=[]):
        colors = ['r', 'y', 'g', 'o']
        fig, ax = plt.subplots()

        for one in range(0, len(lists)):
            x = []
            y = []
            for vec in lists[one]:
                x.append(vec[0])
                y.append(vec[1])
            plt.scatter(x, y, label='words', color=colors[one], s=sizes, alpha=0.75)

        # Could also plt scatter with all x and y data

        #, marker="o"
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.show()
        fig.savefig('/Users/Jess/PycharmProjects/Honors_Thesis_2/graphics/tsne_scatter_' + datetime.utcnow().isoformat('T') + '.png', dpi=350)

