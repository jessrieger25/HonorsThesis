
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


class TSNEVisualizations():

    def __init__(self):
        self.model = TSNE(n_components=2, random_state=0, n_iter=10000, init='pca')
        self.model3D = TSNE(n_components=3, random_state=0, n_iter=10000, init='pca')
        self.colors = ['#eb871b', 'r', 'y', 'g', 'c', 'm', '#b816e0']

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

    def run(self, vectors, words, word2int, sizes={}, separates=[], keywords={}, keyword_categories = [], type='Analysis'):

        np.set_printoptions(suppress=True)

        vectors2D = self.model.fit_transform(vectors)
        vectors3D = self.model3D.fit_transform(vectors)


        max = 0
        min = 0

        for word in words:
            if vectors2D[word2int[word]][1] > max:
                max = vectors2D[word2int[word]][1]
            elif vectors2D[word2int[word]][1] < min:
                min = vectors2D[word2int[word]][1]

        fig, ax = plt.subplots()
        plt.axis([min, max, min, max])

        size_list = list()
        separates_copy = copy.deepcopy(separates)
        separates_3D = copy.deepcopy(separates)

        for word in words:

            if word in keywords:

                separates[keywords[word]-1].append(vectors2D[word2int[word]])
                separates_3D[keywords[word]-1].append(vectors3D[word2int[word]])
                separates_copy[keywords[word]-1].append(word)
            else:
                separates[len(separates)-1].append(vectors2D[word2int[word]])

            size_list.append(sizes[word])
            ax.annotate(word, (vectors2D[word2int[word]][0], vectors2D[word2int[word]][1]))

        # Taken out for running on VM
        # plt.show()

        fig.savefig(os.path.abspath("../graphics_ficino/" + type + "_" + datetime.utcnow().isoformat('T') + '.png'), dpi=350)

        # Help to display later
        # from IPython.display import Image
        # Image('my_figure.png')

        self.scatterplot(vectors2D[:, 0], vectors2D[:, 1], x_label='x', y_label='y', sizes=size_list, lists=separates, list_of_labels=separates_copy, keyword_categories=keyword_categories, type=type)

        self.threeD_plot(vectors3D[:, 0], vectors3D[:, 1], vectors3D[:, 2], separates_3D, keyword_categories, type)


    def scatterplot(self, x_data, y_data, x_label="", y_label="", sizes=[], lists=[], list_of_labels=[], keyword_categories= [], type='Analysis'):
        fig, ax = plt.subplots()

        for one in range(0, len(lists)-1):
            x = []
            y = []
            for ind in range(0, len(lists[one])):
                x.append(lists[one][ind][0])
                y.append(lists[one][ind][1])
            plt.scatter(x, y, label=keyword_categories[one], color=self.colors[one], s=sizes, alpha=0.75)
            for i, txt in enumerate(list_of_labels[one]):
                ax.annotate(txt, (x[i], y[i]))


        # Could also plt scatter with all x and y data

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title('Scatter Plot of the ' + type + ' Model')
        plt.legend()

        # Taken out for running on VM
        # plt.show()

        fig.savefig(os.path.abspath("../graphics_ficino/" + type + "_scatter_" + datetime.utcnow().isoformat('T') + '_' +  '.png'), dpi=450)

    def threeD_plot(self, x_data, y_data, z_data, lists, keyword_categories, type):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for one in range(0, len(lists)):
            x = []
            y = []
            z = []
            for ind in range(0, len(lists[one])):
                x.append(lists[one][ind][0])
                y.append(lists[one][ind][1])
                z.append(lists[one][ind][2])
            ax.scatter(x, y, z, label=keyword_categories[one], color=self.colors[one], alpha=0.75, marker='.')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

        def rotate(angle):
            ax.view_init(azim=angle)

        rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 1), interval=100)
        rot_animation.save('../graphics_ficino/data_rotation_' + datetime.utcnow().isoformat('T') + '_' + type + '.gif', dpi=80, writer='imagemagick')

        # Taken out for running on VM
        # plt.show()
