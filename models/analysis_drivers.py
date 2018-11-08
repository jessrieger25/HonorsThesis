from models.skip_gram_tutorial import SkipGram
from models.tsne import TSNEVisualizations
from models.glove import Glove
from models.word_prep import WordPrep
from models.bag_of_words import BagOfWords
import numpy as np
import json
import os
import sys
from models.watson_api.natural_lang_understanding import NLU
from models.watson_api.tone import ToneAnalyzer
from models.lstm_keras import LSTMKeras
import matplotlib.pyplot as plt
from datetime import datetime

'''
    Analysis Driver Class

    This class drives the Skip Gram, Watson, and GloVe Analysis of the text. It calls the respective programs
    and then invokes the TSNE script to graph the resulting matrix.
    '''
class AnalysisDriver():

    def __init__(self, file_list):

        # Initialize the Word Prep class - creates keyword lists and tokenizes the text.
        self.wp = WordPrep(file_list)

        #"What model would you like to run? s = skipgram, g = glove, w = watson, bow = bag of words"
        self.model = sys.argv[1]
        if self.model == "s":
            self.skip_gram_run()
        if self.model == 'g':
            self.glove_run()
        elif self.model == 'bow':
            self.bag_of_words_run()
        elif self.model == 'w':
            self.watson_sentiment_analysis()

    def skip_gram_run(self):
        sg = SkipGram(self.wp.word_list, self.wp.word2int, self.wp.keywords)
        sg.run()

        tsne_model = TSNEVisualizations()
        tsne_model.run(sg.vectors, self.wp.word_list, self.wp.word2int, sizes=self.wp.word_count(),
                       separates=self.wp.list_of_list, keywords=self.wp.keywords, keyword_categories=self.wp.keyword_categories, type='Skip Gram')

    def glove_run(self):
        g = Glove(self.wp.vocab_list, self.wp.word_list, self.wp.word2int, self.wp.int2word, self.wp.keywords)
        g.run()
        print(len(g.embedding_matrix))
        tsne_model = TSNEVisualizations()
        tsne_model.run(g.embedding_matrix, self.wp.word_list, self.wp.word2int, sizes=self.wp.word_count(), separates=self.wp.list_of_list, keywords=self.wp.keywords, keyword_categories=self.wp.keyword_categories, type='Glove')

    def bag_of_words_run(self):
        bow = BagOfWords()

        # Basic call to create bag of words with english stop words removed.
        bag_of_words, features = bow.create_bag_of_words(self.wp.tokenized_sentences, stem=True)

        # Diplaying the bag of words and the features
        print(bag_of_words)
        print(features)

        # Basic call to get word counts without additional parameters.
        word_counts = bow.get_word_counts(bag_of_words, features)

        # Displaying the word counts
        print(word_counts)

        # Pie Chart Example

        labels = []
        sizes = []
        for key, value in word_counts.items():
            sizes.append((value / 51) * 100)
            labels.append(key)

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:

        # Creating the list for the pie chart

        explode_list = list()
        for i in range(0, len(labels)):
            if i == 2:
                explode_list.append(0.1)
            else:
                explode_list.append(0)

        explode = (explode_list)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        fig1.savefig(os.path.abspath("../graphics_ficino/bow_" + datetime.utcnow().isoformat('T') +  '.png'), dpi=450)


        # # Call to create topics from the words present in the document.
        # processed_keywords, processed_doctopic = bow.create_topics(bag_of_words, features)
        #
        # # Displaying results:
        # print(processed_doctopic.shape)
        # print(processed_keywords)

        # # Call with all parameters
        # # look for 10 topics, include 4 words in each. We can play with this to see where we get the best
        # # results.
        # processed_keywords, processed_doctopic = bow.create_topics(bag_of_words,
        #                                                        features,
        #                                                        N_TOPICS=10,
        #                                                        N_TOP_WORDS=4)

    def watson_sentiment_analysis(self):
        target_labels = np.ndarray([len(self.wp.tokenized_sentences), 13], dtype='float')
        tone_num = -1
        for group in range(0, len(self.wp.tokenized_sentences), 98):
            print("This is group")
            print(group)
            tone_num += 1
            corpus = ""
            for sen in range(group, group+98):
                if sen < len(self.wp.tokenized_sentences):
                    sentence = self.wp.tokenized_sentences[sen]
                    for word in sentence:
                        corpus += word
                    corpus += "  "
                else:
                    break

            # Tone Analysis: DO NOT UNCOMMENT LIGHTLY
            if len(corpus) > 0:

                self.run_tone_analysis(corpus, tone_num)

                with open(os.path.abspath("./watson_api/result_jsons/tone_results_" + str(tone_num) + ".txt"), 'r') as tone:
                    tone_results = json.load(tone)
                tone_vecs = ToneAnalyzer().make_vector(tone_results)

                print("This is tone results")
                print(tone_results)

                # Call analysis: DO NOT UNCOMMENT LIGHTLY
                self.run_nlu(tone_results)

                # From already gen file
                with open(os.path.abspath("./watson_api/result_jsons/nlu_results.txt"),
                          "r") as text:
                    nlu = json.load(text)

                print("This is nlu")
                print(nlu)

                if len(tone_vecs) != len(nlu):
                    raise Exception

                for ind in range(0, len(tone_vecs)):
                    combined_vec = []
                    combined_vec.extend(NLU().make_vector(nlu[ind]))
                    combined_vec.extend(tone_vecs[ind])
                    np.append(target_labels, [combined_vec], axis=0)

            embedding_layer = LSTMKeras(self.wp.tokenized_sentences, target_labels, self.wp.vocab_list).run()
            print(embedding_layer[0])
            tsne_model = TSNEVisualizations()
            tsne_model.run(embedding_layer[0], self.wp.word_list, self.wp.word2int, sizes=self.wp.word_count(),
                           separates=self.wp.list_of_list, keywords=self.wp.keywords, keyword_categories=self.wp.keyword_categories, type='Watson')

    def run_tone_analysis(self, corpus, number):
        print("Running tone analysis")
        tone_results = ToneAnalyzer().analyze_text(corpus)

        with open(os.path.abspath("./watson_api/result_jsons/tone_results_" + str(number) + ".txt"), 'w') as tone:
            json.dump(tone_results, tone)

    def run_nlu(self, tone_results):
        print("Running Natural Language Analysis")
        # NLU Analysis
        nlu_analysis = []

        with open(os.path.abspath("./watson_api/result_jsons/nlu_results.txt"), 'a') as nlu:
            nlu.write('[')
            for ind in range(0, len(tone_results['sentences_tone'])):
                # NLU
                analysis = NLU().analyze_text(tone_results['sentences_tone'][ind]['text'])
                nlu_analysis.append(NLU().make_vector(analysis))
                print(analysis)
                json.dump(analysis, nlu)
                if ind != len(tone_results['sentences_tone'])-1:
                    nlu.write(',')
            nlu.write(']')

# AnalysisDriver([os.path.abspath("../ficino/short_tester.txt")])
#
AnalysisDriver([os.path.abspath("../ficino/book_1-4.txt")])







