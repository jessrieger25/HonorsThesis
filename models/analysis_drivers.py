from models.skip_gram_tutorial import SkipGram
from models.tsne import TSNEVisualizations
from models.glove import Glove
from models.word_prep import WordPrep
from models.bag_of_words import BagOfWords
from nltk.corpus import stopwords
import numpy as np
import json
from models.watson_api.natural_lang_understanding import NLU
from models.watson_api.tone import ToneAnalyzer
from models.lstm_keras import LSTMKeras


class AnalysisDriver():

    def __init__(self, file_list):
        for file in file_list:
            # Load data
            self.corpus_raw = ""
            with open(file, "r") as time:
                for line in time.readlines():
                    self.corpus_raw += line.replace('\n', " ")

            # Convert to Lower
            self.corpus_raw = self.corpus_raw.lower()

        # Initialize word dictionaries
        self.wp = WordPrep(self.corpus_raw)

        # Text management variables
        self.word_count = self.wp.word_count()
        self.eng_stopwords = stopwords.words('english')

        self.model = input("What model would you like to run? s = skipgram, g = glove")
        if self.model == "s":
            self.skip_gram_run()
        if self.model == 'g':
            self.glove_run()
        elif self.model == 'bow':
            self.bag_of_words_run()
        # elif self.model == 'w':
        #     self.watson_sentiment_analysis()

    def skip_gram_run(self):
        sg = SkipGram(self.wp.sen_word_token, self.wp.word_list, self.wp.word2int, self.wp.int2word)
        sg.run()

        with open("/Users/Jess/PycharmProjects/Honors_Thesis_2/models/output.txt", 'w') as f:
            f.write(sg.vectors)

        tsne_model = TSNEVisualizations()
        tsne_model.run(sg.vectors, self.wp.word_list, self.wp.word2int, sizes=self.word_count,
                       separates=self.wp.list_of_list, keywords=self.wp.keywords, type='Skip Gram')

    def glove_run(self):
        g = Glove(self.wp.sen_word_token, self.wp.vocab_list, self.wp.word_list, self.wp.word2int, self.wp.int2word)
        g.run()

        with open("/Users/Jess/PycharmProjects/Honors_Thesis_2/models/output.txt", 'w') as f:
            f.write(g.embedding_matrix)

        print(g.embedding_matrix)
        tsne_model = TSNEVisualizations()
        tsne_model.run(g.embedding_matrix, self.wp.word_list, self.wp.word2int, sizes=self.word_count, separates=self.wp.list_of_list, keywords=self.wp.keywords, type='Glove')

    def bag_of_words_run(self):
        bow = BagOfWords()

        # Basic call to create bag of words with english stop words removed.
        bag_of_words, features = bow.create_bag_of_words(self.wp.sen_word_token, stop_words=self.eng_stopwords, stem=True)

        # Diplaying the bag of words and the features
        print(bag_of_words)
        print(features)

        # Basic call to get word counts without additional parameters.
        word_counts = bow.get_word_counts(bag_of_words, features)

        # Displaying the word counts
        print(word_counts)

        # Call to create topics from the words present in the document.
        processed_keywords, processed_doctopic = bow.create_topics(bag_of_words, features)

        # Displaying results:
        print(processed_doctopic.shape)
        print(processed_keywords)

        # Call with all parameters
        # look for 10 topics, include 4 words in each. We can play with this to see where we get the best
        # results.
        processed_keywords, processed_doctopic = bow.create_topics(bag_of_words,
                                                               features,
                                                               N_TOPICS=10,
                                                               N_TOP_WORDS=4)

    def watson_sentiment_analysis(self):
        target_labels = np.ndarray([len(self.wp.sen_word_token), 13], dtype='float')
        nlu_analysis = []

        # From already gen file
        with open("/Users/Jess/PycharmProjects/Honors_Thesis_2/watson_analysis/result_jsons/nlu_results.txt",
                  "r") as text:
            nlu = json.load(text)

        for ind in range(0, len(self.wp.sen_word_token)):
            # NLU
            analysis = NLU().analyze_text(self.wp.sen_word_token[ind])
            nlu_analysis.append(NLU().make_vector(analysis))
            print(analysis)

        tone_results = ToneAnalyzer().analyze_text(self.wp.sen_word_token)

        tone_vecs = ToneAnalyzer().make_vector(tone_results)
        if len(tone_vecs) != len(nlu_analysis):
            raise Exception

        for ind in range(0, len(tone_vecs)):
            combined_vec = []
            combined_vec.extend(NLU().make_vector(nlu[ind]))
            combined_vec.extend(tone_vecs[ind])
            np.append(target_labels, [combined_vec], axis=0)

        print(target_labels)
        LSTMKeras(self.wp.sen_word_token, target_labels).run()

# AnalysisDriver(["/Users/Jess/PycharmProjects/Honors_Thesis_2/time_machine_skip_gram.txt"])

AnalysisDriver(["/Users/Jess/PycharmProjects/Honors_Thesis_2/ficino/book_1_part_1.txt"])







