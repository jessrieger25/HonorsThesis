from models.watson_api.natural_lang_understanding import NLU
from models.watson_api.tone import ToneAnalyzer
import json
from models.watson_api.lstm_keras import LSTMKeras
import numpy as np


class WatsonDriver():

    def __init__(self, tokenized_sentences ):
        self.tokenized_sentences = tokenized_sentences
        self.target = np.ndarray([len(tokenized_sentences), 13], dtype='float')
        self.nlu_analysis = []
        self.tone_results = {}

    def run(self):

        # From already gen file
        with open("/Users/Jess/PycharmProjects/Honors_Thesis_2/watson_analysis/result_jsons/nlu_results.txt",
                  "r") as text:
            nlu = json.load(text)

        for ind in range(0, len(self.tokenized_sentences)):
            # NLU
            analysis = NLU().analyze_text(self.tokenized_sentences[ind])
            self.nlu_analysis.append(NLU().make_vector(analysis))
            print(analysis)

        self.tone_results = ToneAnalyzer().analyze_text(self.tokenized_sentences)

        tone_vecs = ToneAnalyzer().make_vector(self.tone_results)
        if len(tone_vecs) != len(self.nlu_analysis):
            raise Exception

        for ind in range(0, len(tone_vecs)):
            combined_vec = []
            combined_vec.extend(NLU().make_vector(nlu[ind]))
            combined_vec.extend(tone_vecs[ind])
            np.append(self.target, [combined_vec], axis=0)
            # self.target[ind].extend(tone_vecs[ind])
        print(self.target)
        LSTMKeras(self.tokenized_sentences, self.target).run()

WatsonDriver().run(["/Users/Jess/PycharmProjects/Honors_Thesis_2/time_machine_skip_gram.txt"])