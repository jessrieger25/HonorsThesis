from models.word_prep import WordPrep
from watson_analysis.natural_lang_understanding import NLU
from watson_analysis.tone import ToneAnalyzer


class WatsonDriver():

    def __init__(self):
        self.target = []

    def run(self, file_list):
        for file in file_list:
            # Load data
            self.corpus_raw = ""
            with open(file, "r") as time:
                for line in time.readlines():
                    self.corpus_raw += line.replace('\n', " ")

        tokenized_sen = WordPrep(self.corpus_raw).create_token_sentences()

        for ind in range(0, 1):

            print(tokenized_sen[ind])

            # NLU
            # analysis = NLU().analyze_text(tokenized_sen[ind])
            # self.target.append(NLU().make_vector(analysis))

            # TONE
            tone_results = ToneAnalyzer().analyze_text(tokenized_sen[ind])
            # self.target.append(ToneAnalyzer().make_vector(tone_results))

WatsonDriver().run(["/Users/Jess/PycharmProjects/Honors_Thesis_2/time_machine_skip_gram.txt"])