from watson_developer_cloud import ToneAnalyzerV3
import json


class ToneAnalyzer:

    def __init__(self):

        self.tone_to_int = {
            "anger": 0,
            "confident": 1,
            "fear": 2,
            "joy": 3,
            "analytical": 4,
            "tentative": 5,
            "sadness": 6
        }

        self.tone_analyzer = ToneAnalyzerV3(
            version='2017-09-21',
            iam_apikey='S9ZvtlAibhB78AhVEuLfiUubZULn8bOzaPdG8K5MOLys',
            url='https://gateway-wdc.watsonplatform.net/tone-analyzer/api'
        )

    def analyze_text(self, text):
        tone_analyzer = ToneAnalyzerV3(
            version='2017-09-21',
            iam_apikey='42akR3iVWG3M2p1JCOwmvM_U-W8anY6bdkg8M5ncH4jf',
            url='https://gateway-wdc.watsonplatform.net/tone-analyzer/api'
        )

        text = 'Team, I know that times are tough! Product ' \
               'sales have been disappointing for the past three ' \
               'quarters. We have a competitive product, but we ' \
               'need to do a better job of selling it!'

        tone_analysis = tone_analyzer.tone(
            {'text': text},
            'application/json',
            sentences=True
        ).get_result()
        print(json.dumps(tone_analysis, indent=2))
        # tone_analysis = self.tone_analyzer.tone(
        #     {'text': text},
        #     'application/json',
        #     sentences=True
        # ).get_result()
        # print(json.dumps(tone_analysis, indent=2))
        return tone_analysis

    def make_vector(self, returned_analysis):
        vectors = []

        for sen in returned_analysis['sentences_tone']:
            vector = [0, 0, 0, 0, 0, 0, 0]
            for tone in sen['tones']:
                vector[self.tone_to_int[tone['tone_id']]] = tone['score']
            vectors.append(vector)

        print(vectors)
        return vectors
