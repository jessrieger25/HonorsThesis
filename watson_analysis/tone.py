from watson_developer_cloud import ToneAnalyzerV3
import json


class ToneAnalyzer:

    def __init__(self):

        self.tone_analyzer = ToneAnalyzerV3(
            version='2017-09-21',
            iam_apikey='S9ZvtlAibhB78AhVEuLfiUubZULn8bOzaPdG8K5MOLys',
            url='https://gateway-wdc.watsonplatform.net/tone-analyzer/api'
        )

    def analyze_text(self, text):

        tone_analysis = self.tone_analyzer.tone(
            {'text': text},
            'application/json'
        ).get_result()
        print(json.dumps(tone_analysis, indent=2))
        return tone_analysis

    def make_vector(self, returned_analysis):
        vector = []
        for key, value in returned_analysis['emotion']['document']['emotion'].items():
            vector.append(value)

        vector.append(returned_analysis['sentiment']['document']['score'])

        print(vector)
        return vector
