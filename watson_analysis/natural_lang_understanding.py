from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, SentimentOptions, EmotionOptions
import json


class NLU():
    def __init__(self):
        self.natural_language_understanding = NaturalLanguageUnderstandingV1(
            version='2018-03-19',
            username='46ec366d-6845-478d-8b13-9bcb6c578d19',
            password='42GL7BMCra1k'
        )

    def analyze_text(self, text):
        response = self.natural_language_understanding.analyze(
          text=text,
          features=Features(
            sentiment=SentimentOptions(),
            emotion=EmotionOptions())).get_result()

        print(json.dumps(response, indent=2))
        return response

    def make_vector(self, returned_analysis):
        vector = []
        for key, value in returned_analysis['emotion']['document']['emotion'].items():
            vector.append(value)

        vector.append(returned_analysis['sentiment']['document']['score'])

        print(vector)
        return vector

