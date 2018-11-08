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
        if len(text) > 15:
            response = self.natural_language_understanding.analyze(
              text=text,
              features=Features(
                sentiment=SentimentOptions(),
                emotion=EmotionOptions())).get_result()

            print(json.dumps(response, indent=2))
        else:
            response = {"usage": {"text_units": 1, "text_characters": 65, "features": 2}, "sentiment": {"document": {"score": 0.0, "label": "neutral"}}, "language": "en", "emotion": {"document": {"emotion": {"sadness": 0.0, "joy": 0.0, "fear": 0.0, "disgust": 0.0, "anger": 0.0}}}}
        return response

    def make_vector(self, returned_analysis):
        vector = []
        for key, value in returned_analysis['emotion']['document']['emotion'].items():
            vector.append(value)

        vector.append(returned_analysis['sentiment']['document']['score'])

        print(vector)
        return vector

