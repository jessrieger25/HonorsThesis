
from models.word_prep import WordPrep
from models.utilities import group_phrases, convert_phrases, adjust_keywords

class KeywordManager:

    def __init__(self):

        # DOES NOT MATTER WHICH FILE GOES IN
        self.wp = WordPrep(["/Users/Jess/PycharmProjects/Honors_Thesis_2/ficino/short_tester.txt"])
        self.keywords_list = adjust_keywords(self.wp.keywords)
        self.keyword_categories = self.wp.keyword_categories
        self.category_colors = {0: 'yellow', 1: 'pink', 2: 'purple', 3: 'turquoise', 4: 'red', 5: 'orange'}