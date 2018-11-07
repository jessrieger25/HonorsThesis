
from models.word_prep import WordPrep


class KeywordManager:

    def __init__(self):
        self.wp = WordPrep("Dummy data")
        self.keywords_list = self.wp.keywords
        self.keyword_categories = self.wp.keyword_categories
        self.category_colors = {0: 'yellow', 1: 'pink', 2: 'purple', 3: 'turquoise', 4: 'red', 5: 'orange'}