
from models.word_prep import WordPrep


class KeywordManager:

    def __init__(self):
        self.wp = WordPrep("Dummy data")
        self.keywords_list = self.wp.keywords
        self.keyword_categories = self.wp.keyword_categories
        self.category_colors = {1: 'yellow', 2: 'pink', 3: 'purple', 4: 'turquoise', 5: 'red', 6: 'orange'}