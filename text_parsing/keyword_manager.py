
from models.word_prep import WordPrep


class KeywordManager:

    def __init__(self):
        self.keywords_list = WordPrep("No real corpus").keywords
        self.category_colors = {1: 'yellow', 2: 'pink', 3: 'purple', 4: 'turquoise', 5: 'red', 6: 'orange'}