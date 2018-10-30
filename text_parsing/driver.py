from text_parsing.positional_relations import PositionalRelations
import pprint
from text_parsing.book_structure import Book_structure


class TextAnalysisDriver:

    def __init__(self):
        self.commands = input("What operation would you like to do? \n"
                              "a = average word separation\n"
                              "wr = within range\n"
                              "p = print surrouding words\n"
                              "s = find relative word positions within structure")

        if self.commands == "wr":
            word = input('What word would you like to use?')
            range = input('What range would you like to use?')
            pprint.pprint(PositionalRelations().within_range(word, range))

        if self.commands == 'a':
            word = input('What word would you like to track?')
            pprint.pprint(PositionalRelations().find_average_dist(word))
        if self.commands == 'p':
            word = input('What word would you like to use?')
            range = input('What range would you like to use?')
            pprint.pprint(PositionalRelations().print_surrounding_window(word, int(range)))
        if self.commands == 's':
            book = Book_structure().load_book()
            word = input('What main word would you like to use?')
            other = input('What other word?')
            pprint.pprint(PositionalRelations().structural_relations(word, other, book))
        if self.commands == 'c':
            PositionalRelations().get_counts()

TextAnalysisDriver()
