import re


class Book_structure:

    def load_book(self):

        sentences = []

        # Number of lines on a page - randomly decided
        page_length = 30

        # Input data files are available in the "../input/" directory.
        # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

        file_prefix = '/Users/Jess/Desktop/TimeMachine/TimeMachine'
        file_suffix = ".txt"
        paragraph_denoting_string = '\n'

        book_id = 1

        titles = ['Introduction', 'The Machine', 'The Time Traveller Returns', 'Time Travelling', 'In the Golden Age',
                  'The Sunset of Mankind', 'A Sudden Shock', 'Explanation', 'The Morlocks', 'When Night Came',
                  'The Palace of Green Porcelain', 'In the Darkness', 'The Trap of the White Sphinx',
                  'The Further Vision',
                  'The Time Traveller’s Return', 'After the Story']

        for chapter_counter in range(1, 2):
            filename = file_prefix + str(chapter_counter) + file_suffix
            file = open(filename, 'rt')
            lines = file.readlines()

            page_counter = 0
            paragraph_counter = 1
            paragraph_start_line = 0
            continued_paragraph = False

            sentence_counter = 1
            sentence_string = ""
            continued_sentence = False

            punctuation = re.compile('[\.?!]')
            for line_num in range(0, len(lines)):

                line = lines[line_num]
                if line_num % page_length == 0:
                    page_counter += 1
                    paragraph_counter = 1
                    sentence_counter = 1
                    if line != paragraph_denoting_string:
                        continued_paragraph = True
                    if punctuation.match(line[len(line) - 1]) is None:
                        continued_sentence = True

                if line == paragraph_denoting_string:
                    paragraph_start_line = line_num
                    paragraph_counter += 1
                else:
                    line = line.strip()

                punctuation_present = punctuation.search(line)
                if punctuation_present is not None:
                    sentence_string += line[0:punctuation_present.start() + 1]
                    sentences.append({'sentence': sentence_string.lower(), 'chapter': chapter_counter, 'page': page_counter, 'paragraph': paragraph_counter, 'sentence_num': sentence_counter, 'continued sentence': continued_sentence, 'continued paragraph': continued_paragraph})
                    sentence_counter += 1
                    sentence_string = line[punctuation_present.start() - 1:len(line)]
                else:
                    sentence_string += line
                    continued_sentence = False
                    continued_paragraph = False
            file.close()
            return sentences


