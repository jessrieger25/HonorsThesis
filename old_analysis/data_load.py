#!/usr/bin/python

import psycopg2
import config
import os
import re


class DatabaseActions:
    @staticmethod
    def insert_book(volumeNum, title):
        """ insert a new vendor into the vendors table """
        sql = """INSERT INTO Books(volumeNum, title)
                 VALUES(%s, %s) RETURNING title;"""
        conn = None
        vendor_id = None
        try:
            # read database configuration
            #         params = config()
            conn_string = "host='localhost' dbname='postgres' user='postgres' password='alpaca'"
            # connect to the PostgreSQL database
            conn = psycopg2.connect(conn_string)
            # create a new cursor
            cur = conn.cursor()
            # execute the INSERT statement
            cur.execute(sql, (1, "Time Machine",))
            # get the generated id back
            BID = cur.fetchone()[0]
            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
        finally:
            if conn is not None:
                conn.close()
            conn.close()
        return BID

    @staticmethod
    def insert_chapter(BID, CID, quote, num_pages):
        """ insert a new vendor into the vendors table """
        sql = """INSERT INTO ChapterInfo(BID, CID, quote, numPages)
                 VALUES(%s, %s, %s, %s) RETURNING quote;"""
        conn = None
        vendor_id = None
        try:
            # read database configuration
            #         params = config()
            conn_string = "host='localhost' dbname='postgres' user='postgres' password='alpaca'"
            # connect to the PostgreSQL database
            conn = psycopg2.connect(conn_string)
            # create a new cursor
            cur = conn.cursor()
            # execute the INSERT statement
            cur.execute(sql, (BID, CID, quote, num_pages,))
            # get the generated id back
            CID = cur.fetchone()[0]
            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
        finally:
            if conn is not None:
                conn.close()
            conn.close()
        return CID

    @staticmethod
    def insert_page(BID, CID, PG):
        """ Insert Page into the database """
        sql = """INSERT INTO PageInfo(BID, CID, PG)
                 VALUES(%s, %s, %s) RETURNING PG;"""
        conn = None
        vendor_id = None
        try:
            # read database configuration
            #         params = config()
            conn_string = "host='localhost' dbname='postgres' user='postgres' password='alpaca'"
            # connect to the PostgreSQL database
            conn = psycopg2.connect(conn_string)
            # create a new cursor
            cur = conn.cursor()
            # execute the INSERT statement
            cur.execute(sql, (BID, CID, PG,))
            # get the generated id back
            PG = cur.fetchone()[0]
            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
        finally:
            if conn is not None:
                conn.close()
            conn.close()
        return PG

    @staticmethod
    def insert_paragraph(BID, CID, PG, ParID, continued, num_lines):
        """ Insert Paragraph into the database """
        sql = """INSERT INTO ParagraphInfo(BID, CID, PG, ParID, continued, numLines)
                 VALUES(%s, %s, %s, %s, %s, %s) RETURNING ParID;"""
        conn = None
        vendor_id = None
        try:
            # read database configuration
            #         params = config()
            conn_string = "host='localhost' dbname='postgres' user='postgres' password='alpaca'"
            # connect to the PostgreSQL database
            conn = psycopg2.connect(conn_string)
            # create a new cursor
            cur = conn.cursor()
            # execute the INSERT statement
            continued = str(continued).upper()
            cur.execute(sql, (BID, CID, PG, ParID, continued, num_lines))
            # get the generated id back
            ParID = cur.fetchone()[0]
            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
        finally:
            if conn is not None:
                conn.close()
            conn.close()
        return ParID

    @staticmethod
    def insert_sentence(BID, CID, PG, ParID, SenID, continued, end_punctuation):
        """ Insert Sentence into the database """
        sql = """INSERT INTO SentenceInfo(BID, CID, PG, ParID, SenID, continued, endPunctuation)
                 VALUES(%s, %s, %s, %s, %s, %s, %s) RETURNING SenID;"""
        conn = None
        vendor_id = None
        try:
            # read database configuration
            #         params = config()
            conn_string = "host='localhost' dbname='postgres' user='postgres' password='alpaca'"
            # connect to the PostgreSQL database
            conn = psycopg2.connect(conn_string)
            # create a new cursor
            cur = conn.cursor()
            # execute the INSERT statement
            continued = str(continued).upper()
            cur.execute(sql, (BID, CID, PG, ParID, SenID, continued, end_punctuation,))
            # get the generated id back
            SenID = cur.fetchone()[0]
            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
        finally:
            if conn is not None:
                conn.close()
            conn.close()
        return SenID

    @staticmethod
    def insert_word(BID, CID, PG, ParID, SenID, WORD):
        """ Insert Word into the database """
        sql = """INSERT INTO WordOccurences(BID, CID, PG, ParID, SenID, WORD)
                 VALUES(%s, %s, %s, %s, %s, %s) RETURNING WORD;"""
        conn = None
        vendor_id = None
        try:
            # read database configuration
            #         params = config()
            conn_string = "host='localhost' dbname='postgres' user='postgres' password='alpaca'"
            # connect to the PostgreSQL database
            conn = psycopg2.connect(conn_string)
            # create a new cursor
            cur = conn.cursor()
            # execute the INSERT statement
            cur.execute(sql, (BID, CID, PG, ParID, SenID, WORD,))
            # get the generated id back
            WORD = cur.fetchone()[0]
            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
        finally:
            if conn is not None:
                conn.close()
            conn.close()
        return WORD


class FileLoaderDriver:

    def load_book(self):
        # Number of lines on a page - randomly decided
        page_length = 30

        # Input data files are available in the "../input/" directory.
        # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

        file_prefix = '/Users/Jess/Desktop/TimeMachine/TimeMachine'
        file_suffix = ".txt"
        paragraph_denoting_string = '\n'

        book_id = 1

        DatabaseActions.insert_book(book_id, "Time Machine")

        titles = ['Introduction', 'The Machine', 'The Time Traveller Returns', 'Time Travelling', 'In the Golden Age',
                  'The Sunset of Mankind', 'A Sudden Shock', 'Explanation', 'The Morlocks', 'When Night Came',
                  'The Palace of Green Porcelain', 'In the Darkness', 'The Trap of the White Sphinx',
                  'The Further Vision',
                  'The Time Traveller’s Return', 'After the Story']

        for chapter_counter in range(1, 2):
            DatabaseActions.insert_chapter(1, chapter_counter, titles[chapter_counter - 1], page_length)
            filename = file_prefix + str(i) + file_suffix
            file = open(filename, 'rt')
            lines = file.readlines()

            page_counter = 1
            paragraph_counter = 1
            paragraph_start_line = 0
            continued_paragraph = False

            sentence_counter = 1
            sentence_string = ""
            continued_sentence = False

            punctuation = re.compile('[\.?!]')
            for line_num in range(0, len(lines)):

                line = lines[line_num]
                if line % page_length == 0:
                    DatabaseActions.insert_page(book_id, chapter_counter, page_counter)
                if line != paragraph_denoting_string:
                    continued_paragraph = True
                if punctuation.match(line[len(line) - 1]) is None:
                    continued_sentence = True
                page_counter += 1

                if line == paragraph_denoting_string:
                    DatabaseActions.insert_paragraph(book_id, chapter_counter, page_counter, paragraph_counter, continued_paragraph, line_num-paragraph_start_line)
                    paragraph_start_line = line_num
                else:
                    line = line.strip()

                punctuation_present = punctuation.search(line)
                if punctuation_present is not None:
                    sentence_string += line[0:punctuation_present.start()]
                    DatabaseActions.insert_sentence(book_id, chapter_counter, page_counter, paragraph_counter, sentence_counter,
                                    continued_sentence,
                                    line[punctuation_present.start()])
                    sentence_counter += 1
                    sentence_string = line[punctuation_present.start():len(line)]
                else:
                    sentence_string += line
                    continued_sentence = False
                    continued_paragraph = False

            file.close()