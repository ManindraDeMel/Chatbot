import sqlite3
import json
import re
import os
"""
The entire purpose of this file is to read the reddit data and compile it to a series of databases with respect to the reddit data's year. 
We add each row of the reddit data given the pass certain parameters (no hyperlinks, certain length etc... ). Once this data has been validated its then checked if it is
possibly a reply to another comment. If so it links to two. However, we cannot simply only include linked or paired data in the database, since original comments or new topics
won't always have replies to their topics. Therefore, there will be a lot of data with no replies. 

The database is effectively a first filter for the data and acts as a buffer. 
It linking data together to then be formally connected together in training_data/get_training_data.py

"""

class DataBase:

    def __init__(self, db_path):
        self.queries = []
        self.row_counter = 0
        self.db_connection = sqlite3.connect(db_path)
        self.db = self.db_connection.cursor() 
        self.failed_queries = 0
        self.rows_added = 0
        self.paired_data = 0

    def initiate_table(self):
        self.db.execute("""CREATE TABLE IF NOT EXISTS conversations (original_comment_id TEXT PRIMARY KEY, reply_comment_id TEXT UNIQUE, original_comment_text TEXT,
                    reply_text TEXT, subreddit TEXT, unix_time INT, upvotes INT)""")

    def find_upvotes(self, pid):
        try: 
            query = f'SELECT upvotes FROM conversations WHERE original_comment_id = {pid} LIMIT 1'
            self.db.execute(query)
            row = self.db.fetchone()
            return row[0] if row else False
        except:
            return False

    def find_status(self, limit):
        self.row_counter += 1
        if (self.row_counter % limit == 0):
            print(f"########\nRows parsed: {self.row_counter}\nRows added: {self.rows_added}\nPaired data: {self.paired_data}\nQueries failed: {self.failed_queries}\n########\n\n")

    def add_query(self, query):
        self.queries.append(query)
        if (len(self.queries) > 1000):
            self.db.execute("BEGIN TRANSACTION")
            for query in self.queries:
                try:
                    self.db.execute(query)
                    self.rows_added += 1
                except:
                    self.failed_queries += 1
            self.db_connection.commit()
            self.queries = []

    def find_original_comment(self, comment_id): 
        try: # Finding links between comments
            query = f'SELECT reply_text FROM conversations WHERE reply_comment_id = "{comment_id}" LIMIT 1'
            self.db.execute(query)
            row = self.db.fetchone()
            return (row[0] if row else False)
        except:
            return False

    def insert_into_database(self, comment_id, original_id, reply, subreddit, created_utc, upvotes, original_comment):
        try:
            if (original_comment):
                query = f'INSERT INTO conversations VALUES("{original_id}", "{comment_id}", "{original_comment}", "{reply}", "{subreddit}", {created_utc}, {upvotes});'
                self.paired_data += 1
            else:
                query = f'INSERT INTO conversations (original_comment_id, reply_comment_id, reply_text, subreddit, unix_time, upvotes) VALUES ("{original_id}", "{comment_id}", "{reply}", "{subreddit}", {created_utc}, {upvotes});'
            self.add_query(query)

        except Exception as e:
            print("Insert problem", e)

    def replace_row(self, comment_id, original_id, reply, subreddit, created_utc, upvotes, original_comment):
        try:
            query = f'UPDATE conversations SET original_comment_id = "{original_id}", reply_comment_id = "{comment_id}", original_comment_text = "{original_comment}", reply_text = "{reply}", subreddit = "{subreddit}", unix_time = {created_utc}, upvotes = {upvotes} WHERE original_comment_id = "{original_id}";'
            self.add_query(query)
        except Exception as e:
            print(e)

    @staticmethod
    def format_data(data):
        return(data.replace("\n", " ").replace("/r", " reddit").replace("'", "''").replace('"', "''")) # Removing unessecary text to stop it from messing with the database

    @staticmethod
    def filter_comment(comment):  # Probably want to add filtration to certain sub-reddits. 
        if (len(comment.split()) > 50) or (len(comment) < 1):
            return False
        elif (len(comment) > 1000):
            return False
        possible_url = re.search("(?P<url>https?://[^\s]+)", comment) # checking for URLS
        if (possible_url):
            return False
        elif (comment == "[deleted]") or (comment == "[removed]"):
            return False
        return True

directories = list(map(lambda x: str(x), range(2015, 2016)))
ignore_files  = []
for directory in directories:
    try:
        os.mkdir(f"database/{directory}")
    except:
        pass
    data_files = os.listdir(f"R:/Chatbot_program/reddit_data/{directory}")
    for file_name in data_files:
        if ".bz" not in file_name:
            if file_name not in ignore_files:
                file_name = file_name.split("RC_")[1]
                year = file_name.split("-")[0]
                database = DataBase(f"database/{directory}/{file_name}.db")
                database.initiate_table()
                # 
                print(f"##############\nWorking on database {file_name}\n##############")
                # 
                with open(r'R:\Chatbot_program\reddit_data\{}\RC_{}'.format(year, file_name), buffering=1500) as f:
                    for row in f:
                        row = json.loads(row)
                        original_id = row['parent_id']     
                        date_created = row['created_utc']
                        comment_id = row['name']
                        upvotes = row['score']
                        subreddit = row['subreddit']
                        #############
                        text = DataBase.format_data(row['body'])
                        original_comment = database.find_original_comment(original_id)
                        #############
                        if upvotes >= 3:
                            if DataBase.filter_comment(text):
                                current_comment_score = database.find_upvotes(original_id)
                                if current_comment_score:
                                    if upvotes > current_comment_score:
                                        database.replace_row(comment_id, original_id, text, subreddit, date_created, upvotes, original_comment)
                                else:
                                    database.insert_into_database(comment_id, original_id, text, subreddit, date_created, upvotes, original_comment)

                        database.find_status(100000)