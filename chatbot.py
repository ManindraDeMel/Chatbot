import sqlite3
from numpy import extract
import pandas as p
class Extract_data:
    def __init__(self, file_names):
        self.limit = 5000
        self.last_time = 0
        self.current_length = self.limit
        self.counter = 0
        self.test_done = False
        self.file_names = file_names
    
    def write_file(self, fileName, fileName2):
        with open(fileName, "a", encoding="utf8") as f:
            for content in self.df["original_comment_text"].values:
                if content:
                    f.write(content+"\n")
        with open(fileName2, "a", encoding="utf8") as f:
            for content in self.df["reply_text"].values:
                if content:
                    f.write(content+"\n")
    
    def sort_data(self): # Will most likely have different db's for different time periods
        for db_name in self.file_names:
            self.db_connection = sqlite3.connect(db_name)
            while self.current_length == self.limit:
                query = f"SELECT * FROM conversations WHERE unix_time > {self.last_time} AND original_comment_text NOT NULL ORDER BY unix_time ASC LIMIT {self.limit}"
                self.df = p.read_sql(query, self.db_connection)
                self.last_time = self.df.tail(1)["unix_time"].values[0]
                self.current_length = len(self.df)
                if not self.test_done: # Update the limit if necessary here
                    self.write_file("test.from", "test.to")                    
                    self.test_done = True
                else:
                    self.write_file("train.from", "train.to")
                self.counter += 1
                if (self.counter % 20 == 0):
                    print(self.counter * self.limit, "rows completed so far")

Extract_data(['2015-01.db']).sort_data()
