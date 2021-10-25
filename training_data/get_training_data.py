import sqlite3
import pandas as p
import os

class Extract_data:
    def __init__(self, rows_to_be_pulled, file_names, path):
        self.limit = rows_to_be_pulled
        self.last_time = 0
        self.returned_rows = self.limit
        self.counter = 0
        self.test_append = False
        self.file_names = file_names
        self.db_connection = None
        self.training_directory = path

    def write_file(self, fileName, fileName2):        
        with open(f"{self.training_directory}/{fileName}", "a", encoding="utf8") as f:
            for content in self.df["original_comment_text"].values:
                if content:
                    f.write(content.strip()+"\n")
        with open(f"{self.training_directory}/{fileName2}", "a", encoding="utf8") as f:
            for content in self.df["reply_text"].values:
                if content:
                    f.write(content.strip()+"\n")
    

    def sort_data(self): # Will most likely have different db's for different time periods
        for db_name in self.file_names:
            print(f"#############\n Working on DB {db_name}\n#############")
            self.db_connection = sqlite3.connect(db_name)
            #####
            self.last_time = 0
            self.returned_rows = self.limit
            self.counter = 0
            self.test_append = False
            #####
            while self.returned_rows == self.limit: # If we have less rows then we break out because it means we have reached the end of the database
                query = f"SELECT * FROM conversations WHERE unix_time > {self.last_time} AND original_comment_text NOT NULL ORDER BY unix_time ASC LIMIT {self.limit}" # Order by unix time, and only return a limited amount of rows, we also make sure its pairs of rows. i.e (original comment and a reply)
                self.df = p.read_sql(query, self.db_connection)
                self.last_time = self.df.tail(1)["unix_time"].values[0]
                self.returned_rows = len(self.df) # here we're checking the amount of rows the database returns. Later in the while loop we check if this is equal to the limit and if its less that means that we've reached the end of the database
                if not self.test_append:
                    self.write_file("test_data.original", "test_data.reply")     # The first 10,000 rows will be written to a separate file for testing later on               
                    self.test_append = True
                else:
                    self.write_file("training_data.original", "training_data.reply") # Majority of the data is stored in these two files for training. 
                self.counter += 1
                if (self.counter % 10 == 0):
                    print(self.counter * self.limit, "rows completed so far")
            self.db_connection.close()
                    
path_to_databases = 'R:/Chatbot_program/ChatBot/database'
years = list(map(lambda x:  str(x), range(2007, 2015)))
files = [r"{}/{}/{}".format(path_to_databases, year, file_name) for year in years for file_name in os.listdir(f"{path_to_databases}/{year}")]
Extract_data(10000, files, "training_data/").sort_data()


