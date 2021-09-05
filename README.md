# Kali the deep learning chatbot
https://kalichatbotblog.000webhostapp.com/
##### Python 3.9 (other versions not tested)
If you do not know how to use git LFS to clone large files please download the repo at the following google drive link: https://drive.google.com/drive/folders/1D9cnckHW95dYht2fonaEM20V8JEkAXgK?usp=sharing
### External Modules used
- tensorflow 2.6.0
```shell
py -3.9 -m pip install tensorflow
```
- tensorflow_addons 0.14.0
```shell
py -3.9 -m pip install tensorflow_addons
```
- pandas 1.1.4
```shell
py -3.9 -m pip install pandas
```
- sklearn
```shell
py -3.9 -m pip install sklearn
```
- [Cuda](https://www.youtube.com/watch?v=cL05xtTocmY) (Recommended but optional)

# Understanding the concepts behind the Chatbot
This deep learning chatbot utilizes Neural Machine translation (NMT), Long-short term memory units which make up the network and the addition of the attention layer to pick out important words in a given sentence. The model first encodes the input into comprehensible numerical values for the network, once passed through the network it is then passed to the attention layer and finally to the decoder which returns the chatbot's english reply. 

### Running the Chatbot via Desktop Application

```shell
py -3.9 run.py
```

The provided run.py file in the root directory of the project utilizes tkinter to create a basic GUI for you to interact with the chatbot. Once the file is run a tkinter application should open:

![interaction image](https://cdn.discordapp.com/attachments/715926471159578667/883694859247054929/unknown.png "tkinter example")

The application then acts somewhat like a generic messenger application with the message box at the bottom and the scrollable conversation updated and shown at the top.

### Running the Chatbot via Console App
As provided in the run_console.py file, running the chatbot in a console app is extremely easy. It's simply comprised of this code:

```python
import pre_process as chatbot
chatbot.restore_latest_state()
while True:
    user_input = input("> ")
    print(chatbot.reply(u"{}".format(user_input)))
```
The conversation will now be displayed in the console similar to the desktop application. Additionally feel free to experiment with extensions or add more functionality to the console application however you like.

![console example image](https://media.discordapp.net/attachments/715926471159578667/883697980899741697/unknown.png "console example")

Ending the program will just require you to close the console, or if you want, some extra functionality can be added to the given code to exit upon a button press, user input (such as 'exit') or anything similar

### Training the Chatbot
Training the chatbot, is provided with the train.py file found in the user directory. In the console debugging logs will appear similar to:

![training example](https://media.discordapp.net/attachments/715926471159578667/883648362509910016/unknown.png "training debug example")

The loss represents the accuracy of the network, whilst the epochs and batches represent the different portions of the data the chatbot is being trained on. 

# Chatbot parameters 
Located in chatbot.py several adjustable parameters can be found. Notably: 
```python
CONST_TRAINING_CHECKPOINT_DIRECTORY = "training_checkpoints/"
CONST_TRAINING_FILES_DIRECTORIES = ("training_data/training_data.original", "training_data/training_data.reply")
```
Where the checkpoint directory is where you want the chatbot to save it's state in the training process and the training files directory is the directory that contain the .original and .reply files for training.

Futhermore parameters for the training process itself include:
```python
CONST_BUFFER_SIZE = 32000 
# limits how much we read from the IO/Stream. We wouldn't want a buffer overflow...
CONST_BATCH_SIZE = 32 
# The batch sizes can vary depending on the computation power of your computer
dataset_limit = 30000  # Limit for dataset sizes
```
Depending on the capability of your computer these numbers can be increased and decreased accordingly. If you find your computer often crashing, reducing the batch size and the dataset limit may solve the issue.

## Finding the right Dataset
Finding the right dataset is crucial to the overall success of the project. Optimistically, you want to have ~250,000 individual conversations at the very least to attain a somewhat realistic deep learning chatbot. I suggest using the dataset I used [reddit data](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/). Around 1.5 TB is needed for the data and the databases which filter through the data. 


### Filtering the data into a database
Given you have a dataset filled with original topics / starting messages and one to many replies to these topics, we have to first sort through this dataset and link pairs of these original topics and replies together in a SQlite database. In addition to this, we will also do some initial filtration, such as removing hyperlinks, certain words, sentence length limits etc...

This purpose is fulfilled in the gen_database.py file found at database/gen_database.py. The creates a database for in my instance, the [reddit data](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/) i'm using to train my chatbot. The database sorts through all this data and pairs comments with other comments which can then be used for training the chatbot. 

After completion you should have a database with a structure similar to:

![Database structured example](https://media.discordapp.net/attachments/715926471159578667/883648685819437057/unknown.png "Database example")

Filled with data that should look like:

![Database filled example](https://media.discordapp.net/attachments/715926471159578667/883648921447043072/unknown.png "Filled Database")

Futhermore, if you're using my dataset, each month of data will be separated out into separate databases as depicted:
![Database filled example](https://cdn.discordapp.com/attachments/715926471159578667/883666219885023282/unknown.png "Filled Database")

### Pairing the data into different files
Once the data has been inserted into the database we need to look for all the pairs of conversations in the database and separate them into different files. Where .original depicts the start comment / message and .reply depicts the associated reply to that original message. This is done by the get_training_data.py file which reads all the provided databases and splits it into a small portion of test_data for after training and the rest of the filtered data is added towards the training_data files:

![training data example](https://media.discordapp.net/attachments/715926471159578667/883648572480966706/unknown.png "training data example")


## Training data parameters
Since filtration is completed in the database insertion, parameters to this filtration can be deducted or added.

In gen_database.py the filters for each sentence can be found in the following method:
```python
    @staticmethod
    def filter_comment(comment):  # Could add filtration for sub-reddit's. 
        if (len(comment.split()) > 50) or (len(comment) < 1):
            return False
        elif (len(comment) > 1000):
            return False
        possible_url = re.search("(?P<url>https?://[^\s]+)",comment) # checking for URLS
        if (possible_url):
            return False
        elif (comment == "[deleted]") or (comment == "[removed]"):
            return False
        return True
```
Currently the database filters out URL's and makes sure the message length is appropriate and exists. 

Similarly in the get_training_data.py file also has path parameters that need to be addressed
```python
header = r'D:/Data/ChatBot/database/'
Extract_data(10000, [header + r'2015-01.db', header + r'2015-02.db']).sort_data()
```
The header is the directory of all the databases the generator file has created. Futhermore, the second parameter of the Extract data class will need to be modified depending on the amount of databases created. I will most likely optimize this in a future update so python simply reads the filenames in the header directory with the extension of .db. 
