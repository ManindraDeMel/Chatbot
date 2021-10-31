import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Stop annoying tensorflow information debug logs
import tensorflow as tf
import tensorflow_addons as tfa
import unicodedata
import re
from sklearn.model_selection import train_test_split
import io

"""
This file contains the entire model for my project. It involves the preperation of the data, the encoder, the decoder,
the training process, and the definition of the loss function which calculates the total error of each iteration. There are 
three major components to this deep learning chatbot. 

First we make sure to tokenize the data properly, and add futher formatting in 
addition to prior formatting. A method must be made to then convert these inputs into a numerical value, which is the purpose of the
encoder, this is then passed through the network which is comprised of several Long short-term memory units (LSTM) which are very similar to
classical recurrent neural networks (RNN) used for natural language processing (NMT). 

However, unlike RNNs, LSTM's don't suffer from a concept known as the
vanishing gradient problem, which not only stagnates the training of the network but also tends to forget earlier phrases in the sentence. In order to further compensate
for the short term memory loss, we also use attention. 

Attention as the name suggests makes the chatbot react attentively to specific phrases it may find valuable or important to the overall message in the user's input. (this is further explained in the attention method below)
We all these components comprised and passed through the network, it is then decoded. The decoding layer is what actually
holds the attention and again, as the name suggests, it decodes the numerical outputs of the chatbot and then converts it into text to be returned to the user. 
"""



CONST_TRAINING_CHECKPOINT_DIRECTORY = "training_checkpoints/"
CONST_TRAINING_FILES_DIRECTORIES = ("training_data/training_data.original", "training_data/training_data.reply")
"""
Format_dataset clarifies and formats the user input and the training and test data. 
"""
class Format_dataset:
    @staticmethod
    def format_sentence(sentence) -> str:
        sentence = Format_dataset.unicode_to_ascii(sentence.lower().strip())
        # creating a space between a words and punctuation.
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence).strip()
        # We start and end each sentence with a token as given. 
        sentence = '<start> ' + sentence + ' <end>'
        return sentence

    @staticmethod
    def read_data(original_path, reply_path, limit) -> list:        
        original_lines = io.open(original_path, encoding='UTF-8').read().strip().split("\n") # Reading from each of the files
        reply_lines = io.open(reply_path, encoding='UTF-8').read().strip().split("\n")
        #############
        if (len(original_lines) <= len(reply_lines)):
            sentence_pairs = [[Format_dataset.format_sentence(original_lines[l]), Format_dataset.format_sentence(reply_lines[l])] for l in range(len(original_lines[:limit]))]
        else:
            sentence_pairs = [[Format_dataset.format_sentence(original_lines[l]), Format_dataset.format_sentence(reply_lines[l])] for l in range(len(reply_lines[:limit]))]
        #############        
        return zip(*sentence_pairs)

    @staticmethod # basically, here we're assinging tokens to specific words. 
    def tokenize(lang: list):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
        # This derives information from the sentence I.e the word_counts, the ID's etc..
        tokenizer.fit_on_texts(lang)
        # converts a list of sentences [sentences_1, sentence_2, sentence_3, ......,] to a multi-dimensional array of corresponding integer ids of words [[id_w1, id_w2], [id_w3], ...., [id_wn]]
        tensor = tokenizer.texts_to_sequences(lang)
        # Adds padding to the sentences to keep a fixed input. This is not the best solution. Having a dynamic or bi-directional RNN would be more ideal of a solution.
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        return tensor, tokenizer

    @staticmethod
    # Convert to ASCII, inspired from: 
    # https://www.delftstack.com/howto/python/python-unicode-to-string/#:~:text=In%20summary%2C%20to%20convert%20Unicode,do%20not%20have%20ASCII%20counterparts.
    def unicode_to_ascii(sentence): 
        return ''.join(char for char in unicodedata.normalize('NFD', sentence) if unicodedata.category(char) != 'Mn')

    @staticmethod
    # Utilizing the above methods to form a dataset
    def load_dataset(original_file, reply_file, dataset_limit=None):
        reply, original_comment = Format_dataset.read_data(original_file, reply_file, dataset_limit)
        original_tensor, original_tokenizer = Format_dataset.tokenize(original_comment)
        reply_tensor, reply_tokenizer = Format_dataset.tokenize(reply)
        return original_tensor, reply_tensor, original_tokenizer, reply_tokenizer


    def call(self, dataset_limit, BUFFER_SIZE, batch_size): # This is what is returned upon defining the class. I.E this methods utilizes the other private methods to return an overall output
        original_file, reply_file = CONST_TRAINING_FILES_DIRECTORIES
        original_tensor, reply_tensor, self.original_tokenizer, self.reply_tokenizer = Format_dataset.load_dataset(original_file, reply_file, dataset_limit)
        ######################## Split arrays or matrices into random train and test subsets
        original_tensor_train, original_tensor_val, reply_tensor_train, reply_tensor_val = train_test_split(original_tensor, reply_tensor, test_size=0.2)  
        ########################
        train_dataset = tf.data.Dataset.from_tensor_slices((original_tensor_train, reply_tensor_train))  # Creating a dataset object for the training data
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)
        val_dataset = tf.data.Dataset.from_tensor_slices((original_tensor_val, reply_tensor_val)) 
        val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

        return train_dataset, val_dataset, self.original_tokenizer, self.reply_tokenizer

"""
The encoder object is responsible for both turning the training data; words, into numbers that the network can parse.
"""
class Encoder(tf.keras.Model):
    def __init__(self, batch_size, input_layer_shape, chatbot_memory_size, units):        
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.units = units # The amount of units/lstm's in the network
        self.input_layer = tf.keras.layers.Embedding(chatbot_memory_size, input_layer_shape)
        # Below is the initialization of the LSTM layer involved in the encoder. This means that the encoder also helps by storing some memory of the message,
        # thus helping with the training process. 
        self.lstm_layer = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def initialize_hidden_layers(self) -> list: # Initially fill the hidden layers with 0's
        return [tf.zeros((self.batch_size, self.units)), tf.zeros((self.batch_size, self.units))]

    def call(self, x, hidden) -> tuple: # encode the words
        return iter(self.lstm_layer(self.input_layer(x), initial_state=hidden))

"""
The decoder, similar to the encoder converts one instance to another. In this case, we convert the output of the network (numerical) 
into words, punctuations etc... 
"""
class Decoder(tf.keras.Model):
    def __init__(self, batch_size, input_layer_dim, memory_size, units, attention_type='luong') -> None:        
        super(Decoder, self).__init__()
        self.batch_size = batch_size # each batch size
        self.units = units # The amount of units/lstm's in the network
        self.attention_type = attention_type # two types of attention types, explained in the attention method 
        self.input_layer = tf.keras.layers.Embedding(memory_size, input_layer_dim) 
        self.output_layer = tf.keras.layers.Dense(memory_size) 
        self.network = tf.keras.layers.LSTMCell(self.units) # Initialize the LSTM network
        self.batch = tfa.seq2seq.sampler.TrainingSampler() # A batch from the data.
        self.attention_layer = Decoder.attention(self.units, None, self.batch_size*[max_length_input], self.attention_type) # The attention layer which is then added to the decoder
        self.rnn_network_attention = self.add_attention() # This adds the attention layer to the RNN/LSTM
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_network_attention, sampler=self.batch, output_layer=self.output_layer) # The decoder 

    """
    What does Attention do for the chatbot? Attention lets the chatbot focus on specific parts of an input and it ignores other 
    parts of the sentence. For example if we have a given sentence 'The tree fell unexpectedly short.' we tend to focus on words like 'tree'
    and 'fell' and tend to ignore words such as 'the' which don't add much to the overall meaning of the sentence. Although, this is what the attention mechanism does, instead 
    of focusing words, it focuses on specific numerical values which may seem valuable. 
    """
    @staticmethod
    def attention(units, memory, memory_sequence_length, attention_type='luong'):  
        # There are two types of attention provided by tensorflow. The difference between these two attention types is how the score is calculated for the chatbot's reply
        return tfa.seq2seq.BahdanauAttention(units=units, memory=memory, memory_sequence_length=memory_sequence_length) if (attention_type == 'bahdanau') else tfa.seq2seq.LuongAttention(units=units, memory=memory, memory_sequence_length=memory_sequence_length)

    def init_network(self, batch_size, encoder_state, Dtype): # initialize the network
        return self.rnn_network_attention.get_initial_state(batch_size=batch_size, dtype=Dtype).clone(cell_state=encoder_state)

    def add_attention(self): # Adds attention to the decoder 
        return tfa.seq2seq.AttentionWrapper(self.network,self.attention_layer, attention_layer_size=self.units)

    def call(self, inputs, init_dimensions): # return the decoded numerical values into words
        x = self.input_layer(inputs)
        outputs, _, _ = self.decoder(x, initial_state=init_dimensions, sequence_length=self.batch_size*[max_length_output-1])
        return outputs

"""
Training mechanism for the chatbot. This is the training process it takes on each mini-batch of data provided. 
"""
@tf.function
def train(user_input, real_replies, hidden_layer):
    _loss = 0
    with tf.GradientTape() as gradients: # here we calculate all the gradients for the network. Similar to how backpropagation calculates gradients
        encoder_output, encoder_hidden, encoder_current = encoder(user_input, hidden_layer) # Encode the words into tensors/numbers, returns the call
        dec_input = real_replies[:, :-1]   # Ignore the tokens at the start and the end <start> and <genshin>
        real_replies = real_replies[:, 1:]       
        # Set the AttentionMechanism object with encoder_outputs
        decoder.attention_layer.setup_memory(encoder_output)
        # Create AttentionWrapperState as initial_state for decoder
        decoder_initial_state = decoder.init_network(CONST_BATCH_SIZE, [encoder_hidden, encoder_current], tf.float32) # initialise the decoder
        ####################### Pass the initial parameters through the network
        network_prediction = decoder(dec_input, decoder_initial_state)
        #######################
        logits = network_prediction.rnn_output # output of the network
        _loss = loss(real_replies, logits) # Taking the total error/loss the network produced and comparing it to the real_replies replies
        variables = encoder.trainable_variables + decoder.trainable_variables
        new_gradients = gradients.gradient(_loss, variables)
        optimizer.apply_gradients(zip(new_gradients, variables)) # update the gradients of the network
    return _loss



"""
The loss function or the error function, determines how correct the chatbot is and is used to calculate the gradients of the network. 
However, unlike classical network networks, optimizations have been made to make the computation of these losses faster, using a SparseCategoricalCrossentropy. The equation
for this method of calculated the loss can be found at https://stats.stackexchange.com/questions/326065/cross-entropy-vs-sparse-cross-entropy-when-to-use-one-over-the-other.

To summarize, SparseCategoricalCrossentropy is effectively the well known equation (expected - predicted) however, it has been optimized and is both efficient in time and space complexities.
"""
def loss(real_replies, network_output):
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = cross_entropy(y_true=real_replies, y_pred=network_output)
    loss = tf.reduce_mean(tf.cast(tf.logical_not(tf.math.equal(real_replies, 0)), dtype=loss.dtype) * loss)
    return loss

# This code is used for the training process, It'll just be cleaner placed here.

# Setting up the network
CONST_BUFFER_SIZE = 32000 # limits how much we read from the IO/Stream. We wouldn't want a buffer overflow...
CONST_BATCH_SIZE = 32 # The batch sizes can vary depending on the computation power of your computer
dataset_limit = 30000  # Limit for dataset sizes
####################
#################### Start formatting the data
####################
print("#############\n Reading The Training Data\n############# (~ 5 minutes)")
dataset_creator = Format_dataset() 
train_dataset, val_dataset, original_data_list, reply_data_list = dataset_creator.call(dataset_limit, CONST_BUFFER_SIZE, CONST_BATCH_SIZE)

original_message_batch, reply_batch = next(iter(train_dataset))

# Setting up the dimensions/shapes for the initial chatbot network. 
chatbot_memory_original_size = len(original_data_list.word_index)+1 # This is the chatbot's dictionary, for all the words its learnt and tokenized
chatbot_memory_reply_size = len(reply_data_list.word_index) + 1
max_length_input = original_message_batch.shape[1]
max_length_output = reply_batch.shape[1]
num_batches = dataset_limit//CONST_BATCH_SIZE # dividing the set into smaller epochs
####################
#################### Initialize the encoder
####################
print("\n#############\n Setting up Encoder\n#############")
encoder = Encoder(CONST_BATCH_SIZE, 256, chatbot_memory_original_size, 1024)
random_hidden = encoder.initialize_hidden_layers()  # test input, to create the encoder layer sizes. 
random_output, hidden_layer_example, current_sample = encoder(original_message_batch, random_hidden)

####################
#################### Initialize the decoder
####################
print("\n#############\n Setting up Decoder\n#############")
decoder = Decoder(CONST_BATCH_SIZE, 256, chatbot_memory_reply_size, 1024)
random_batch = tf.random.uniform((CONST_BATCH_SIZE, max_length_output))
print("\n#############\n Setting up the Attention Mechanism\n#############")
decoder.attention_layer.setup_memory(random_output) # Setting up attention layer
initial_state = decoder.init_network(CONST_BATCH_SIZE, [hidden_layer_example, current_sample], tf.float32)
decoder(random_batch, initial_state)

# This is an optimizer, it includes relevant information for NMT and deep learning chatbots
# Since it has this context opf what type of network we're training, it improves the speed of the program. 
print("\n#############\n Integrating the optimizer\n#############")
optimizer = tf.keras.optimizers.Adam() 

# Parameters used for training and saving the state of the network. 
print("\n#############\n Setting up for Training\n#############")
state_path = os.path.join(CONST_TRAINING_CHECKPOINT_DIRECTORY, "ckpt")
current_state = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
print("\n#############\n Chatbot Loaded!\n#############")