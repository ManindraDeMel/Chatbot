from pre_process import *

"""
Run this file to start a conversation with the chatbot. 
"""

"""
Todo:

- randomly split responses by sentences and send as separate messages, so it doesn't always seem so sequential to the user
- Memorize the entire conversation, and raise it to the bot if a user repeats themselves. I might add like a boredem meter or something, which
   once filled, the bot leaves the conversation.

- Emojis?

"""

current_state.restore(tf.train.latest_checkpoint(CONST_TRAINING_CHECKPOINT_DIRECTORY)).expect_partial() # Restoring the latest training parameters
print("\n############# Kali the Deep Learning Chatbot #############\n")
while True:
    user = input("> ")
    reply(u"{}".format(user)) # The score associated with the chatbot reply, is what the chatbot thinks on how good it's response is. The closer to 0, the better.

