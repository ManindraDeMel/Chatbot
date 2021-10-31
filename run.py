from model import *
"""
A simple file to run the chatbot within the command line
"""
Chatbot.restore_latest_state()
while True:
    user_input = input("> ")
    print(Chatbot.reply(u"{}".format(user_input))) 