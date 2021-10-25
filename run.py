from model import *

Chatbot.restore_latest_state()
while True:
    user_input = input("> ")
    print(Chatbot.reply(u"{}".format(user_input), False))