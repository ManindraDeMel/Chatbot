import pre_process as chatbot

chatbot.restore_latest_state()
while True:
    user_input = input("> ")
    print(chatbot.reply(u"{}".format(user_input)))