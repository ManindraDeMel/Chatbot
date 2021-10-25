from logging import debug
from flask import Flask, redirect, sessions, url_for, render_template, request, session, send_file, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room, close_room, rooms, disconnect
from threading import Lock
# from model import *

def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': count})

async_mode = None
chatbot_list = []
app = Flask(__name__)
app.secret_key = "testkey123"
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/genUserId", methods=["POST", "GET"])
def generate_user_id():
    id = len(chatbot_list)
    # chatbot_list.append(Chatbot())
    return redirect(f"/chat/id={id}")

@app.route("/chat/<id>")
def chat(id):
    return render_template("chat.html", async_mode=socketio.async_mode)

@socketio.on("connect")
def connect():
    # global thread
    # with thread_lock:
    #     if thread is None:
    #         thread = socketio.start_background_task(background_thread)
    emit('confirmed', {'data': 'Connected'})

@socketio.on("message")
def handle_message(user_input):
    emit ("respond", {'data': "This is a test message"})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=9090, debug=True)#, host='kalichatbot.ddns.net', port=80, debug=True)



"""
Todo:

- randomly split responses by sentences and send as separate messages, so it doesn't always seem so sequential to the user

- Memorize the entire conversation, and raise it to the bot if a user repeats themselves. I might add like a boredem meter or something, which
   once filled, the bot leaves the conversation.

- Emojis?


chatbot.restore_latest_state()
while True:
    user_input = input("> ")
    print(chatbot.reply(u"{}".format(user_input)))
"""
