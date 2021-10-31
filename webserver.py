from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from model import *
"""
This is the webserver file, this is effectively the backend which handles all the users on the front end. It acts as a medium to communicate between the 
user and the chatbot. Although this aspect of my project is not the focus, it's still a vital aspect for functionality and ease of use, a website also looks
far better than any desktop application. In this specific implementation, I use websockets, specifically the flask-socketio module to communicate between the user and the chatbot. I 
specifically decided on using websocket's for its speed, which somewhat accounts for the slow chatbot response time and also is something I've never implementend about, and hence is
a new technique of communcation I've learnt. 
"""
# Initalizing the server
async_mode = None
app = Flask(__name__)
app.secret_key = "sD+@@!89-+--_($&***#-"
socketio = SocketIO(app, async_mode=async_mode)

@app.route("/") # When the user first lands on the website they're met with a landing page
def home():
    return render_template("index.html")

@app.route("/chat") # When the user clicks on the chat button we want to redirect to that page and also begin accepting websocket requests
def chat():
    return render_template("chat.html", async_mode=socketio.async_mode) # Accept websocket requests

@socketio.on("connect") # Sending a confirmation for the initialization of a new handshake between the new user and the client. 
def connect():
    emit('confirmed', {'data': 'Handshake Successful!'})

@socketio.on("message") # This is the chatbot's response too the user's message
def handle_message(user_input):
    response = Chatbot.reply(u'{}'.format(user_input))
    emit ("respond", {'data': response})

if __name__ == "__main__":
    print("Once chatting with the chatbot, send 3 ~ 5 redundant messages just to establish the web socket connection to the chatbot (takes like 3 minutes to establish)")
    Chatbot.restore_latest_state()
    socketio.run(app, host="0.0.0.0", port=5000)
