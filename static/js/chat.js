const msgerChat = $(".msger-chat")[0];
// This is the script used to communicate with the backend
// Icons made by Freepik from www.flaticon.com, 
const BOT_IMG = "/static/images/kali.png";
const PERSON_IMG = "/static/images/default_user.png";
const BOT_NAME = "Kali";
const PERSON_NAME = "You";

const socket = io();

socket.on('confirmed', (msg) => { // Initializing the connection with the backend. 
  console.log(msg.data)
});

socket.on('respond', (msg) => { // handling responses from the chatbot 
  botResponse(msg.data);
});

function formatDate(date) {// Getting the date
  const h = "0" + date.getHours();
  const m = "0" + date.getMinutes();

  return `${h.slice(-2)}:${m.slice(-2)}`; 
}

$(".msger-inputarea").on("submit", (event) => { // When the user submits their message we send that to the backend with the function below
  event.preventDefault();
  const msgText = document.getElementById("inputBox").value; // get the message value
  if (!msgText) return; // check the message isn't empty
  appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText); // append the user's message on the right
  document.getElementById("inputBox").value = "";  // empty the input box
  socket.emit('message', {data: msgText}) // send the message to the backend

});

function appendMessage(name, img, side, text) { // this is a general function which appends HTML for each message to the chat.html file. 
  //   Simple solution for small apps
  const msgHTML = `
    <div class="msg ${side}-msg">
      <div class="msg-img" style="background-image: url(${img})"></div>

      <div class="msg-bubble">
        <div class="msg-info">
          <div class="msg-info-name">${name}</div>
          <div class="msg-info-time">${formatDate(new Date())}</div>
        </div>

        <div class="msg-text">${text}</div>
      </div>
    </div>
  `;

  msgerChat.insertAdjacentHTML("beforeend", msgHTML);
  msgerChat.scrollTop += 500;
}

function botResponse(response) {  
  // Get response from the backend
  const delay = response.split(" ").length * 100; // to make the bot a bit more realistic, we add a slight delay based on the length of the response.
  setTimeout(() => {
    appendMessage(BOT_NAME, BOT_IMG, "left", response); // we then append the message to the HTML 
  }, delay);
}
