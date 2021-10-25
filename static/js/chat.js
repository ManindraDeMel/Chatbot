const msgerChat = $(".msger-chat")[0];

// Icons made by Freepik from www.flaticon.com
const BOT_IMG = "/static/images/kali.png";
const PERSON_IMG = "/static/images/default_user.png";
const BOT_NAME = "Kali";
const PERSON_NAME = "You";

const socket = io();

socket.on('confirmed', (msg) => {
  console.log(msg.data)
});

socket.on('respond', (msg) => {
  botResponse(msg.data);
});

function formatDate(date) {
  const h = "0" + date.getHours();
  const m = "0" + date.getMinutes();

  return `${h.slice(-2)}:${m.slice(-2)}`;
}

$(".msger-inputarea").on("submit", (event) => {
  event.preventDefault();
  const msgText = document.getElementById("inputBox").value;
  if (!msgText) return;
  appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText);
  document.getElementById("inputBox").text = "";  
  socket.emit('message', {data: msgText})

});

function appendMessage(name, img, side, text) {
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
  const delay = response.split(" ").length * 100;
  setTimeout(() => {
    appendMessage(BOT_NAME, BOT_IMG, "left", response);
  }, delay);
}
