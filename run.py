import pre_process as chatbot
from tkinter import *
"""
The application is just temporary, and is just meant to demonstrate the prototype in a clearer manner. 
"""

"""
Run this file to start a conversation with the chatbot. 
"""
class Window:
   def __init__(self, init_width, init_height, bg_colours : tuple, text_colour : str, font):
      self.window = Tk()
      self.width = init_width
      self.font = font
      self.height = init_height
      self.bg_colour, self.bg_colour_2 = bg_colours
      self.text_colour = text_colour
      # Window config
      self.window.title("Kali: A Deep learning chatbot")
      self.window.resizable(width=True, height=True)
      self.window.configure(width=self.width, height=self.height, bg=self.bg_colour)
      # Text config
      self.text = Text(self.window, width=20, height=2,bg=self.bg_colour, fg=self.text_colour, font=self.font, padx=5, pady=5)
      self.text.place(relheight=0.745, relwidth=1, rely=0.08)
      self.text.configure(cursor="arrow", state=DISABLED)
      # Title config (Heading)
      head_label = Label(self.window, bg=self.bg_colour, fg=self.text_colour, text="Kali: A Deep learning chatbot", font=self.font, pady=10)
      head_label.place(relwidth=1)
      bottom_label = Label(self.window, bg=self.bg_colour_2, height=80)
      bottom_label.place(relwidth=1, rely=0.82)
      # The user input/message box config
      self.message_box = Entry(bottom_label, bg=self.bg_colour,fg=self.text_colour, font=self.font)
      self.message_box.place(relwidth=0.7, relheight=0.03, rely=0.04, relx=0.04)
      self.message_box.focus()
      self.message_box.bind("<Return>", self.enter)
      # Submit button config
      send_button = Button(bottom_label, text="Enter", font=self.font, width=20, bg=self.bg_colour,command=lambda: self.enter(None))
      send_button.place(relx=0.8, rely=0.04, relheight=0.03, relwidth=0.14)
      # Scrollbar
      scrollbar = Scrollbar(self.text)
      scrollbar.place(relheight=1, relx=0.95)
      scrollbar.configure(command=self.text.yview)

   def enter(self, event): # When enter is pressed, add the user message and reply to the application
      message = self.message_box.get()
      self.add_message(message)

   def add_message(self, message): # Adding message visually
      if not message:
         return None
      self.message_box.delete(0, END)
      user_input = f"You: {message}\n\n"
      self.text.configure(state=NORMAL)
      self.text.insert(END, user_input)
      self.text.configure(state=DISABLED)

      chatbot_reply = u"{}\n\n".format(chatbot.reply(message))
      self.text.configure(state=NORMAL)
      self.text.insert(END, chatbot_reply)
      self.text.configure(state=DISABLED)

      self.text.see(END)

chatbot.current_state.restore(chatbot.tf.train.latest_checkpoint(chatbot.CONST_TRAINING_CHECKPOINT_DIRECTORY)).expect_partial()  # Restoring the latest training parameters
window = Window(650, 750, ("#FFFFFF", "#5377e6"), "#000000", ("Century Gothic", 14))
print("Application running...")
window.window.mainloop()


"""
Todo:

- randomly split responses by sentences and send as separate messages, so it doesn't always seem so sequential to the user

- Memorize the entire conversation, and raise it to the bot if a user repeats themselves. I might add like a boredem meter or something, which
   once filled, the bot leaves the conversation.

- Emojis?

"""
