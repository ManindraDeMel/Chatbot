from chatbot import *
import time

"""
This file trains the actual network. It saves the state every {CONST_STATE_SAVE_POINT} epochs.
Currently the network is being trained in a mini-batch gradient descent manner. 
"""

CONST_EPOCHS = 100 # Amount of epochs
CONST_BATCH_OUTPUT = 100 # Every 100 batches, it will show the average loss
CONST_STATE_SAVE_POINT = 5 # save the network every 5 epochs

print("\n################# Training begun #################\n")
for epoch in range(CONST_EPOCHS):
    start_time = time.time()
    hidden_layer = encoder.initialize_hidden_layers() # Initialize the encoder 
    total_loss = 0 # initialize the initial error/loss for the network
    for (batch, (original_comment, reply)) in enumerate(train_dataset.take(num_batches)):
        batch_loss = train(original_comment, reply, hidden_layer) # Train on the batch
        total_loss += batch_loss # Update the total loss with each batchs' loss 
        if batch % CONST_BATCH_OUTPUT == 0:
            print(f'Epoch: {epoch + 1} Batch: {batch} Loss: {batch_loss.numpy():.4f}')
        # saving the state of the model every 5 epochs
    if (epoch + 1) % CONST_STATE_SAVE_POINT == 0:
        current_state.save(file_prefix=state_path) 

    print('\n\n##################\nEpoch {} Loss {:.4f}'.format(epoch + 1, total_loss / num_batches))
    time_taken = (time.time() - start_time) / 60
    print(f'Time taken for 1 epoch {time_taken} mins\n##################')
