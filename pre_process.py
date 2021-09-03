from chatbot import *

"""
In this function, we process the user input, format it and pass it through the trained network.
"""
def get_reply(sentence, beam_width=10) -> tuple:
    ############# Process and format the input #############
    sentence = dataset_creator.format_sentence(sentence) # formatting the sentence exactly like how the training data was formatted
    try:
        inputs = [original_data_list.word_index[word] for word in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=max_length_input,padding='post')
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]
        ############ Pass the data into the encoder and through the LTSM's
        encoder_start_state = [tf.zeros((inference_batch_size, 1024)), tf.zeros((inference_batch_size, 1024))]
        encoder_output, encoder_hidden, encoder_current = encoder(inputs, encoder_start_state)
        ## Tokens for the sentences
        start_tokens = tf.fill([inference_batch_size],reply_data_list.word_index['<start>'])
        end_token = reply_data_list.word_index['<end>']
        ##
        encoder_output = tfa.seq2seq.tile_batch(encoder_output, multiplier=beam_width)
        ############ Pass the encoded details to the decoder
        decoder.attention_layer.setup_memory(encoder_output)
        hidden_state = tfa.seq2seq.tile_batch([encoder_hidden, encoder_current], multiplier=beam_width)
        decoder_initial_state = decoder.rnn_network_attention.get_initial_state(batch_size=beam_width*inference_batch_size, dtype=tf.float32)
        decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)
        decoder_instance = tfa.seq2seq.BeamSearchDecoder(decoder.rnn_network_attention, beam_width=beam_width, output_layer=decoder.output_layer)
        decoder_input_layer_matrix = decoder.input_layer.variables[0]
        ############ The final network output
        outputs = decoder_instance(decoder_input_layer_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)[0]
        ############ return the response and the score associated with it
        return tf.transpose(outputs.predicted_ids, perm=(0, 2, 1)).numpy(), tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0, 2, 1)).numpy()
    except KeyError: # I probably want to return what word caused the key error, but its fine for now. 
        return None



def reply(sentence, dont_show_all_responses = True) -> None: # This functions wraps all the processing for the chatbot's reply and passes the user input
    response = get_reply(sentence)
    if response:
        result, beam_scores = response
        if dont_show_all_responses: 
            for beam, score in zip(result, beam_scores):
                output = reply_data_list.sequences_to_texts(beam)
                output = [a[:a.index('<end>')] for a in output] # Ignore the tokens
                beam_score = [a.sum() for a in score] # Sum all the individual scores
                max_score = (0, beam_score[0])
                for i in range(len(output)):
                    if beam_score[i] > max_score[1]:
                        max_score = (i, beam_score[i])
            print(f'Kali: {output[max_score[0]]}, score: {beam_score[max_score[0]]}\n')
        else:
            for beam, score in zip(result, beam_scores):
                output = reply_data_list.sequences_to_texts(beam)
                output = [a[:a.index('<end>')] for a in output]
                beam_score = [a.sum() for a in score]
                for i in range(len(output)):
                    print(f'({i+1}) Kali: {output[i]}, score: {beam_score[i]}\n')
    else:
        print("The chatbot does not know a word in the given sentence and thus cannot respond. :(\n") # I tokenize by words rather than characters to give my chatbot more context.
        # However, a downfall to this is that the chatbot cannot respond to words it's never encountered before, its kind of similar to how humans can't respond or 
        # comprehend the words they've never learnt before