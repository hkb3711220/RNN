import tensorflow as tf
from tensorflow.nn.rnn_cell import BasicLSTMCell
from load_preprocess import source_inputs, target_data, source_vocab_to_int,target_vocab_to_int, target_sequence_length, max_target_seq_length

encode_inputs = tf.placeholder(tf.int32, [None, None])
decode_data = tf.placeholder(tf.int32, [None, None])
decoder_sequence_length = tf.placeholder(tf.int32, [None])
batch_size = tf.placeholder(tf.int32,[])
num_units = 20
encoder_embed_size = decoder_embed_size = 20
encoder_vol_size = len(source_vocab_to_int)
decoder_vol_size = len(target_vocab_to_int)

#create embedding variable
def embedding_variable(vol_size, embed_size):
    init_embed = tf.random_uniform([vol_size, embed_size])
    return tf.Variable(init_embed)

def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    # get '<GO>' id
    go_id = target_vocab_to_int['<GO>']

    # The input data is [max_encoder_length, batch_size]
    # Change input data into [Batch size , max_encoder_length]

    target_data = tf.transpose(target_data, [1, 0])
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat([tf.fill([batch_size, 1], go_id), after_slice], 1)

    return after_concat

def encoder_layer(embed_inputs, num_units):
    # Run encoder RNN
    encoder_cell = BasicLSTMCell(num_units)
    # The input data is [max_encoder_length, batch_size]
    # For the format of input data into [Batch size , max_encoder_length]
    # SO, time_major = Ture
    encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                         embed_inputs,
                                         dtype=tf.float32,
                                         time_major=True)

    return encoder_output, encoder_state

def decoder_train_layer(decoder_cell,
                        encoder_state,
                        batch_size,
                        decoder_embed_input,
                        decoder_sequence_length,
                        output_layer):

    #TrainingHelper
    helper = tf.contrib.seq2seq.TrainingHelper(inputs = decoder_embed_input,
                                               sequence_length = decoder_sequence_length)

    # BasicDecoder
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                              helper,
                                              encoder_state,
                                              output_layer)

    # unrolling the decoder layer
    outputs, _ ,_= tf.contrib.seq2seq.dynamic_decode(decoder)

    return outputs


def decoder_inference_layer(decoder_cell,
                            encoder_state,
                            decoder_embed_input,
                            start_token,
                            end_token,
                            output_layer):

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embed_input,
                                                      tf.fill([batch_size], start_token),
                                                      end_token)

    # BasicDecoder
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                              helper,
                                              encoder_state,
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)

    return outputs

def decoder_layer(num_units,
                  embed_decoder,
                  decoder_vol_size,
                  encoder_state,
                  decoder_embed_input,
                  decoder_sequence_length,
                  attention_mechanism):

    #define decoder RNN cell with Attention Mechanism
    decoder_cell = BasicLSTMCell(num_units)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
    decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

    with tf.variable_scope('decode'):
        output_layer = tf.layers.Dense(decoder_vol_size)
        train_output = decoder_train_layer(decoder_cell,
                                           decoder_initial_state,
                                           batch_size,
                                           decoder_embed_input,
                                           decoder_sequence_length,
                                           output_layer=output_layer)
    with tf.variable_scope('decode'):
        infer_output = decoder_inference_layer(decoder_cell,
                                               decoder_initial_state,
                                               embed_decoder,
                                               start_token = target_vocab_to_int['<GO>'],
                                               end_token = target_vocab_to_int['<EOS>'],
                                               output_layer=output_layer)

    return train_output, infer_output

with tf.variable_scope('encoder'):
    #embeding scource inputt
    embed_encoder = embedding_variable(encoder_vol_size, encoder_embed_size)
    encoder_embed_input = tf.nn.embedding_lookup(embed_encoder, encode_inputs)

    #the hidden layers' states in encoder layer will be given to decode layer
    encoder_output, encoder_state = encoder_layer(encoder_embed_input, num_units=num_units)

#Add Attention Mechanism
with tf.variable_scope('Attention'):
    # attention_states: [batch_size, max_encoder_time, num_units]
    attention_states = tf.transpose(encoder_output, [1, 0, 2])
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, attention_states)

with tf.variable_scope('decoder'):
     #embeding target inputs
     #decoder divide into two parts trainning and inference
     #share the same architecture and its parameters
     #same embedding vector should be shared via training and inference phases
    target_inputs_decode = process_decoder_input(decode_data, target_vocab_to_int, batch_size)
    embed_decoder = embedding_variable(decoder_vol_size, decoder_embed_size)
    decoder_embed_input = tf.nn.embedding_lookup(embed_decoder, target_inputs_decode)
    train_output, infer_output = decoder_layer(num_units = num_units,
                                               embed_decoder = embed_decoder,
                                               decoder_vol_size = decoder_vol_size,
                                               encoder_state = encoder_state,
                                               decoder_embed_input = decoder_embed_input,
                                               decoder_sequence_length = decoder_sequence_length,
                                               attention_mechanism=attention_mechanism)

    logits = train_output.rnn_output
    prediction = infer_output.sample_id

with tf.variable_scope('Optimization') as scope:
    decode_label = tf.transpose(decode_data, [1,0])
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decode_label, logits=logits)
    loss = tf.reduce_mean(crossent)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    grads_and_vars = optimizer.compute_gradients(loss)
    clipped_grads_and_vars = [(tf.clip_by_value(grad,0,1), var) for grad, var in grads_and_vars]
    training_op = optimizer.apply_gradients(clipped_grads_and_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #result = sess.run(train_output, feed_dict={encode_inputs:source_inputs[:,:3], decode_data:target_data[:3], batch_size:3, decoder_sequence_length:target_sequence_length[:3]})
    output = sess.run([prediction], feed_dict={encode_inputs:source_inputs[:,:3],
                                          decode_data:target_data[:,:3],
                                          batch_size:3,
                                          decoder_sequence_length: target_sequence_length[:3]})

    print(output)
