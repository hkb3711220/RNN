import tensorflow as tf
from tensorflow.python.util import nest

rnn = tf.nn.rnn_cell

class Prt_Net(object):

    def __init__(self, batch_size=128, lr=1.0, num_unit=256):
        """
        used a single layer LSTM with either 256 or 512 hidden units,
        trained with stochastic gradient descent with a learning rate of 1.0,
        batch size of 128,
        random uniform weight initialization from -0.08 to 0.08, and L2 gradient clipping of 2.0.

        the inputs are planar point sets P = {P1, . . . , Pn} with n elements each,
        where Pj = (xj , yj ) are the cartesian coordinates of the points over which we find the convex hull

        The outputs CP = {C1, . . . , Cm(P)} are sequences representing the solution associated to the point set P. I

        """
        self.num_unit = num_unit
        self.lr = lr
        self.batch_size = batch_size
        self.max_encode_leg = 5
        self.max_target_leg = 5
        self.initializer = tf.random_uniform_initializer(-0.08, 0.08, dtype=tf.float32)

        self.encode_seq = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_encode_leg, 2])
        self.encode_seq_leg = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.target_seq = tf.placeholder(tf.int32, shape=[self.batch_size,self.max_target_leg])
        self.target_seq_leg = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.input_dim = self.encode_seq.get_shape().as_list()[2]

    def _encoder(self, encode_seq, encode_seq_leg):

        """
        a single layer LSTM for encode.

        conv1d:
        Given an input tensor of shape [batch, in_width, in_channels] if data_format is "NWC",
        and a filter / kernel tensor of shape [filter_width, in_channels, out_channels],
        if data_format does not start with "NC", a tensor of shape [batch, in_width, in_channels] is reshaped to [batch, 1, in_width, in_channels],
        and the filter is reshaped to [1, filter_width, in_channels, out_channels].
        The result is then reshaped back to [batch, out_width, out_channels]


        """
        with tf.variable_scope('encoder'):

            input_embed = tf.get_variable(name='input_embed',
                                          shape=[1, self.input_dim, self.num_unit],
                                          initializer=self.initializer)

            encode_input = tf.nn.conv1d(encode_seq, input_embed, 1, padding='VALID')

            cell = rnn.LSTMCell(self.num_unit)
            init_state = cell.zero_state(self.batch_size, tf.float32)
            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell, encode_input,
                                                                encode_seq_leg, init_state)


        return encoder_outputs, encoder_states

    def _decoder_train(self, cell, target_embed_inputs, target_init_states, encoder_outputs):
        """
        input one of elements in target_embed_inputs to decode layer
        the shape of target_embed_inputs is (128, 6, 256)
        the shape of inputs to decocde layers every time  is (128, 1, 256)

        1.tf.squeeze:
        Removes dimensions of size 1 from the shape of a tensor. (deprecated arguments)
        """
        with tf.variable_scope('decoder_train'):

            decode_states = target_init_states

            predict_distribution = []
            predict_index = []

            for i in range(self.max_target_leg + 1):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_input = tf.squeeze(target_embed_inputs[:, i, :]) #(batch_size, num_unit)
                decoder_output, decoder_states = cell(cell_input, decode_states) #(batch_size, num_unit)
                p_distribution = self._attention(decoder_output, encoder_outputs, scope='choose_index_{}'.format(i)) #(batch_size, max_target_leg) one of elements -> embed_input
                p_index = tf.argmax(p_distribution, axis=1) #(batch_size,)

                predict_distribution.append(p_distribution)
                predict_index.append(p_index)

            return tf.convert_to_tensor(predict_distribution, dtype=tf.float32), tf.convert_to_tensor(predict_index, dtype=tf.float32)

    def _decoder_inference(self, cell, first_decode_input, target_init_states, encoder_outputs):

        with tf.variable_scope('decoder_infer'):
            decode_states = target_init_states
            predict_input = first_decode_input

            predict_distribution = []
            predict_index = []

            for i in range(self.max_target_leg + 1):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_input = tf.squeeze(predict_input)
                decoder_output, decode_states = cell(cell_input, decode_states)
                p_distribution = self._attention(decoder_output, encoder_outputs, scope='choose_index_{}'.format(i))
                p_index = tf.argmax(p_distribution, axis=1, output_type=tf.int32)

                index_paris = self._index_matrix_to_paris(p_index)
                predict_inputs = tf.stop_gradient(tf.gather_nd(encoder_outputs, index_paris))

                predict_distribution.append(p_distribution)
                predict_index.append(p_index)

            return tf.convert_to_tensor(predict_distribution, dtype=tf.float32), tf.convert_to_tensor(predict_index, dtype=tf.float32)

    def _attention(self, decode_output, encoder_outputs, scope):
        """
        attention mechanism in Pointer Network is blow.
        uij = vTtanh(W1ej + W2di) j ∈ (1, . . . , n)
        p(Ci|C1, . . . , Ci−1,P) = softmax(ui)

        where softmax normalizes the vector ui (of length n) to be an output distribution over the dictionary
        of inputs, and v, W1, and W2 are learnable parameters of the output model.
        """

        W1 = tf.get_variable(name='W1', shape=(self.num_unit, self.num_unit), initializer=self.initializer)
        W2 = tf.get_variable(name='W2', shape=(self.num_unit, self.num_unit), initializer=self.initializer)
        vT = tf.get_variable(name='vT', shape=(self.num_unit, 1), initializer=self.initializer)
        W2di = tf.matmul(decode_output, W2)

        ui = []
        for j in range(self.max_encode_leg + 1):
            W1ej = tf.matmul(tf.squeeze(encoder_outputs[:, j, :]), W1)
            uij = tf.matmul(tf.nn.tanh(W1ej + W2di), vT) #(batch_size, 1)
            ui.append(tf.squeeze(uij))

        ui = tf.transpose(ui, [1, 0]) #(batch_size, max_encode_leg+1 ) → [batch_size, max_encode_leg+1]

        return tf.nn.softmax(ui)  #[batch_size, max_encode_leg+1]

    def _index_matrix_to_paris(self, target_seq):
        """
        tf.tile:
        This operation creates a new tensor by replicating input multiples times.

        the sample of Output of this function is
        [[3,1,2], [2,3,1]] shape=[2, 3] -> [[[0, 3], [1, 1], [2, 2]], shape=[2,3,2]
                                           [[0, 2], [1, 3], [2, 1]]]
        """

        shape = tf.shape(target_seq)
        first_indices = tf.range(shape[0])
        if len(target_seq.get_shape()) == 2:
            first_indices = tf.tile(tf.expand_dims(first_indices, 1), #expand to [128, 1]
                                    [1, shape[1]]) #[128, 5]

        return tf.stack([first_indices, target_seq], axis=len(target_seq.get_shape())) #[128, 5, 2]

    def _loss(self, logits):
        """
        1.tf.sequence_mask:
        Returns a mask tensor representing the first N positions of each cell.
        2.tf.contrib.seq2seq.sequence_loss:
        Weighted cross-entropy loss for a sequence of logits.
        　-logits: A Tensor of shape [batch_size, sequence_length, num_decoder_symbols] and dtype float.
                  The logits correspond to the prediction across all classes at each timestep.
          -targets: A Tensor of shape [batch_size, sequence_length] and dtype int. The target represents the true class at each timestep.
          -weights: A Tensor of shape [batch_size, sequence_length] and dtype float. weights constitutes the weighting of each prediction in the sequence.
                    When using weights as masking, set all valid timesteps to 1 and all padded timesteps to 0,
                    e.g. a mask returned by tf.sequence_mask.

        """
        logits = tf.transpose(predict_distributuion, [1, 0, 2])
        mask = tf.sequence_mask(target_seq_leg, max_target_leg, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(logits,
                                                self.target_seq,
                                                mask)

        return loss

    def _trian_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.lr)
        return optimizer.minimize(loss)

    def build(self, is_training=True):
        """
        1.gather_nd:
        https://www.tensorflow.org/api_docs/python/tf/gather_nd

        """

        tf.logging.info("Create a model..")
        encoder_outputs, encoder_states = self._encoder(self.encode_seq, self.encode_seq_leg)

        first_decode_input = tf.expand_dims(tf.get_variable(name='first_decode_input',
                                                            shape=(self.batch_size, self.num_unit),
                                                            initializer=self.initializer), axis=1) #batchsize, 1, num_unit
        encoder_outputs = tf.concat([first_decode_input, encoder_outputs], axis=1) #[128, 6，256]
        target_init_state = encoder_states
        decoder_cell = rnn.LSTMCell(self.num_unit)

        if is_training:
            target_index_paris = self._index_matrix_to_paris(self.target_seq) #[128, 5, 2]
            target_embed_inputs = tf.stop_gradient(tf.gather_nd(encoder_outputs, target_index_paris))

            #Add start symbol to target_embed_inputs
            target_embed_inputs = tf.concat([first_decode_input, target_embed_inputs], axis=1)

            #Add end symbol(zeros) to target seq, shape become [128, 6]
            zeros_index = tf.tile(tf.zeros(shape=[1, 1], dtype=tf.int32),
                                          [self.batch_size, 1],
                                          name='zero_index')
            target_seq = tf.concat([self.target_seq, zeros_index], 1)

            predict_distribution, predict_index = self._decoder_train(decoder_cell,
                                                                     target_embed_inputs,
                                                                     target_init_state,
                                                                     encoder_outputs)


        elif not is_training:
            predict_distribution, predict_index = self._decoder_inference(decoder_cell,
                                                                         first_decode_input,
                                                                         target_init_state,
                                                                         encoder_outputs)


        return predict_distribution, predict_index


Prt_Net().build()
