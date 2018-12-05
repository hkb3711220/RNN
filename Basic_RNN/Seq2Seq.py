import tensorflow as tf
import numpy as np
import os
from tensorflow.nn.rnn_cell import GRUCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.seq2seq import LuongAttention, AttentionWrapper, GreedyEmbeddingHelper, TrainingHelper, sequence_loss

cwd = os.getcwd()

class Seq2Seq(object):

    def __init__(self):
        self.num_units = 128
        self.num_layers = 3
        self.batch_size = 100
        self.max_feature = 100
        self.max_dec_leg = 10
        self.max_enc_leg = 10
        self.enc_vocab_size = 10000
        self.dec_vocab_size = 10000

        self.enc_data = tf.placeholder(tf.int32, [self.batch_size, self.max_enc_leg], name='enc_data')
        self.dec_data = tf.placeholder(tf.int32, [self.batch_size, self.max_dec_leg + 1], name='dec_data') #START WITH <GO>
        self.dec_label = tf.placeholder(tf.int32, [self.batch_size, self.max_dec_leg + 1], name='dec_label') #END OF <EOS>
        self.enc_seq_leg = tf.placeholder(tf.int32, [self.batch_size], name='enc_seq_leg')
        self.dec_seq_leg = tf.placeholder(tf.int32, [self.batch_size], name='dec_seq_leg')
        self.input_keep_prob = tf.placeholder(tf.float32)

    def main(self, x, enc_vocab_size, enc_seq_leg,
                   y, dec_vocab_size, dec_seq_leg, input_keep_prob,
                   enc_embed_size=300,dec_embed_size=300):

        #embeding
        enc_inputs = tf.contrib.layers.embed_sequence(x, enc_vocab_size, enc_embed_size)

        print('Start encoder')
        enc_outputs, enc_states = self._enc_layer(enc_inputs,
                                                  num_units=self.num_units,
                                                  num_layers=self.num_layers,
                                                  input_keep_prob=input_keep_prob,
                                                  seq_leg=enc_seq_leg,
                                                  scope='enc')
        #Attention_Layers
        attention_states = tf.transpose(enc_outputs, [0, 1, 2])
        attention_mechanism = LuongAttention(self.num_units, memory = attention_states)

        #dec
        dec_embed = tf.Variable(tf.random_uniform([dec_vocab_size, dec_embed_size]))
        dec_inputs = tf.nn.embedding_lookup(dec_embed, y)

        dec_cells = self._dec_cells(attention_mechanism,
                                    num_units=self.num_units,
                                    num_layers=self.num_layers,
                                    input_keep_prob=input_keep_prob,
                                    scope='dec_cell')

        dec_cells = AttentionWrapper(dec_cells, attention_mechanism)
        dec_init_state = dec_cells.zero_state(self.batch_size, tf.float32).clone(cell_state=enc_states)

        print('Start decoder')
        output_layer = tf.layers.Dense(dec_vocab_size)

        with tf.variable_scope('train'):
            train_helper = TrainingHelper(dec_inputs, sequence_length=dec_seq_leg)

            dec_train_outputs = self._dec_layer(dec_cells,
                                                dec_init_state,
                                                train_helper,
                                                output_layer,
                                                scope='dec_train')


        with tf.variable_scope('inference'):
            infer_helper = GreedyEmbeddingHelper(dec_embed,
                                                 tf.fill([self.batch_size], tgt_dict['<GO>']),
                                                 tgt_dict['<EOS>'])

            dec_infer_outputs = self._dec_layer(dec_cells,
                                                dec_init_state,
                                                infer_helper,
                                                output_layer,
                                                scope='dec_infer',
                                                Infer=True)

        return dec_train_outputs, dec_infer_outputs

    def loss(self, logitss, labels, max_dec_leg):
        """
        same code as below

        mask = tf.sequence_mask(labels, max_dec_leg, dtype=tf.float32)
        target_weight = tf.cast(mask, dtype=tf.float32)#[batch_size, max_dec_leg]
        crossentroy = tf.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = (tf.reduce_sum(crossentroy*target_weight)/batch_size)

        """

        mask = tf.sequence_mask(labels, max_dec_leg, dtype=tf.float32)
        loss = sequence_loss(logits, labels, mask)

        return loss

    def train_op(self, cost):

        max_gradient_norm = 1
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        params = tf.trainable_variables()
        gradients = tf.gradients(cost, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

        return train_op

    def accuracy(self, infer_output, labels):

        """
        infer_output.sample_id  [batch_size, max_dec_leg + 1]
        labels  [batch_size, max_dec_leg + 1]
        """

        predictions = tf.identity(infer_output.sample_id, name='predictions')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

        return accuracy

    def _enc_layer(self, inputs, num_units, num_layers,
                      input_keep_prob, seq_leg, scope):

        with tf.variable_scope(scope):
            cells = [GRUCell(num_units) for _ in range(num_layers)]
            cells = [DropoutWrapper(cell, input_keep_prob) for cell in cells]
            cells = MultiRNNCell(cells)

            outputs, states = tf.nn.dynamic_rnn(cells,
                                                inputs,
                                                sequence_length=seq_leg,
                                                time_major=False,
                                                dtype=tf.float32)

            return outputs, states


    def _dec_cells(self, attention_mechanism, num_units,
                        num_layers, input_keep_prob, scope):

            with tf.variable_scope(scope):
                cells = [GRUCell(num_units) for _ in range(num_layers)]
                cells = [DropoutWrapper(cell, input_keep_prob) for cell in cells]
                cells = MultiRNNCell(cells)

            return cells

    def _dec_layer(self, dec_cell, enc_state,
                    helper, output_layer, scope, Infer=False):

        with tf.variable_scope(scope):
            dec = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                  helper,
                                                  enc_state,
                                                  output_layer)

            if Infer:
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(dec,
                                                                  impute_finished=True,
                                                                  maximum_iterations=max_tgt_seq_leg)
            else:
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(dec)

        return outputs

if __name__ == '__main__':

    model = Seq2Seq()
    model.train()
    model.predict()
