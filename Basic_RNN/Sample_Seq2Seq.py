import tensorflow as tf
import numpy as np
import os
from tensorflow.nn.rnn_cell import GRUCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.seq2seq import LuongAttention, AttentionWrapper, GreedyEmbeddingHelper, TrainingHelper
from load_preprocess_revised import sr_seq_leg, sr_inputs, sr_dict
from load_preprocess_revised import tgt_seq_leg, tgt_inputs, tgt_label, tgt_dict, max_tgt_seq_leg

cwd = os.getcwd()

def as_array(data, idx):
    return np.asarray([data[i] for i in idx])

def next_batch(batch_size, data, labels, tgt_label, data_seq_leg, labels_seq_leg):

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = as_array(data, idx)
    labels_shuffle = as_array(labels, idx)
    tgt_labels_shuffle = as_array(tgt_label, idx)
    data_seq_leg_shuffle = as_array(data_seq_leg, idx)
    labels_seq_leg_shuffle = as_array(labels_seq_leg, idx)

    return data_shuffle, labels_shuffle, tgt_labels_shuffle, data_seq_leg_shuffle, labels_seq_leg_shuffle

class Seq2Seq(object):

    def __init__(self):
        self.num_units = 128
        self.num_layers = 3
        self.batch_size = 100
        self.n_epoch = 1000
        self.embed_size = 30

    def model(self, x, enc_vocab_size, enc_embed_size, enc_seq_leg,
              y, dec_vocab_size, dec_embed_size,
              dec_seq_leg, input_keep_prob, inference=False):

        #embeding
        enc_inputs = tf.contrib.layers.embed_sequence(x,
                                                      enc_vocab_size,
                                                      enc_embed_size)
        #enc_inputsは以下のように書けます
        #enc_embed = tf.random_uniform([vocab_size, embed_size])
        #enc_inputs = tf.nn.embedding_lookup(params=enc_embed, ids=x)

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
        dec_initial_state = dec_cells.zero_state(self.batch_size, tf.float32).clone(cell_state=enc_states)

        print('Start decoder')
        output_layer = tf.layers.Dense(dec_vocab_size)

        with tf.variable_scope('train'):
            train_helper = TrainingHelper(dec_inputs, sequence_length=dec_seq_leg)

            dec_train_outputs = self._dec_layer(dec_cells,
                                                dec_initial_state,
                                                train_helper,
                                                output_layer,
                                                scope='dec_train')


        with tf.variable_scope('prediction'):
            infer_helper = GreedyEmbeddingHelper(dec_embed,
                                                 tf.fill([self.batch_size], tgt_dict['<GO>']),
                                                 tgt_dict['<EOS>'])

            dec_infer_outputs = self._dec_layer(dec_cells,
                                                dec_initial_state,
                                                infer_helper,
                                                output_layer,
                                                scope='dec_infer',
                                                Infer=True)

        return dec_train_outputs, dec_infer_outputs

    def cost(self, outputs, label):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=outputs)
        cost = tf.reduce_mean(crossent)

        return cost

    def training(self, cost):
        max_gradient_norm = 1
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        params = tf.trainable_variables()
        gradients = tf.gradients(cost, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        training_op = optimizer.apply_gradients(zip(clipped_gradients, params))

        return training_op

    def train(self):

        enc_vocab_size = len(sr_dict)
        dec_vocab_size = len(tgt_dict)

        enc_data = tf.placeholder(tf.int32, [None, None], name='enc_data')
        dec_data = tf.placeholder(tf.int32, [None, None], name='dec_data')
        dec_label = tf.placeholder(tf.int32, [None, None], name='dec_label')
        enc_seq_leg = tf.placeholder(tf.int32, [None], name='enc_seq_leg')
        dec_seq_leg = tf.placeholder(tf.int32, [None], name='dec_seq_leg')

        input_keep_prob = tf.placeholder(tf.float32)

        logits, infer_output = self.model(enc_data, enc_vocab_size, self.embed_size, enc_seq_leg,
                                        dec_data, dec_vocab_size, self.embed_size,
                                        dec_seq_leg, input_keep_prob)

        predictions = tf.identity(infer_output.sample_id, name='predictions')
        cost = self.cost(logits.rnn_output, dec_label)
        training_op = self.training(cost)
        saver = tf.train.Saver()

        min_loss = None
        with tf.Session() as sess:
            print("start train")
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(self.n_epoch):
                data_batch, labels_batch, tgt_label_batch, data_seq_leg_batch, labels_seq_leg_batch = next_batch(self.batch_size,
                                                                                                                 sr_inputs,
                                                                                                                 tgt_inputs,
                                                                                                                 tgt_label,
                                                                                                                 sr_seq_leg,
                                                                                                                 tgt_seq_leg)

                loss_train, _ = sess.run([cost, training_op], feed_dict={enc_data:data_batch,
                                                                                    dec_data:labels_batch,
                                                                                    dec_label:tgt_label_batch,
                                                                                    enc_seq_leg:data_seq_leg_batch,
                                                                                    dec_seq_leg:labels_seq_leg_batch,
                                                                                    input_keep_prob:0.5})

                if min_loss is None:
                    min_loss = loss_train
                    saver.save(sess, cwd + "//my_model.ckpt")
                elif loss_train < min_loss:
                    min_loss = loss_train
                    saver.save(sess, cwd + "//my_model.ckpt")

                if epoch % 100 == 0:
                    print("step", epoch, "loss:", loss_train)
            print("end train")


    def predict(self):

        model = tf.train.import_meta_graph('./my_model.ckpt.meta')
        graph = tf.get_default_graph()
        enc_data = graph.get_tensor_by_name('enc_data:0')
        enc_seq_leg = graph.get_tensor_by_name('enc_seq_leg:0')
        input_keep_prob = graph.get_tensor_by_name('Placeholder:0')
        infer_output = graph.get_tensor_by_name('predictions:0')

        with tf.Session() as sess:
            #for op in tf.get_default_graph().get_operations():
                #print(op.name)

            print("start predict")

            model.restore(sess, './my_model.ckpt')
            data_batch, _, _, data_seq_leg_batch, _ = next_batch(self.batch_size,
                                                                 sr_inputs,
                                                                 tgt_inputs,
                                                                 tgt_label,
                                                                 sr_seq_leg,
                                                                 tgt_seq_leg)

            predictions = sess.run([infer_output], feed_dict={enc_data:data_batch,
                                                              enc_seq_leg:data_seq_leg_batch,
                                                              input_keep_prob:1.0})


            print(predictions)
            print("end predict")

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
