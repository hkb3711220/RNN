import sys
sys.path.append('./')

import tensorflow as tf
from utils import gen_data
from Seq2Seq import Seq2Seq
import os
os.chdir(os.path.dirname(__file__))
cwd = os.getcwd()

class trainer():

    def __init__(self):

        self.n_epoch = 1
        self.batch_size = 100
        self.fr_enc_vocab, self.fr_enc_seq_leg, self.max_fr_enc_leg, self.fr_enc_data = gen_data(language_name='fr')
        self.en_dec_vocab, self.en_dec_seq_leg, self.max_en_dec_leg, self.en_dec_data, self.en_dec_label = gen_data(language_name='en', decode=True)

        self.enc_data = tf.placeholder(tf.int32, [self.batch_size, self.max_fr_enc_leg], name='enc_data')
        self.dec_data = tf.placeholder(tf.int32, [self.batch_size, self.max_en_dec_leg+1], name='dec_data') #START WITH <GO>
        self.dec_label = tf.placeholder(tf.int32, [self.batch_size, self.max_en_dec_leg+1], name='dec_label') #END OF <EOS>
        self.enc_seq_leg = tf.placeholder(tf.int32, [self.batch_size], name='enc_seq_leg')
        self.dec_seq_leg = tf.placeholder(tf.int32, [self.batch_size], name='dec_seq_leg')

        self.input_keep_prob = tf.placeholder(tf.float32)
        self.Seq2Seq = Seq2Seq(self.batch_size)
        self.train_output, self.infer_output = self.Seq2Seq.main(self.enc_data, len(self.fr_enc_vocab), self.enc_seq_leg,
                                                                 self.dec_data, len(self.en_dec_vocab), self.dec_seq_leg,
                                                                 self.input_keep_prob, self.max_en_dec_leg)

        print(self.train_output.rnn_output)

        self.loss = self.Seq2Seq.loss(self.train_output.rnn_output, self.dec_label, self.dec_seq_leg, self.max_en_dec_leg)
        tf.summary.scalar('loss', self.loss)
        self.train_op = self.Seq2Seq.train_op(self.loss)
        self.accuracy = self.Seq2Seq.accuracy(self.infer_output, self.dec_label)
        tf.summary.scalar('loss', self.accuracy)

    def train(self):
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(cwd  + '/log', sess.graph)

            X_train, X_test = self.fr_enc_data[:110000, :], self.fr_enc_data[110000:, :]
            y_input_train, y_input_test = self.en_dec_data[:110000, :], self.en_dec_data[110000:, :]
            y_label_train, y_label_test = self.en_dec_label[:110000, :], self.en_dec_label[110000:, :]
            enc_leg_train, enc_leg_test = self.fr_enc_seq_leg[:110000], self.fr_enc_seq_leg[11000:]
            dec_leg_train, dec_leg_test = self.en_dec_seq_leg[:110000], self.en_dec_seq_leg[11000:]

            for epoch in range(self.n_epoch):
                for i in range(X_train.shape[0] // self.batch_size):
                    X_batch = X_train[self.batch_size*i:self.batch_size*(i+1), :]
                    y_input_batch = y_input_train[self.batch_size*i : self.batch_size*(i+1), :]
                    y_label_batch = y_input_train[self.batch_size*i : self.batch_size*(i+1), :]
                    enc_leg_batch = enc_leg_train[self.batch_size*i : self.batch_size*(i+1)]
                    dec_leg_batch = dec_leg_train[self.batch_size*i : self.batch_size*(i+1)]

                    train_loss, _, train_accuracy =sess.run([self.loss, self.train_op, self.accuracy],
                                                             feed_dict={self.enc_data:X_batch, self.dec_data:y_input_batch,
                                                                        self.dec_label:y_label_batch, self.enc_seq_leg:enc_leg_batch,
                                                                        self.dec_seq_leg:dec_leg_batch, self.input_keep_prob:0.8})

                    print('Epoch:', epoch, ' No:', i, 'train loss:', train_loss, 'train accuracy:', train_accuracy)

                if epoch % 10 == 0:
                    test_loss, test_accuracy = sess.run([self.loss, self.accuracy], feed_dict={self.enc_data:X_test, self.dec_data:y_input_test,
                                                                         self.dec_label:y_label_test, self.enc_seq_leg:enc_leg_test,
                                                                         self.dec_seq_leg:dec_leg_test, self.input_keep_prob:1.0})

                    print(test_accuracy)

trainer().train()
