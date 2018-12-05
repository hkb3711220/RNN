import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.chdir(os.path.dirname(__file__))

def add_start_symbol(word_index, offset_data):
    """
    Delete <EOS>, Add <GO> Symbol to each sequence
    """

    shape = offset_data.shape
    go_index = np.array([0] * shape[0])
    go_index = np.expand_dims(go_index.T, 1)


    offset_data = offset_data[:, :-1]
    offset_data_with_start = np.concatenate([go_index, offset_data], axis=1)

    return offset_data_with_start

def gen_data(language_name, decode=False):

    """
    args:
    target: the name of vocab
    decode: if language is decode data

    output:
    word_index: wrod vocabulary
    seq_leg: each sequence length
    max_len: max length of sequence
    offset_data: the sequence pad
    offset_data_with_start: the sequence with start symbol
    (the start symbol is 0)
    """

    path = os.path.join('data', 'small_vocab_{}'.format(language_name))

    text = []
    with open(path, 'r') as f:
        for line in f.readlines():
            text.append(line)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    word_index = tokenizer.index_word #the dictionary for word {1: word}
    data = tokenizer.texts_to_sequences(text)

    seq_leg = [len(text) for text in data] #sequence length
    max_len = max(seq_leg) #the max length of sequence

    if decode:
        offset_data = pad_sequences(data, maxlen=max_len+1)
        offset_data = np.fliplr(offset_data)
        offset_data_with_start = add_start_symbol(word_index, offset_data)

        return word_index, seq_leg, max_len, offset_data_with_start, offset_data

    else:
        offset_data = pad_sequences(data, maxlen=max_len)
        return word_index, seq_leg, max_len, offset_data
