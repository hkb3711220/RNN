import os
import re
import tensorflow as tf
import numpy as np
import pickle

os.chdir(os.path.dirname(__file__))

def load_data(language):

    if language == 'en':
        small_vocab = open('data/small_vocab_en')
    if language == 'fr':
        small_vocab = open('data/small_vocab_fr')

    small_vocab_revised = [sequence.replace('\n', ' ') for sequence in small_vocab]
    sequence_list = [re.sub("[^\w]", " ", sequence).split() for sequence in small_vocab_revised]

    return sequence_list

def create_dictionary(sequence_list, output=None):

    text = []
    for sequence in sequence_list:
        [text.append(word) for word in sequence]
    text = list(set(text))

    word_dict = {}
    for i, word in enumerate(text):
        word_dict[word] = i

    word_dict['<PAD>'] = len(word_dict)
    if sequence_list:
        word_dict['<GO>'] = len(word_dict)
        word_dict['<EOS>'] = len(word_dict)

    with open('%s.dump'%(output), 'wb') as f:
        pickle.dump(word_dict, f)

def change_string2int(sequence_list, vocab_dict):
    text_list = []

    for sequence in sequence_list:
        text_int = []
        for word in sequence:
            if word in vocab_dict:
                text_int.append(vocab_dict[word])

        text_list.append(text_int)

    return text_list

class preprocess():

    def load_preprocess(self):

        en_dict_path = './en.dump'
        fr_dict_path = './fr.dump'
        en_exist = os.path.exists(en_dict_path)
        fr_exist = os.path.exists(fr_dict_path)

        en_text = load_data(language='en')
        if en_exist == False:
            create_dictionary(en_text, output='en')

        en_dict = pickle.load(open('en.dump', 'rb'))
        en_int_text = change_string2int(en_text, en_dict)

        fr_text = load_data(language='fr')
        if fr_exist == False:
            create_dictionary(fr_text, output='fr')

        fr_dict = pickle.load(open('fr.dump', 'rb'))
        fr_int_text = change_string2int(fr_text, fr_dict)

        return en_int_text, en_dict, fr_int_text, fr_dict

#encoder_input  = ['hello','how','are','you','<PAD>','<PAD>','<PAD'>]
#decoder_input  = ['<GO>','i','am','fine','<EOS>','<PAD>','<PAD>']
#tgt_label   = ['i','am','fine','<EOS>','<PAD>','<PAD>']
#inference phase, the output of each time step will be the input for the next time step
#preprocess the tgt label data for inference phase <go>
#saying like this is the start of the translation

sr_int_text, sr_dict, tgt_int_text, tgt_dict = preprocess().load_preprocess()

sr_seq_leg = [len(seq) for seq in sr_int_text]
max_sr_seq_leg = max(sr_seq_leg)
sr_inputs = [seq + [sr_dict['<PAD>']]*(max_sr_seq_leg-len(seq)) for seq in sr_int_text]

tgt_seq_leg = [len([sr_dict['<GO>']] + seq) for seq in tgt_int_text]
max_tgt_seq_leg = int(max(tgt_seq_leg)-1)
tgt_inputs = [[sr_dict['<GO>']] + seq + [sr_dict['<EOS>']] + [sr_dict['<PAD>']]*(max_tgt_seq_leg-len(seq)) for seq in tgt_int_text]
tgt_label = [seq + [sr_dict['<EOS>']] + [sr_dict['<PAD>']]*(max_tgt_seq_leg -len(seq)) for seq in tgt_int_text]
tgt_seq_leg = [len(seq) for seq in tgt_label]
