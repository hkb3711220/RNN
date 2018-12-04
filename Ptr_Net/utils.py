import numpy as np
import pandas as pd
import os
from tqdm import tqdm

os.listdir('./dataset')
os.chdir(os.path.dirname(__file__))

def read_data(file_path):

    encode_seq, encode_leg, targets_seq, targets_leg = [],[],[],[]

    with open(file_path, mode='r') as f:
        for line in tqdm(f):
            split_line = line.split('output')
            #the inputs are planar point sets P = {P1, . . . , Pn} with n elements each, where
            #Pj = (xj , yj ) are the cartesian coordinates of the points over which we find the
            #convex hull
            inputs = np.array(split_line[0].split(), dtype=np.float32).reshape((-1,2))
            targets = np.array(split_line[1].split(), dtype=np.int32)[:-1]

            encode_seq.append(inputs)
            targets_seq.append(targets)
            encode_leg.append(inputs.shape[0])
            targets_leg.append(targets.shape[0])

    return encode_seq, encode_leg, targets_seq, targets_leg

def gen_data(target):

    file_name = 'convex_hull_5_{}.txt'.format(target)
    file_path = os.path.join('./dataset', file_name)

    encode_seq, encode_leg, targets_seq, targets_leg = read_data(file_path)
    max_encode_leg = max(encode_leg)
    max_target_leg = max(targets_leg)

    encode_seq_offset = np.zeros((len(encode_leg), max_encode_leg, 2), dtype=np.float32)
    target_seq_offset = np.zeros((len(targets_leg), max_target_leg), dtype=np.int32)

    for i, leg in enumerate(encode_leg):
        encode_seq_offset[i, :leg] = encode_seq[i]

    for i , leg in enumerate(targets_leg):
        target_seq_offset[i, :leg] = targets_seq[i]

    return encode_seq_offset, target_seq_offset, encode_leg, targets_leg
