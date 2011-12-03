"""

Converts tiffs, jpgs, etc. to RGB values and save in file called data for processing by convolutional_mlp.py

"""


import os
import random
import struct
from PIL import Image

size = (28, 28)
data_dir = '../data/dot_no_dot/'
data_class_count = 2

def convert_image_to_seq(im):
    seq = []
    for s in im.getdata():
        a = s / 256.0
        seq.append(a)
    return seq

def load_data_hollywood():
    print "... loading data from Dot-No-Dot Dataset"
    test_set = []
    train_set = []
    valid_set = []
    data_sets = [train_set, valid_set, test_set]

    for f in os.listdir(data_dir):
        name, ext = os.path.splitext(f)
        if ext == ".pgm":
            im = Image.open(data_dir +"/"+ f)
            label = 1
            if name.find("No") != -1 :
                label = 0
            input_data = (convert_image_to_seq(im), label)
            set_index = 0
            if name.find("Test") != -1 :
                set_index = 2
            elif name.find("Valid") != -1 :
                set_index = 1
            data_sets[set_index].append(input_data)
                             
    def extract_data_tuples(data_set):
        train_x = []
        train_y = []
        for data in data_set:
            x, y = data
            train_x += x
            train_y.append(y)
        return (train_x, train_y)

    rval = []
    for data_set in data_sets:
        rval.append(extract_data_tuples(data_set))
        
    return rval

if __name__ == '__main__': 
    load_data_hollywood()
