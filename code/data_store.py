"""

Converts tiffs, jpgs, etc. to RGB values and save in file called data for processing by convolutional_mlp.py

"""


import os
import random
from PIL import Image

size = (92, 112)
data_dir = '../data/att_faces/'
data_set_class_count = 40

def convert_image_to_seq(im):
    seq = []
    for s in im.getdata():
        a = s / 256.0
        seq.append(a)
    return seq

def load_data_hollywood():
    test_set = []
    train_set = []
    valid_set = []
    data_sets = [train_set, valid_set, test_set]

    test_set_class_size = 1
    valid_set_class_size = 2;
    for f in os.listdir(data_dir):
        d = data_dir + f;
        a = test_set_class_size
        b = valid_set_class_size
        if os.path.isdir(d):
            for imgf in os.listdir(d):
                im = Image.open(d + '/' + imgf)
                label = int(f[1:]) - 1
                input_data = (convert_image_to_seq(im), label)
                if a > 0 :
                    a -= 1
                    test_set.append(input_data);
                elif b > 0:
                    b -= 1
                    valid_set.append(input_data);
                else:
                    train_set.append(input_data);

    for data_set in data_sets:
        random.shuffle(data_set)
        
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

#if __name__ == '__main__': 
#    load_data_hollywood()
