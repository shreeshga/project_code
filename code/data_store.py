"""

Converts tiffs, jpgs, etc. to RGB values and save in file called data for processing by convolutional_mlp.py

"""


import os
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
    test_set_x = []
    test_set_y = []
    train_set_x = []
    train_set_y = []
    valid_set_x = []
    valid_set_y = []

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
                if a > 0 :
                    a -= 1
                    test_set_x += convert_image_to_seq(im);
                    test_set_y.append(label);
                elif b > 0:
                    b -= 1
                    valid_set_x += convert_image_to_seq(im);
                    valid_set_y.append(label);
                else:
                    train_set_x += convert_image_to_seq(im);
                    train_set_y.append(label);

    rval = [(train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)]
    return rval

#if __name__ == '__main__': 
#    load_data_hollywood()
