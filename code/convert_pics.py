"""

Converts tiffs, jpgs, etc. to RGB values and save in file called data for processing by convolutional_mlp.py

"""


import numpy, time, cPickle, gzip, sys, os
from PIL import Image
import theano
import theano.tensor as T
from pprint import pprint 
import random

size = (128, 128)
data_class_count = 4
# 1 = bp 2 = dr 3 = jd 4 = rr
dirs = ['brad_pitt', 'daniel_radcliffe', 'johnny_depp', 'ryan_reynolds']

def make_bitmap(dir_name):
    for myfile in os.listdir("../data/%s" %dir_name):
        # reduce size to 128 by 128 thumbnail first
        f, ext = os.path.splitext(myfile)
        if ext == '.jpg':
            im = Image.open('../data/%s/%s' %(dir_name, myfile))
            im.thumbnail(size, Image.NEAREST)
            background = Image.new('RGBA', size, (255, 255, 255, 0))
            background.paste(
                    im,
                    ((size[0] - im.size[0]) / 2, (size[1] - im.size[1]) / 2))
            im = background
    	        
            # convert to bitmap
            if len(im.split()) == 4:
                r, g, b, a = im.split()
                im = Image.merge("RGB", (r,g,b))
            
            im = im.convert('L')
            if im.size != size:
                im = im.resize(size)
            im.save('../data/%s/%s' %(dir_name, f+".bmp"))

def convert_image_to_seq(im):
    seq = []
    for s in im.getdata():
        if type(s) == int:
            b = s
        else:
            # TODO need to figure out if this is the correct thing to do. 
            # Picking the first entry of the 3-tuple.
            b = s[0]
        a = b / 256.0
        seq.append(a)
    return seq

def load_data_hollywood():
    print "... loading data from custom dataset"
    test_set_x = []
    test_set_y = []
    train_set_x = []
    train_set_y = []
    valid_set_x = []
    valid_set_y = []

    def resize_bitmap_image(im):
        background = Image.new('L', size, 255)
        background.paste(
                im,
                ((size[0] - im.size[0]) / 2, (size[1] - im.size[1]) / 2))
        im = background
        return im;

    for f in os.listdir('../data/test_file'):
        myfile, ext = os.path.splitext(f)
        if ext == '.bmp' and f.startswith("cropped"):
            im = Image.open('../data/test_file/%s' %(myfile+ext))
            test_set_x += convert_image_to_seq(resize_bitmap_image(im))

    def add_entry_to_dataset(train_set_x, train_set_y, im):
        train_set_x += convert_image_to_seq(resize_bitmap_image(im))
        if d == 'brad_pitt':
            train_set_y.append(0)
        elif d == 'daniel_radcliffe':
            train_set_y.append(1)
        elif d == 'johnny_depp':
            train_set_y.append(2)
        else:
            train_set_y.append(3)
    
    test_set_y = [0, 1, 2, 3]
    i = 0
    for d in dirs:
        for myfile in os.listdir('../data/%s' %d):
            f, ext = os.path.splitext(myfile)
            if ext == '.bmp' and f.startswith("cropped"):
                im = Image.open('../data/%s/%s' %(d, f+ext))
                if (i%4) == 0:
                    add_entry_to_dataset(valid_set_x, valid_set_y, im)
                else:
                    add_entry_to_dataset(train_set_x, train_set_y, im)
                i += 1

    #print len(train_set_x), len(train_set_y), len(valid_set_x), len(valid_set_y), len(test_set_x), len(test_set_y),
    rval = [(train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__': 

    make_bitmap('test_file')
    make_bitmap('brad_pitt')
    make_bitmap('daniel_radcliffe')
    make_bitmap('johnny_depp')
    make_bitmap('ryan_reynolds')

