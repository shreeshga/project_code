"""

Converts tiffs, jpgs, etc. to RGB values and save in file called data for processing by convolutional_mlp.py

"""


import numpy, time, cPickle, gzip, sys, os
from PIL import Image
import theano
import theano.tensor as T
from pprint import pprint 
import random

size = (64, 64)
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
        im = None
        im = background
        # im.save('../data/%s/%s' %(dir_name, myfile + ".thumbnail"), "JPEG")

# convert to bitmap
    if len(im.split()) == 4:
        r, g, b, a = im.split()
        im = Image.merge("RGB", (r,g,b))
        im = im.convert('L')
        im.save('../data/%s/%s' %(dir_name, f+".bmp"))
    else:
        im = im.convert('L')
        im.save('../data/%s/%s' %(dir_name, f+".bmp"))

def convert_image_to_seq(im):
    seq = []
    #print "convert img seq"
    #pprint (im.info)
    for s in im.getdata():
        if type(s) == int:
            b = s
        else:
            b = s[0]
        a = float(b) / 256.0
        seq.append(float(a))
    #print 'Image Seq :', len(seq)
    return seq
    #return numpy.array(seq, dtype=theano.config.floatX)

def load_data_hollywood():

    make_bitmap('test_file')
    make_bitmap('brad_pitt')
    make_bitmap('daniel_radcliffe')
    make_bitmap('johnny_depp')
    make_bitmap('ryan_reynolds')
    
    test_set_x = []
    test_set_y = []
    train_set_x = []
    train_set_y = []
    valid_set_x = []
    valid_set_y = []

#  converts thumbnail to bitmap and appends resulting string to test_set_x
    for f in os.listdir('../data/test_file'):
        myfile, ext = os.path.splitext(f)
        if ext == '.bmp':
            im = Image.open('../data/%s/%s' %('test_file', myfile+ext))
            if im.size != size:
                im = im.resize(size)
            test_set_x += convert_image_to_seq(im)
    
    test_set_y = [0, 1, 2, 3]
    index_train = []
    index_valid = []
    i = 0
    num_valid = 0
    num_train = 0
    for d in dirs:
        for myfile in os.listdir('../data/%s' %d):
            f, ext = os.path.splitext(myfile)
            if ext == '.bmp':
                im = Image.open('../data/%s/%s' %(d, f+ext))
                if im.size != size:
                    im = im.resize(size)
                if (i%2) == 0:
                    index_train.append(num_train)
                    train_set_x += convert_image_to_seq(im)
                    #print len(train_set_x)
                    if d == 'brad_pitt':
                        train_set_y.append(0)
                    elif d == 'daniel_radcliffe':
                        train_set_y.append(1)
                    elif d == 'johnny_depp':
                        train_set_y.append(2)
                    else:
                        train_set_y.append(3)
                    num_train += 1
                else:
                    index_valid.append(num_valid)
                    valid_set_x += convert_image_to_seq(im)
                    if d == 'brad_pitt':
                        valid_set_y.append(0)
                    elif d == 'daniel_radcliffe':
                        valid_set_y.append(1)
                    elif d == 'johnny_depp':
                        valid_set_y.append(2)
                    else:
                        valid_set_y.append(3)
                    num_valid += 1
                #i = i+1
                
    valid_set_x = list(train_set_x)
    valid_set_y = list(train_set_y)

    train_reorder_x = []
    train_reorder_y = []
    valid_reorder_x = []
    valid_reorder_y = []
    #print num_valid,num_train
    '''
    random.shuffle(index_train)
    random.shuffle(index_valid)
    for k in index_train:
        train_reorder_x.append(train_set_x[k])
        train_reorder_y.append(train_set_y[k])

    for k in index_valid:
        valid_reorder_x.append(valid_set_x[k])
        valid_reorder_y.append(valid_set_y[k])

    train_set_x = train_reorder_x
    train_set_y = train_reorder_y
    valid_set_x = valid_reorder_x
    valid_set_y = valid_reorder_y
    '''

    def shared_dataset(data_x, data_y):
        """ Function that loads the dataset into shared variables
        
        The reason we store our dataset in shared variables is to allow 
        Theano to copy it into the GPU memory (when code is run on GPU). 
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared 
        variable) would lead to a large decrease in performance.
        """
        data_x = numpy.asarray(data_x, dtype=theano.config.floatX)
        data_y = numpy.asarray(data_y, dtype=theano.config.floatX)
        return data_x, data_y
#        shared_x = theano.shared(data_x);
#        shared_y = theano.shared(data_y);
#        return shared_x, T.cast(shared_y, 'int32')

#    a = train_set_x
    '''
    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    '''

    #print train_set_x.get_value().shape
    #x = 784
    #for b in a:
    #    if(x != len(b)):
    #        print len(b)
    #raw_input("Enter something to continue...")
    #print 'train set size: ',len(train_set_x)
    rval = [(train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__': 

#    load_data_hollywood()
    make_bitmap('test_file')
    make_bitmap('brad_pitt')
    make_bitmap('daniel_radcliffe')
    make_bitmap('johnny_depp')
    make_bitmap('ryan_reynolds')

#    a = [[0.1, 0.2],[0.3,0.4]]
#    b = numpy.asarray(a, dtype=theano.config.floatX)
#    print 'train_set_y  = %s, %s, %s' %(type(b[0][0]), type(b[0]), type(b))
