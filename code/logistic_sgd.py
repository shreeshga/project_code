"""
This tutorial introduces logistic regression using Theano and stochastic 
gradient descent.  

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability. 

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of 
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method 
suitable for large datasets, and a conjugate gradient optimization method 
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" - 
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import numpy, time, cPickle, gzip, sys, os
import numpy.lib.shape_base as shape_base
import theano
import theano.tensor as T

import convert_pics

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W` 
    and bias vector :math:`b`. Classification is done by projecting data 
    points onto a set of hyperplanes, the distance to which is used to 
    determine a class membership probability. 
    """




    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the 
                      architecture (one minibatch)
        
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in 
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in 
                      which the labels lie

        """ 

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out) 
        self.W = theano.shared(value=numpy.zeros((n_in,n_out), dtype = theano.config.floatX),
                                name='W')
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype = theano.config.floatX),
                               name='b')


        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W)+self.b)

        # compute prediction as class whose probability is maximal in 
        # symbolic form
        self.y_pred=T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]





    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e., number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP) with one row per example and one column per class 
        # LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]]
        # and T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])


    def errors(self, y):
        """Return a float representing the number of errors in the minibatch 
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the 
                  correct label
        """

        # check if y has same dimension of y_pred 
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', 
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype        
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    local_data = convert_pics.load_data_hollywood()
    #print type(local_data)    

    mdata = []
    for data in local_data:    
        x,y = data
        x = numpy.asarray(x, dtype='float32')
        y = numpy.asarray(y, dtype='int32')
        #print (a)
        #print (len(b)* 128*128)
        #print 'X :',x.shape
        #print 'Y :',y.shape[0] * 128 * 128
        x = x.reshape((y.shape[0], 64*64))
        #print a
        #c = (a,b)
        #print c
        mdata.append((x,y))
    local_data = mdata
    
    lx, ly = local_data[0]
    #print 'lx :',type(lx)    
    #print 'ly :',type(ly)    
    train_set_x = lx #T.as_tensor_variable(lx, 'train_set_x', 128)
    train_set_y = ly #T.as_tensor_variable(ly, 'train_set_y', 128)
    lx, ly = local_data[1]
    valid_set_x = lx #T.as_tensor_variable(lx, 'valid_set_x', 128)
    valid_set_y = ly #T.as_tensor_variable(ly, 'valid_set_y', 128)
    lx, ly = local_data[2]
    test_set_x = lx #T.as_tensor_variable(lx, 'test_set_x', 128)
    test_set_y = ly #T.as_tensor_variable(ly, 'test_set_y', 128)
    

    test_set = (test_set_x, test_set_y)
    train_set = (train_set_x, train_set_y)
    #print type(train_set)
    #print train_set
    valid_set = (valid_set_x, valid_set_y)
    
#    lx, ly = local_data[0]
#    test_temp_x = T.fmatrix('test_temp_x')
#    train_temp_x = T.fmatrix('train_temp_x')
#    valid_temp_x = T.fmatrix('valid_temp_x')
#    train_temp_x.reshape((128,128))
#    valid_temp_x.reshape((128,128)) 
#    test_temp_x.reshape((128,128))
#    train_temp_y = T.lvector('train_temp_y')
#    valid_temp_y = T.lvector('valid_temp_y')
#    test_temp_y = T.lvector('test_temp_y')
  #  train_temp_y.reshape(128)
  #  valid_temp_y.reshape(128)
  #  test_temp_y.reshape(128)

   # train_set = (numpy.array(train_temp_x, dtype= T.config.floatX),numpy.array(train_temp_y, ndmin=len(ly),dtype='int64'))
   # lx, ly = local_data[1]
   # valid_set = (numpy.array(valid_temp_x, dtype= T.config.floatX),numpy.array(valid_temp_y, ndmin=len(ly),dtype='int64'))
   # lx, ly = local_data[2]
   # test_set = (numpy.array(test_temp_x, dtype=T.config.floatX),numpy.array(test_temp_y, ndmin=len(ly),dtype='int64'))


     #Load the dataset 
    '''
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    '''

    #print 'Train_set :',type(train_set[0])
    #print train_set[0]
    #raw_input("Enter input")

    def shared_dataset(data_xy, local_xy, log):
        """ Function that loads the dataset into shared variables
        
        The reason we store our dataset in shared variables is to allow 
        Theano to copy it into the GPU memory (when code is run on GPU). 
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared 
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
#        local_x, local_y = local_xy
#        data_x = data_x[:len(local_y)]
#        data_y = data_y[:len(local_y)]

        #print log
#        for i in range(0, len(local_x)):
#            #print i, data_x[i].shape, local_x[i].shape
#            for j in range(0, 128*128):
#                data_x[i][j] = local_x[i][j]
#
#        for i in range(0, len(local_y)):
#            #print i, local_y[i]
#            data_y[i] = local_y[i]

        
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y))
        #print 'shared_y :', type(shared_y)               
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are 
        # floats it doesn't make sense) therefore instead of returning 
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        
        return shared_x, T.cast(shared_y, 'int32')

    # expects numpy.ndarray and not tensor.sharedvariable
#    train_set = (numpy.array(train_temp_x, ndmin=len(ly), dtype='float32'), numpy.array(train_temp_y), ndmin=len(ly), dtype='int64')
#    lx, ly = local_data[1]
#    valid_set = (numpy.array(valid_temp_x, ndmin=len(ly), dtype='float32'), numpy.array(valid_temp_y), ndmin=len(ly), dtype='int64')
#    lx, ly = local_data[2]
#    test_set = (numpy.array(test_temp_x, ndmin=len(ly), dtype='float32'), numpy.array(test_temp_y), ndmin=len(ly), dtype='int64')


    test_set_x,  test_set_y  = shared_dataset(test_set, local_data[0], 'test')
    valid_set_x, valid_set_y = shared_dataset(valid_set, local_data[1], 'valid')
    train_set_x, train_set_y = shared_dataset(train_set, local_data[2], 'train')

#    print 'train_set_y  = %s, %s, %s, %s' %(type(train_set[0][0][0]), type(train_set[0][0]), train_set_x.name, type(train_set))
#    print train_set_x.get_value().shape
#    raw_input("Enter Something...")
    rval = [(train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)]
    return rval




def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset='../data/mnist.pkl.gz',
        batch_size = 600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear 
    model

    This is demonstrated on MNIST.
    
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic 
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer 

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from 
                         http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x , test_set_y  = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  / batch_size


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'


    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch 
    x     = T.matrix('x')  # the data is presented as rasterized images
    y     = T.ivector('y') # the labels are presented as 1D vector of 
                           # [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression( input=x, n_in=64*64, n_out=10)

    # the cost we minimize during training is the negative log likelihood of 
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y) 

    # compiling a Theano function that computes the mistakes that are made by 
    # the model on a minibatch
    test_model = theano.function(inputs = [index], 
            outputs = classifier.errors(y),
            givens={
                x:test_set_x[index*batch_size:(index+1)*batch_size],
                y:test_set_y[index*batch_size:(index+1)*batch_size]})

    validate_model = theano.function( inputs = [index], 
            outputs = classifier.errors(y),
            givens={
                x:valid_set_x[index*batch_size:(index+1)*batch_size],
                y:valid_set_y[index*batch_size:(index+1)*batch_size]})

    # compute the gradient of cost with respect to theta = (W,b) 
    g_W = T.grad(cost = cost, wrt = classifier.W)
    g_b = T.grad(cost = cost, wrt = classifier.b)

    # specify how to update the parameters of the model as a dictionary
    updates ={classifier.W: classifier.W - learning_rate*g_W,\
              classifier.b: classifier.b - learning_rate*g_b}

    # compiling a Theano function `train_model` that returns the cost, but in 
    # the same time updates the parameter of the model based on the rules 
    # defined in `updates`
    train_model = theano.function(inputs = [index], 
            outputs = cost, 
            updates = updates,
            givens={
                x:train_set_x[index*batch_size:(index+1)*batch_size],
                y:train_set_y[index*batch_size:(index+1)*batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience              = 5000  # look as this many examples regardless
    patience_increase     = 2     # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = min(n_train_batches, patience/2)  
                                  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 

    best_params          = None
    best_validation_loss = numpy.inf
    test_score           = 0.
    start_time = time.clock()

    done_looping = False 
    epoch = 0  
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = epoch * n_train_batches + minibatch_index

            if (iter+1) % validation_frequency == 0: 
                # compute zero-one loss on validation set 
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index+1,n_train_batches, \
                    this_validation_loss*100.))


                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold :
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score  = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best ' 
                       'model %f %%') % \
                        (epoch, minibatch_index+1, n_train_batches,test_score*100.))

            if patience <= iter :
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %  
                 (best_validation_loss * 100., test_score*100.))
    print 'The code run for %d epochs, with %f epochs/sec'%(epoch,1.*epoch/(end_time-start_time))
    print >> sys.stderr, ('The code for file '+os.path.split(__file__)[1]+' ran for %.1fs' % ((end_time-start_time)))

if __name__ == '__main__':
    sgd_optimization_mnist()
