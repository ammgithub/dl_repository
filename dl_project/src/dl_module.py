"""
Created on March 31, 2017

"""
__author__ = 'amm'
__date__  = "March 31, 2017"
__version__ = 0.0

import numpy as np
from array import array
from struct import unpack
from itertools import izip as zip
from itertools import count

np.set_printoptions(linewidth = 100, edgeitems = 'all', suppress = True, 
                 precision = 4)

def load_mnist_images(fname):
    """Load MNIST images, for training and testing.  
    Training set size is 60000.  Test set size is 10000.
    
    Parameters
    ----------
    In    : fname
    Out   : trainX, testX
    
    Examples
    --------
    trainX = load_mnist_labels('train-images-idx3-ubyte')
    testX = load_mnist_labels('t10k-images-idx3-ubyte')
    """
    f = open(fname, 'rb')
    dataheader = unpack(">IIII", f.read(16))
    magic = dataheader[0]
    num_samples = dataheader[1]
    num_rows = dataheader[2]
    num_cols = dataheader[3]
    assert magic == 2051, "Check magic."
    assert num_samples == 10000 or num_samples == 60000, "Check num_samples."
    assert num_rows == 28, "Check num_rows."
    assert num_cols == 28, "Check num_cols."
    # unsigned char, 1 byte "B"
    images_raw = array("B", f.read())
    f.close()
    images_array = np.array(images_raw)
    
    # returns pixels nicely
    images = np.reshape(images_array, (num_samples, num_rows, num_cols))
    # Use 'F' to slice column-major
    images = np.reshape(images, (num_samples, num_rows*num_cols), 'F')
  
    # Rescale to interval [0,1]
    return images / 255.0

def load_mnist_labels(fname):
    """Load MNIST labels, for training and testing.  
    Training set size is 60000.  Test set size is 10000.
    
    Parameters
    ----------
    In    : fname
    Out   : trainY, testY
    
    Examples
    --------
    trainY = load_mnist_labels('train-labels-idx1-ubyte')
    testY = load_mnist_labels('t10k-labels-idx1-ubyte')
    """
    f = open(fname, 'rb')
    dataheader = unpack(">II", f.read(8))
    magic = dataheader[0]
    num_samples = dataheader[1]
    assert magic == 2049, "Check magic."
    assert num_samples == 10000 or num_samples == 60000, "Check num_samples."
    # unsigned char, 1 byte "B"
    labels_raw = array("B", f.read())
    f.close()
    labels = np.array(labels_raw)
    
    return labels

def load_mnist(boolean_first_two, shuffle_flag=True, plot_flag=False):
    """Load MNIST data: 
    Loads MNIST data from directory specified in datapath. The only 
    preprocessing done at this point is a division by the max value 
    in the trainX and testX, 255. Hence input data is scaled to the
    interval [0, 1]. 
    
    Note: design matrix has dimension (num_samples x num_attributes)
    
    execfile('C:\\Users\\amalysch\\git\\dl_repository\\dl_project\\src\\dl_module.py')
    
    Parameters
    ----------
    boolean_first_two  :  boolean, decides whehter to load only zeros and ones
    plot_flag          :  boolean, plots first and last 5 images for training
                          testing
    shuffle_flag       :  boolean, decides whehter to shuffle or not (mostly for 
                          debugging)
    
    Examples
    --------
    trainX, testX, trainY, testY = load_mnist(boolean_first_two, shuffle_flag, plot_flag)
    trainX, testX, trainY, testY = load_mnist(boolean_first_two)
    trainX, testX, trainY, testY = load_mnist(1, True, False)
    trainX, testX, trainY, testY = load_mnist(0, True, True)
    trainX, testX, trainY, testY = load_mnist(1)
    trainX, testX, trainY, testY = load_mnist(0)
    """
    datapath = 'C:\\Users\\amalysch\\DATA'
    fname_train_images = 'train-images-idx3-ubyte'
    fname_train_labels = 'train-labels-idx1-ubyte'
    fname_test_images = 't10k-images-idx3-ubyte'
    fname_test_labels = 't10k-labels-idx1-ubyte'
    trainX = load_mnist_images('\\'.join([datapath, fname_train_images]))
    trainY = load_mnist_labels('\\'.join([datapath, fname_train_labels]))
    testX = load_mnist_images('\\'.join([datapath, fname_test_images]))
    testY = load_mnist_labels('\\'.join([datapath, fname_test_labels]))
    
    if boolean_first_two:
        # keep only labels 0 or 1: 
        # 12665 training samples (5923 x 0, 6742 x 1)
        # 2115 test samples (980 x 0, 1135 x 1)
        # Group samples with 0s and 1s, needed only for debugging, see reshuffle
        # TODO: list comprehension can probably be shortened.
        trX0 = np.array([trainX[i, :] for i, j in enumerate(trainY) if j == 0])
        trX1 = np.array([trainX[i, :] for i, j in enumerate(trainY) if j == 1])
        trY0 = np.array([trainY[i] for i, j in enumerate(trainY) if j == 0])
        trY1 = np.array([trainY[i] for i, j in enumerate(trainY) if j == 1])
        del trainX, trainY
        trainX = np.vstack((trX0, trX1))
        trainY = np.hstack((trY0, trY1))
        del trX0, trX1, trY0, trY1
        
        tstX0 = np.array([testX[i, :] for i, j in enumerate(testY) if j == 0])
        tstX1 = np.array([testX[i, :] for i, j in enumerate(testY) if j == 1])
        tstY0 = np.array([testY[i] for i, j in enumerate(testY) if j == 0])
        tstY1 = np.array([testY[i] for i, j in enumerate(testY) if j == 1])
        del testX, testY # Save memory
        testX = np.vstack((tstX0, tstX1))
        testY = np.hstack((tstY0, tstY1))
        del tstX0, tstX1, tstY0, tstY1

#         # TODO: delete after testing (0s and 1s are mixed)
#         trainX = np.array([trainX[i, :] for i, j in enumerate(trainY) if j == 0 or j == 1])
#         trainY = np.array([trainY[i] for i, j in enumerate(trainY) if j == 0 or j == 1])
#         testX = np.array([testX[i, :] for i, j in enumerate(testY) if j == 0 or j == 1])
#         testY = np.array([testY[i] for i, j in enumerate(testY) if j == 0 or j == 1])
    
    if shuffle_flag: 
        # Reshuffle train and test set
        idx = list(np.random.permutation(len(trainY)))
        trainX = np.array([trainX[i, :] for i in idx])
        trainY = np.array([trainY[i] for i in idx])
        idx = list(np.random.permutation(len(testY)))
        testX = np.array([testX[i, :] for i in idx])
        testY = np.array([testY[i] for i in idx])
    
    if plot_flag: 
        print "\nPrinting FIRST 5 training images."
        for i in range(5):
            render = check_output(trainX[i, :], 28, 28); print render
            print trainY[i]
        print "\nPrinting LAST 5 training images."
        for i in range(5):
            render = check_output(trainX[-(i+1), :], 28, 28); print render
            print trainY[-(i+1)]
        print "\nPrinting FIRST 5 test images."
        for i in range(5):
            render = check_output(testX[i, :], 28, 28); print render
            print testY[i]
        print "\nPrinting LAST 5 test images."
        for i in range(5):
            render = check_output(testX[-(i+1), :], 28, 28); print render
            print testY[-(i+1)]
        
    # Standardize training set: mean = 0, sample_std = 1
    m = trainX.mean(axis=0, keepdims=1) # Design matrix is transposed, axis=0 not axis=1
    s = trainX.std(axis=0, ddof=1, keepdims=1)
    trX = trainX - m
    del trainX
    trainX = np.divide(trX, s+0.1)
    
    # Standardize test set with training mean and training sample_std
    tstX = testX - m
    del testX
    testX = np.divide(tstX, s+0.1)
    
    return trainX, testX, trainY, testY

def check_output(single_image, num_rows, num_cols):
    """Check output. 
    
    Parameters
    ----------
    single_image : individual sample from trainX or testX
    num_rows     : number of rows (MNIST, 28)
    num_cols     : number of cols (MNIST, 28)
    
    Examples
    --------
    render = check_output(single_image, num_rows, num_cols)
    render = check_output(single_image, 28, 28)
    render = check_output(trX[0, :], 28, 28); print render
    for i in range(20):
    render = check_output(trX[i, :], 28, 28); print render
    print trY[i]
    
    """
    assert single_image.ndim == 1, "Check only single sample trainX or testX."
    a = np.reshape(single_image, (num_rows, num_cols), 'F')
    single_image = np.reshape(a, (1, num_rows*num_cols)).flatten()
    render = ''
    for i in range(len(single_image)):
        if i % num_cols == 0: render += '\n'
        # input data is rescaled to interval [0, 1]
        if 255*single_image[i] > 10:
            render += 'X'
        else:
            render += '.'
    return render
    
if __name__ == '__main__':
    print "Running..."
    # Select only 0s and 1s
    boolean_first_two = 0
    shuffle_flag = True
    plot_flag = True
    
    trainX, testX, trainY, testY = load_mnist(boolean_first_two, shuffle_flag, plot_flag)
    
    # When shuffle is off: testing first and last sample
    if not(shuffle_flag) and boolean_first_two:
        assert trainX[0,:].sum() == 39.561047980868921, \
        "Zeros and ones only: check training input."
        assert trainX[12664,:].sum() == -120.36191028696315, \
        "Zeros and ones only: check training input."
        assert testX[0,:].sum() == 88.945449293439594, \
        "Zeros and ones only: check test input."
        assert testX[2114,:].sum() == -66.791473643172466, \
        "Zeros and ones only: check test input."
    elif not(shuffle_flag) and not(boolean_first_two):
        assert trainX[0,:].sum() == 19.28293399623935, \
        "All samples: check training input."
        assert trainX[12664,:].sum() == -144.33543748893206, \
        "All samples: check training input."
        assert testX[0,:].sum() == -60.315061997267819, \
        "All samples: check test input."
        assert testX[2114,:].sum() == 2.0797377929497003, \
        "All samples: check test input."
    
    print "Finished."
        
    
    
    
    
