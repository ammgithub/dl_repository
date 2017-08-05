"""
Created on March 31, 2017

"""
__author__ = 'amm'
__date__  = "March 31, 2017"
__version__ = 0.0

import numpy as np
from scipy.optimize import minimize, check_grad, show_options
from struct import unpack
from array import array
from time import time
# import warnings

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

def load_mnist(binary, shuffle_flag=True, visual_flag=False):
    """Load MNIST data: 
    Loads MNIST data from directory specified in datapath. The only 
    preprocessing done at this point is a division by the max value 
    in the trainX and testX, 255. Hence input data is scaled to the
    interval [0, 1]. 
    
    Note: design matrix has dimension (num_samples x num_attributes)
    
    execfile('C:\\Users\\amalysch\\git\\dl_repository\\dl_project\\src\\dl_module.py')
    
    Parameters
    ----------
    binary  :  boolean, decides whehter to load only zeros and ones
    visual_flag          :  boolean, plots first and last 5 images for training
                          testing
    shuffle_flag       :  boolean, decides whehter to shuffle or not (mostly for 
                          debugging)
    
    Examples
    --------
    trainX, testX, trainY, testY = load_mnist(binary, shuffle_flag, visual_flag)
    trainX, testX, trainY, testY = load_mnist(binary)
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
    
    if binary:
        
        ###################################################
        # binary: keep only labels 0 or 1: 
        # 12665 training samples (5923 x 0, 6742 x 1)
        # 2115 test samples (980 x 0, 1135 x 1)
        # Group samples with 0s and 1s, needed only for debugging, 
        # see shuffle
        ###################################################
        
        # Default is sorted. TODO: sorting can be removed later
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

#         # TODO: The below code filters 0s and 1s, but samples are mixed
#         trainX = np.array([trainX[i, :] for i, j in enumerate(trainY) if j == 0 or j == 1])
#         trainY = np.array([trainY[i] for i, j in enumerate(trainY) if j == 0 or j == 1])
#         testX = np.array([testX[i, :] for i, j in enumerate(testY) if j == 0 or j == 1])
#         testY = np.array([testY[i] for i, j in enumerate(testY) if j == 0 or j == 1])
    
    if shuffle_flag: 
        # shuffle train and test set
        idx = list(np.random.permutation(len(trainY)))
        trainX = np.array([trainX[i, :] for i in idx])
        trainY = np.array([trainY[i] for i in idx])
        idx = list(np.random.permutation(len(testY)))
        testX = np.array([testX[i, :] for i in idx])
        testY = np.array([testY[i] for i in idx])
    
    if visual_flag: 
        print "\nPrinting FIRST 5 training images."
        for i in range(5):
            render = check_output(trainX[i, :]); print render
            print trainY[i]
        print "\nPrinting LAST 5 training images."
        for i in range(5):
            render = check_output(trainX[-(i+1), :]); print render
            print trainY[-(i+1)]
        print "\nPrinting FIRST 5 test images."
        for i in range(5):
            render = check_output(testX[i, :]); print render
            print testY[i]
        print "\nPrinting LAST 5 test images."
        for i in range(5):
            render = check_output(testX[-(i+1), :]); print render
            print testY[-(i+1)]
    
    # Standardize training set: mean = 0, sample_std = 1
    m = trainX.mean(axis=0, keepdims=1) # Design matrix transposed, axis=0 not axis=1
    s = trainX.std(axis=0, ddof=1, keepdims=1)
    trX = trainX - m
    del trainX
    trainX = np.divide(trX, s+0.1)
    
    # Standardize test set with training mean and training sample_std
    tstX = testX - m
    del testX
    testX = np.divide(tstX, s+0.1)
    return trainX, testX, trainY, testY

def check_output(single_image):
    """Check output. 
    Note: design matrix has dimension (num_samples x num_attributes)
        
    Assume we always have:
    ----------------------
    num_rows = 28
    num_cols = 28
    
    Parameters
    ----------
    single_image : individual sample from trainX or testX
    num_rows     : number of rows (MNIST, 28)
    num_cols     : number of cols (MNIST, 28)
    
    Examples
    --------
    render = check_output(single_image)
    render = check_output(trX[0, :]); print render
    for i in range(20):
        render = check_output(trX[i, :]); print render
    """
    num_rows = 28
    num_cols = 28
    assert single_image.ndim == 1, "Check only single sample trainX or testX."
    a = np.reshape(single_image, (num_rows, num_cols), 'F')
    single_image = np.reshape(a, (1, num_rows*num_cols)).flatten()
    render = ''
    for i in range(len(single_image)):
        if i % num_cols == 0: render += '\n'
        # display_threshold = 1 => skinny display
        # display_threshold = 254 => fat display
        display_threshold = 10
        # input data is rescaled to interval [0, 1]
        if 255*single_image[i] > display_threshold:
            render += '.'
        else:
            render += 'M'
    return render

def logistic_regression(theta, trainX, trainY):
    """Compute logistic regression function values and gradient.  
    Column of ones is added to account for bias. 
    
    My data for m = 3 samples and n = 2 attributes:  
    
             | x11 x12 |    | x1 |                      | y1 | 
    trainX = | x21 x22 | =  | x2 |             trainY = | y2 |
             | x31 x32 |    | x3 |                      | y3 |
    
                           1
    h(xi) =  -------------------------------
             1 + exp(-Theta1*xi1-Theta2*xi2)
    
           m
    J = - sum ( yi*log(h(xi)) + (1-yi)*log(1-h(xi)) )
          i=1
          
    maximize    sum ( y(i)*log(h) + (1-y(i))*log(1-h) )
    minimize   -sum ( y(i)*log(h) + (1-y(i))*log(1-h) )

    Parameters
    ----------
    In    : theta, trainX, trainY
    Out   : fval, grad

    Examples
    --------
    fval, grad = logistic_regression(theta, trainX, trainY)
    """
    # Add column of ones for bias
    trainX = np.hstack((np.ones((trainX.shape[0], 1)), trainX))
    num_samples = trainX.shape[0] # samples=12665
    # Initialize
    fval = 0.0
    error = np.zeros((num_samples, 1))
    for i in range(num_samples):
        h = sigmoid(np.inner(theta, trainX[i, :]))
        # np.log(1-h) can lead to problems for h = 1.0
        h = np.where(h == 1.0, 1-1e-12, h)
        fval -= ( trainY[i]*np.log(h) + (1-trainY[i])*np.log(1-h) )
        error[i] = h - trainY[i]
    # Negative gradient for a minimization, must be flattened for np.minimize
    grad = np.dot(trainX.T, error).flatten()
    return fval, grad

def logistic_regression_vec(theta, trainX, trainY):
    """Compute logistic regression function values and gradient. 
    Same as logistic_regression, but vectorized.  
    
    Parameters
    ----------
    In    : theta, trainX, trainY
    Out   : fval, grad

    Examples
    --------
    fval, grad = logistic_regression_vec(theta, trainX, trainY)
    """
    # Add column of ones for bias
    trainX = np.hstack((np.ones((trainX.shape[0], 1)), trainX))
    h = sigmoid(np.inner(trainX, theta))
    # np.log(1-h) can lead to problems for h = 1.0
    h = np.where(h == 1.0, 1-1e-12, h)
    fval = -( trainY * np.log(h) + (1 - trainY) * np.log(1-h) ).sum()
    error = h - trainY
    # Negative gradient for a minimization, must be flattened for np.minimize
    grad = np.dot(trainX.T, error).flatten()
    return fval, grad

def get_binary_accuracy(theta, X, Y):
    """Compute the accuracy. X and Y can be either training data (trainX, trainY)
    or test data (testX, testY).  
    
    In    : theta, trainX, trainY or theta, testX, testY
    Out   : accuracy

    Examples
    --------
    accuracy = get_binary_accuracy(theta, X, Y)
    """
    res_bool = sigmoid(np.sign(np.dot(np.hstack((np.ones((X.shape[0], 1)), \
                                                  X)), theta))) > 0.5
    # multiplication by 1 converts True/False to 1/0, conversion int to float
    return sum((Y == res_bool*1)*1) / float(len(Y))
    
def sigmoid(x): 
    """
    Computes: 1. / (1 + np.exp(-x))
    Examples
    --------
    sig = sigmoid(x)
    """
    if isinstance(x, list): x = np.array(x)
    return 1.0 / (1.0 + np.exp(-x))  

def logistic_regression_vec_fun(theta, trainX, trainY):
    """For gradient check: compute logistic regression function values ONLY. 
    For more information see 'logistic_regression_vec'. 
    
    Parameters
    ----------
    In    : theta, trainX, trainY
    Out   : fval_only

    Examples
    --------
    fval_only = logistic_regression_vec(theta, trainX, trainY)
    """
    trainX = np.hstack((np.ones((trainX.shape[0], 1)), trainX))
    h = sigmoid(np.inner(trainX, theta))
    h = np.where(h == 1.0, 1-1e-12, h)
    fval_only = -( trainY * np.log(h) + (1 - trainY) * np.log(1-h) ).sum()
    return fval_only

def logistic_regression_vec_gradient(theta, trainX, trainY):
    """For gradient check: compute logistic regression gradient ONLY. 
    For more information see 'logistic_regression_vec'. 
    
    Parameters
    ----------
    In    : theta, trainX, trainY
    Out   : grad_only
    
    Examples
    --------
    grad_only = logistic_regression_vec_gradient(theta, trainX, trainY)
    """
    trainX = np.hstack((np.ones((trainX.shape[0], 1)), trainX))
    h = sigmoid(np.inner(trainX, theta))
    h = np.where(h == 1.0, 1-1e-12, h)
    error = h - trainY
    grad_only = np.dot(trainX.T, error).flatten()
    return grad_only

def softmax_regression(theta, trainX, trainY):
    """Compute softmax regression function values. Similar to 
    logistic regression a column of ones is added to account for bias. 
    
    softmax regression has one redundant class, need to analyze 
    only num_classes - 1. 
    
    My data for m = 5 samples, n = 2 attributes, and k = 3 classes:  
    
             | x11 x11 |    | x1 |                      | y1 |   | 3 |  
             | x21 x21 |    | x2 |                      | y2 |   | 1 |
    trainX = | x31 x31 | =  | x3 |             trainY = | y3 | = | 2 |
             | x41 x41 |    | x4 |                      | y4 |   | 2 |
             | x51 x51 |    | x5 |                      | y5 |   | 3 |
    
    My data for m = 5 samples and k = 3 classes:  

                | th11 th12 th13 |
                | th21 th22 th23 |
    theta_mat = | th31 th32 th33 |
                | th41 th42 th43 |
                | th51 th52 th53 |
    
    maximize    sum ( y(i)*log(h) + (1-y(i))*log(1-h) )
    minimize   -sum ( y(i)*log(h) + (1-y(i))*log(1-h) )
    
                 m   k
    minimize   -sum sum ( ( (yi==j) * log( p(yi=j) ) ) )
                i=1 j=1

    Parameters
    ----------
    In    : theta (concatenated to vector for 9 classes), trainX, trainY
    Out   : fval, grad

    Examples
    --------
    fval, grad = softmax_regression(theta, trainX, trainY)
    """
    # Add column of ones for bias, trainX is skinny
    trainX = np.hstack((np.ones((trainX.shape[0], 1)), trainX))
    num_samples = trainX.shape[0] # samples = 60000
    num_attributes = trainX.shape[1] # num_attributes = 785 = 784 + 1
    num_classes = int(theta.size / float(num_attributes) + 1); # 10
    # theta_mat is skinny
    theta_mat = vec_to_mat(theta, num_attributes, num_classes-1) # 785 x 9
    
    # Initialize
    fval = 0.0
    error = np.zeros((num_samples, num_classes-1))
    for i in range(num_samples):
        denom = 0.0
        for j in range(num_classes-1):
            denom = denom + np.exp(np.inner(theta_mat[:, j], trainX[i, :]))
        for j in range(num_classes-1):
            error[i, j] = 1*(trainY[i]==j) - \
                np.exp(np.inner(theta_mat[:, j], trainX[i, :])) / denom
            p = np.exp(np.inner(theta_mat[:, j], trainX[i, :])) / denom
            fval -= ( 1*(trainY[i]==j) * np.log(p) )
    grad_mat = np.dot(trainX.T, error)
    # Negative gradient for a minimization, must be flattened for np.minimize
    grad = mat_to_vec(grad_mat).flatten()
#     grad = np.dot(trainX.T, error).flatten()
    return fval, grad

def softmax_regression_vec(theta, trainX, trainY):
    """Compute softmax regression function values. 
    Same as softmax_regression, but vectorized.  
    
    Parameters
    ----------
    In    : theta (concatenated to vector for 9 classes), trainX, trainY
    Out   : fval, grad
    
    Examples
    --------
    fval, grad = softmax_regression_vec(theta, trainX, trainY)
    """
    # Add column of ones for bias, trainX is skinny
    trainX = np.hstack((np.ones((trainX.shape[0], 1)), trainX))
#     num_samples = trainX.shape[0] # samples = 60000
    num_attributes = trainX.shape[1] # num_attributes = 785 = 784 + 1
    num_classes = int(theta.size / float(num_attributes) + 1); # 10
    # theta_mat is skinny
    theta_mat = vec_to_mat(theta, num_attributes, num_classes-1) # 785 x 9
    
    # A is skinny (num_samples x num_classes-1) = (60000, 9)
    A = np.inner(theta_mat.T, trainX).T
    B = np.exp(A)
    # denom = B.sum(axis=1), replicate column (num_samples x num_classes)
    denom = np.array([list(B.sum(axis=1)),]*(num_classes-1)).transpose()
    # probabilities (num_samples x num_classes-1)
    P = B / denom

    # 0-1 array num_samples x num_classes-1
    mask = np.zeros((trainY.shape[0], num_classes-1))
    for j in range(num_classes-1):
        mask[:, j] = 1*(trainY == j)
        
    # Objective function 
    sum_over_classes = (mask * np.log(P)).sum(axis=1) # (num_samples x 1) flattened
    fval = -sum_over_classes.sum(axis=0) # sum over all samples m
    
    # Gradient
    error = mask - P
    grad_mat = np.dot(trainX.T, error)
    # Negative gradient for a minimization, must be flattened for np.minimize
    grad = mat_to_vec(grad_mat).flatten()
    return fval, grad

def softmax_regression_vec_fun(theta, trainX, trainY):
    """Compute softmax regression function values, fval ONLY. 
    
    Parameters
    ----------
    In    : theta (concatenated to vector for 9 classes), trainX, trainY
    Out   : fval
    
    Examples
    --------
    fval = softmax_regression_vec(theta, trainX, trainY)
    """
    # Add column of ones for bias, trainX is skinny
    trainX = np.hstack((np.ones((trainX.shape[0], 1)), trainX))
    num_attributes = trainX.shape[1] # num_attributes = 785 = 784 + 1
    num_classes = int(theta.size / float(num_attributes) + 1); # 10
    # theta_mat is skinny
    theta_mat = vec_to_mat(theta, num_attributes, num_classes-1) # 785 x 9
    
    # A is skinny (num_samples x num_classes-1) = (60000, 9)
    A = np.inner(theta_mat.T, trainX).T
    B = np.exp(A)
    # denom = B.sum(axis=1), replicate column (num_samples x num_classes)
    denom = np.array([list(B.sum(axis=1)),]*(num_classes-1)).transpose()
    # probabilities (num_samples x num_classes-1)
    P = B / denom

    # 0-1 array num_samples x num_classes-1
    mask = np.zeros((trainY.shape[0], num_classes-1))
    for j in range(num_classes-1):
        mask[:, j] = 1*(trainY == j)
        
    # Objective function 
    sum_over_classes = (mask * np.log(P)).sum(axis=1) # (num_samples x 1) flattened
    fval = -sum_over_classes.sum(axis=0) # sum over all samples m
    
    return fval

def softmax_regression_vec_gradient(theta, trainX, trainY):
    """Compute softmax regression function values, gradient ONLY. 
    
    Parameters
    ----------
    In    : theta (concatenated to vector for 9 classes), trainX, trainY
    Out   : grad
    
    Examples
    --------
    grad = softmax_regression_vec(theta, trainX, trainY)
    """
    # Add column of ones for bias, trainX is skinny
    trainX = np.hstack((np.ones((trainX.shape[0], 1)), trainX))
#     num_samples = trainX.shape[0] # samples = 60000
    num_attributes = trainX.shape[1] # num_attributes = 785 = 784 + 1
    num_classes = int(theta.size / float(num_attributes) + 1); # 10
    # theta_mat is skinny
    theta_mat = vec_to_mat(theta, num_attributes, num_classes-1) # 785 x 9
    
    # A is skinny (num_samples x num_classes-1) = (60000, 9)
    A = np.inner(theta_mat.T, trainX).T
    B = np.exp(A)
    # denom = B.sum(axis=1), replicate column (num_samples x num_classes)
    denom = np.array([list(B.sum(axis=1)),]*(num_classes-1)).transpose()
    # probabilities (num_samples x num_classes-1)
    P = B / denom

    # 0-1 array num_samples x num_classes-1
    mask = np.zeros((trainY.shape[0], num_classes-1))
    for j in range(num_classes-1):
        mask[:, j] = 1*(trainY == j)
        
    # Gradient
    error = mask - P
    grad_mat = np.dot(trainX.T, error)
    # Negative gradient for a minimization, must be flattened for np.minimize
    grad = mat_to_vec(grad_mat).flatten()
    return grad

def mat_to_vec(amat):
    """
    Convert matrix to vector (column-wise)
    
    Examples
    --------
    a = mat_to_vec(amat)
    """
    return amat.T.reshape(amat.size, 1)

def vec_to_mat(a, r, c):
    """
    Convert vector to matrix (column-wise)

    Examples
    --------
    amat = vec_to_mat(a, r, c)
    """
    return np.reshape(a, (c, r)).T

if __name__ == '__main__':
    """
    execfile('dl_module.py')
    show_options('minimize', 'SLSQP', True)
    """
    import os
    os.chdir('C:\\Users\\amalysch\\git\\dl_repository\\dl_project\src')
    # Avoid Memory error
    if globals().has_key('trainX'): del trainX, testX, trainY, testY
    
    print "\n"
    print 60 * '-'
    print 18 * ' ' + " Deep Learning Exercises "
    print 60 * '-'
    print "(1) Load MNIST data only (0, 1, ..., 9)."
    print "(2) Run logistic regression on binary data (0, 1)."
    print "(3) Run softmax regression on all classes (1, ..., 10)."
    print 60 * '-'

    invalid_input = True
    while invalid_input:
        try:
            user_in = int(raw_input("Make selection (1)-(3): "))
            invalid_input = False
        except ValueError as e:
            print "%s is not a valid selection. Please try again. "\
            %e.args[0].split(':')[1]

    if user_in == 1:
        print "\n(1) Loading MNIST data..."\
        # Select only 0s and 1s
        binary = 0
        # Shuffle data
        shuffle_flag = True
        #########################################################
        # visual_flag prints a visual render for: 
        # the FIRST 5 training images.
        # the LAST 5 training images.
        # the FIRST 5 test images.
        # the LAST 5 test images.
        #########################################################
        visual_flag = True
        trainX, testX, trainY, testY = load_mnist(binary, shuffle_flag, visual_flag)
        # Test first and last sample, if no shuffling is done.
        if not(shuffle_flag) and binary:
            assert trainX[0,:].sum() == 39.561047980868921, \
            "Zeros and ones only: check training input."
            assert trainX[12664,:].sum() == -120.36191028696315, \
            "Zeros and ones only: check training input."
            assert testX[0,:].sum() == 88.945449293439594, \
            "Zeros and ones only: check test input."
            assert testX[2114,:].sum() == -66.791473643172466, \
            "Zeros and ones only: check test input."
        elif not(shuffle_flag) and not(binary):
            assert trainX[0,:].sum() == 19.28293399623935, \
            "All samples: check training input."
            assert trainX[12664,:].sum() == -144.33543748893206, \
            "All samples: check training input."
            assert testX[0,:].sum() == -60.315061997267819, \
            "All samples: check test input."
            assert testX[2114,:].sum() == 2.0797377929497003, \
            "All samples: check test input."
    elif user_in == 2:
        print "\n(2) Running logistic regression on binary data (0, 1) ..."
        binary = 1
        shuffle_flag = True
        visual_flag = False
        trainX, testX, trainY, testY = load_mnist(binary, shuffle_flag, visual_flag)

        theta0 = 0.001*np.random.uniform(0, 1, (trainX.shape[1]+1, 1)).flatten()
        
        print "Logistic regression: Checking gradient for theta0 (vector of length %d) ..." \
                % theta0.shape[0]
        check_gradient = check_grad(logistic_regression_vec_fun, \
                                    logistic_regression_vec_gradient, \
                                    theta0, trainX, trainY)
        print "Difference (2-Norm) between closed form and approximation: %3.6f" \
                % check_gradient 
        print "\nOptimizing ...\n"
        tstart = time()
        # logistic_regression ~ 35 seconds, logistic_regression_vec ~ 7 seconds
        res = minimize(logistic_regression_vec, theta0, args = (trainX, trainY), \
                        method='cg', jac = True, options={'disp': True})
        
        print "Optimization successful? %s"%res.success
        print "Optimization status: %d"%res.status
        theta = res.x
        fvalopt = res.fun
        gradopt = res.jac
        accuracy_train = get_binary_accuracy(theta, trainX, trainY)
        accuracy_test = get_binary_accuracy(theta, testX, testY)
        
        print "Accuracy for the training set: {:.1f}%".format(100*accuracy_train)
        print "Accuracy for the test set: {:.1f}%".format(100*accuracy_test)
        print "Elapsed time: %3.1f Seconds"%(time()-tstart)
    elif user_in == 3:
        print "\n(3) Running softmax regression on all classes (1, ..., 10) ..."
        binary = 0
        num_classes = 10
        shuffle_flag = False
        visual_flag = False
        trainX, testX, trainY, testY = load_mnist(binary, shuffle_flag, visual_flag)
        
        # theta is num_samples x num_classes-1  (softmax has one redundant class)
        theta0_mat = 0.001*np.random.uniform(0, 1, (trainX.shape[1]+1, num_classes-1))
#         theta0 = theta0_mat.T.reshape(theta0_mat.size, 1)
        theta0 = mat_to_vec(theta0_mat)
        
        print "Softmax: Checking gradient for theta0 (vector of length %d) ..." \
                % theta0.shape[0]
        check_gradient = check_grad(softmax_regression_vec_fun, \
                                    softmax_regression_vec_gradient, \
                                    theta0, trainX, trainY)
        print "Softmax: Difference (2-Norm) between closed form and approximation: %3.6f" \
                % check_gradient 
        
        print "\nOptimizing ...\n"
        tstart = time()
        # logistic_regression ~ 35 seconds, logistic_regression_vec ~ 7 seconds
        res = minimize(softmax_regression_vec, theta0, args = (trainX, trainY), \
                        method='cg', jac = True, options={'disp': True})
        
        print "Optimization successful? %s"%res.success
        print "Optimization status: %d"%res.status
        theta = res.x
        fvalopt = res.fun
        gradopt = res.jac
        accuracy_train = get_binary_accuracy(theta, trainX, trainY)
        accuracy_test = get_binary_accuracy(theta, testX, testY)
        
        print "Accuracy for the training set: {:.1f}%".format(100*accuracy_train)
        print "Accuracy for the test set: {:.1f}%".format(100*accuracy_test)
        print "Elapsed time: %3.1f Seconds"%(time()-tstart)
    else:
        print "Invalid selection. Program terminating. "
    print "Finished."
        
    
    
    
    
