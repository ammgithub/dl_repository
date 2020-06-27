from .mnist.mnist import MNIST
import numpy as np
from scipy.optimize import minimize, check_grad


def main(train_images_fname: str = None,
         train_labels_fname: str = None,
         test_images_fname: str = None,
         test_labels_fname: str = None) -> str:

    data_list = []
    if train_images_fname is not None:
        mnist_data = MNIST(tr_images_fname=train_images_fname,
                           tr_labels_fname=train_labels_fname,
                           ts_images_fname=test_images_fname,
                           ts_labels_fname=test_labels_fname,
                           two_class_flag=False,
                           visual_flag=False,
                           shuffle_flag=False)
    num_classes = 10
    weight_decay = False
    if weight_decay:
        print("Weight decay is on.")

    # theta is num_samples x num_classes-1  (softmax has one redundant class)
    theta0_mat = 0.001 * np.random.uniform(0, 1, (mnist_data.trainX.shape[1] + 1, num_classes - 1))
    theta0 = mat_to_vec(theta0_mat)

    # results in memory error even for reduced sample sizes
    #         print("Softmax: Checking gradient for theta0 (vector of length %d) ..." \
    #                 % theta0.shape[0])
    #         check_gradient = check_grad(softmax_regression_vec_fun, \
    #                                     softmax_regression_vec_gradient, \
    #                                     theta0, trainX, trainY)
    #         print("Softmax: Difference (2-Norm) between closed form and approximation: %3.6f" \
    #                 % check_gradient)

    print("\nRunning vectorised softmax regression on all classes (1,...,10) ...")
    print("\nOptimizing ...\n")
    # softmax_regression_vec: ~35 seconds,  softmax_regression: ~1500 seconds
    res = minimize(softmax_regression_vec, theta0, args=(mnist_data.trainX, mnist_data.trainY, weight_decay),
                   method='cg', jac=True, options={'disp': True})

    print("Optimization successful? %s" % res.success)
    print("Optimization status: %d" % res.status)
    print("Optimization message: %s" % res.message)
    theta = res.x
    fvalopt = res.fun
    gradopt = res.jac

    # expand theta to include the last class.
    theta = np.hstack((theta, np.zeros(mnist_data.trainX.shape[1] + 1)))  # (7850, 1)
    accuracy_train = get_multiclass_accuracy(theta, mnist_data.trainX, mnist_data.trainY)
    accuracy_test = get_multiclass_accuracy(theta, mnist_data.testX, mnist_data.testY)

    print("Softmax Vec: Accuracy for the training set: {:.1f}%".format(100 * accuracy_train))
    print("Softmax Vec: Accuracy for the test set: {:.1f}%".format(100 * accuracy_test))

    result = 'Done'
    print('\n', result)
    print("Done with run_handler_softmax_vec", '\n')
    return result


def softmax_regression_vec(theta, trainX, trainY, weight_decay=False):
    """Compute softmax regression function values.
    Same as softmax_regression, but vectorized.
    This function is significantly faster, 1500 seconds vs 35 seconds.

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
    num_samples = trainX.shape[0]  # samples = 60000
    num_attributes = trainX.shape[1]  # num_attributes = 785 = 784 + 1
    num_classes = int(theta.size / float(num_attributes) + 1);  # 10
    # theta_mat is skinny
    theta_mat = vec_to_mat(theta, num_attributes, num_classes - 1)  # 785 x 9

    if weight_decay: mylambda = 2.0  # 1.0

    # A is skinny (num_samples x num_classes-1) = (60000, 9)
    A = np.inner(theta_mat.T, trainX).T
    B = np.exp(A)
    # denom = B.sum(axis=1), replicate column (num_samples x num_classes)
    denom = np.array([list(B.sum(axis=1)), ] * (num_classes - 1)).transpose()
    # probabilities (num_samples x num_classes-1)
    P = B / denom

    # 0-1 array num_samples x num_classes-1
    mask = np.zeros((trainY.shape[0], num_classes - 1))
    for j in range(num_classes - 1):
        mask[:, j] = 1 * (trainY == j)

    # Objective function
    sum_over_classes = (mask * np.log(P)).sum(axis=1)  # (num_samples x 1) flattened
    if weight_decay:
        # weight decay
        fval = -sum_over_classes.sum(axis=0) / float(num_samples) + \
               0.5 * mylambda * (theta_mat * theta_mat).sum(axis=1).sum(axis=0)
    else:
        # sum over all samples m
        fval = -sum_over_classes.sum(axis=0) / float(num_samples)

        # Gradient
    error = mask - P
    if weight_decay:
        # weight decay
        grad_mat = -np.dot(trainX.T, error) / float(num_samples) + \
                   mylambda * theta_mat
    else:
        grad_mat = -np.dot(trainX.T, error) / float(num_samples)

    # Negative gradient for a minimization, must be flattened for np.minimize
    grad = mat_to_vec(grad_mat).flatten()
    return fval, grad


def get_multiclass_accuracy(theta, X, Y):
    """compute the accuracy for softmax (multi-class case).
    x and y can be either training data (trainx, trainy) or
    test data (testx, testy).

    in    : theta, trainx, trainy or theta, testx, testy
    out   : accuracy

    examples
    --------
    accuracy = get_multiclass_accuracy(theta, x, y)
    """
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    num_attributes = X.shape[1]  # num_attributes = 785 = 784 + 1
    # Have added zeros for last class (785 x 10)
    num_classes = int(theta.size / float(num_attributes))  # theta: (7850, 1)
    theta_mat = vec_to_mat(theta, num_attributes, num_classes)  # 785 x 10
    t = np.inner(theta_mat.T, X).T  # (60000, 10)
    label_idx = np.argmax(t, axis=1)
    return sum(1 * (Y == label_idx)) / float(len(Y))


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
    num_samples = trainX.shape[0]  # samples = 60000
    num_attributes = trainX.shape[1]  # num_attributes = 785 = 784 + 1
    num_classes = int(theta.size / float(num_attributes) + 1);  # 10
    # theta_mat is skinny
    theta_mat = vec_to_mat(theta, num_attributes, num_classes - 1)  # 785 x 9

    # A is skinny (num_samples x num_classes-1) = (60000, 9)
    theta_mat.T.shape
    trainX.shape
    A = np.inner(theta_mat.T, trainX).T
    B = np.exp(A)
    # denom = B.sum(axis=1), replicate column (num_samples x num_classes)
    denom = np.array([list(B.sum(axis=1)), ] * (num_classes - 1)).transpose()
    # probabilities (num_samples x num_classes-1)
    P = B / denom

    # 0-1 array num_samples x num_classes-1
    mask = np.zeros((trainY.shape[0], num_classes - 1))
    for j in range(num_classes - 1):
        mask[:, j] = 1 * (trainY == j)

    # Objective function
    sum_over_classes = (mask * np.log(P)).sum(axis=1)  # (num_samples x 1) flattened
    fval = -sum_over_classes.sum(axis=0) / float(num_samples)  # sum over all samples m

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
    num_samples = trainX.shape[0]  # samples = 60000
    num_attributes = trainX.shape[1]  # num_attributes = 785 = 784 + 1
    num_classes = int(theta.size / float(num_attributes) + 1);  # 10
    # theta_mat is skinny
    theta_mat = vec_to_mat(theta, num_attributes, num_classes - 1)  # 785 x 9

    # A is skinny (num_samples x num_classes-1) = (60000, 9)
    A = np.inner(theta_mat.T, trainX).T
    B = np.exp(A)
    # denom = B.sum(axis=1), replicate column (num_samples x num_classes)
    denom = np.array([list(B.sum(axis=1)), ] * (num_classes - 1)).transpose()
    # probabilities (num_samples x num_classes-1)
    P = B / denom

    # 0-1 array num_samples x num_classes-1
    mask = np.zeros((trainY.shape[0], num_classes - 1))
    for j in range(num_classes - 1):
        mask[:, j] = 1 * (trainY == j)

    # Gradient
    error = mask - P
    grad_mat = -np.dot(trainX.T, error) / float(num_samples)
    # Negative gradient for a minimization, must be flattened for np.minimize
    grad = mat_to_vec(grad_mat).flatten()
    return grad


