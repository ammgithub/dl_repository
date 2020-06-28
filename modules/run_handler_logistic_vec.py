from .mnist.mnist import MNIST
import numpy as np
from scipy.optimize import minimize, check_grad


def main(train_images_fname: str = None,
         train_labels_fname: str = None,
         test_images_fname: str = None,
         test_labels_fname: str = None) -> str:

    if train_images_fname is not None:
        mnist_data = MNIST(tr_images_fname=train_images_fname,
                           tr_labels_fname=train_labels_fname,
                           ts_images_fname=test_images_fname,
                           ts_labels_fname=test_labels_fname,
                           two_class_flag=True,
                           visual_flag=False,
                           shuffle_flag=True)

    print("\nRunning logistic regression on binary data (0, 1) ...")
    theta0 = 0.001 * np.random.uniform(0, 1, (mnist_data.trainX.shape[1] + 1, 1)).flatten()
    print("Logistic regression: Checking gradient for theta0 (vector of length %d) ..."
          % theta0.shape[0])
    check_gradient = check_grad(logistic_regression_vec_fun,
                                logistic_regression_vec_gradient,
                                theta0, mnist_data.trainX, mnist_data.trainY)
    print("Difference (2-Norm) between closed form and approximation: %3.6f"
          % check_gradient)

    print("\nOptimizing ...\n")
    # logistic_regression ~ 35 seconds, logistic_regression_vec ~ 7 seconds
    res = minimize(logistic_regression_vec, theta0, args=(mnist_data.trainX, mnist_data.trainY),
                   method='cg', jac=True, options={'disp': True})

    print("Optimization successful? %s" % res.success)
    print("Optimization status: %d" % res.status)
    print("Optimization message: %s" % res.message)
    theta = res.x
    fvalopt = res.fun
    gradopt = res.jac
    accuracy_train = get_binary_accuracy(theta, mnist_data.trainX, mnist_data.trainY)
    accuracy_test = get_binary_accuracy(theta, mnist_data.testX, mnist_data.testY)

    print("Accuracy for the training set: {:.1f}%".format(100 * accuracy_train))
    print("Accuracy for the test set: {:.1f}%".format(100 * accuracy_test))

    result = 'Done'
    print('\n', result)
    print("Done with run_handler_logistic_vec", '\n')
    return result


def logistic_regression_vec(theta, trainX, trainY):
    """Compute logistic regression function values and gradient.
    Same as logistic_regression, but vectorized.
    This function is significantly faster, 35 seconds vs 7 seconds.

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
    h = np.where(h == 1.0, 1 - 1e-12, h)
    fval = -(trainY * np.log(h) + (1 - trainY) * np.log(1 - h)).sum()
    error = h - trainY
    # Negative gradient for a minimization, must be flattened for np.minimize
    grad = np.dot(trainX.T, error).flatten()
    return fval, grad


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
    h = np.where(h == 1.0, 1 - 1e-12, h)
    fval_only = -(trainY * np.log(h) + (1 - trainY) * np.log(1 - h)).sum()
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
    h = np.where(h == 1.0, 1 - 1e-12, h)
    error = h - trainY
    grad_only = np.dot(trainX.T, error).flatten()
    return grad_only


def get_binary_accuracy(theta, X, Y):
    """Compute the accuracy. X and Y can be either training data (trainX, trainY)
    or test data (testX, testY).

    In    : theta, trainX, trainY or theta, testX, testY
    Out   : accuracy

    Examples
    --------
    accuracy = get_binary_accuracy(theta, X, Y)
    """
    res_bool = sigmoid(np.sign(np.dot(np.hstack((np.ones((X.shape[0], 1)),
                                                 X)), theta))) > 0.5
    # multiplication by 1 converts True/False to 1/0, conversion int to float
    return sum((Y == res_bool * 1) * 1) / float(len(Y))


def sigmoid(x):
    """
    Computes: 1. / (1 + np.exp(-x))
    Examples
    --------
    sig = sigmoid(x)
    """
    if isinstance(x, list): x = np.array(x)
    return 1.0 / (1.0 + np.exp(-x))
