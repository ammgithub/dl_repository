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

    print("\nRunning softmax regression on all classes (1,...,10) ...")
    print("\nOptimizing ...\n")
    # softmax_regression_vec: ~35 seconds,  softmax_regression: ~1500 seconds
    res = minimize(softmax_regression, theta0, args=(mnist_data.trainX, mnist_data.trainY, weight_decay),
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

    print("Softmax: Accuracy for the training set: {:.1f}%".format(100 * accuracy_train))
    print("Softmax: Accuracy for the test set: {:.1f}%".format(100 * accuracy_test))

    result = 'Done'
    print('\n', result)
    print("Done with run_handler_softmax", '\n')
    return result


def softmax_regression(theta, trainX, trainY, weight_decay=False):
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
    num_samples = trainX.shape[0]  # samples = 60000
    num_attributes = trainX.shape[1]  # num_attributes = 785 = 784 + 1
    num_classes = int(theta.size / float(num_attributes) + 1)  # 10
    # theta_mat is skinny
    theta_mat = vec_to_mat(theta, num_attributes, num_classes - 1)  # 785 x 9

    # Initialize
    fval = 0.0
    error = np.zeros((num_samples, num_classes - 1))
    for i in range(num_samples):
        denom = 0.0
        for j in range(num_classes - 1):
            denom = denom + np.exp(np.inner(theta_mat[:, j], trainX[i, :]))
        for j in range(num_classes - 1):
            error[i, j] = 1 * (trainY[i] == j) - \
                          np.exp(np.inner(theta_mat[:, j], trainX[i, :])) / denom
            p = np.exp(np.inner(theta_mat[:, j], trainX[i, :])) / denom
            fval -= (1 * (trainY[i] == j) * np.log(p))
    grad_mat = -np.dot(trainX.T, error)
    # Negative gradient for a minimization, must be flattened for np.minimize
    grad = mat_to_vec(grad_mat).flatten()
    #     grad = np.dot(trainX.T, error).flatten()
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


