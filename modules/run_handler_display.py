from .mnist.mnist import MNIST
import numpy as np
from scipy.optimize import minimize, check_grad


def main(train_images_fname: str = None,
         train_labels_fname: str = None,
         test_images_fname: str = None,
         test_labels_fname: str = None) -> str:

    if train_images_fname is not None:
        print("\nDisplaying MNIST data ...")
        mnist_data = MNIST(tr_images_fname=train_images_fname,
                           tr_labels_fname=train_labels_fname,
                           ts_images_fname=test_images_fname,
                           ts_labels_fname=test_labels_fname,
                           two_class_flag=False,
                           visual_flag=True,
                           shuffle_flag=False)
