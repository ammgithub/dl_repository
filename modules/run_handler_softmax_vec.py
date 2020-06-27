from .mnist.mnist import MNIST


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

    result = 'Done'
    print(result, '\n')
    print("Done with run_handler_softmax", '\n')
    return result


