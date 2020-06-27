

def main(train_images_fname: str = None,
         train_labels_fname: str = None,
         test_images_fname: str = None,
         test_labels_fname: str = None) -> str:

    data_list = []
    if train_images_fname is not None:
        data_list = get_data(train_images_fname)

    result = 'Done'
    print(result, '\n')
    print("Done with run_handler_softmax", '\n')
    return result


def get_data(fname: str) -> str:
    return "Done with data"
