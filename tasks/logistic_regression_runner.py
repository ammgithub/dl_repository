"""
runfile('logistic_regression_runner.py', wdir='/home/admin_thinkpad/Git/dl_repository/tasks')
"""
from modules import run_handler_logistic_vec, run_handler_logistic
import time as mytime

start_time = mytime.time()
input_file_path = r'/home/admin_thinkpad/Git/dl_repository/tests/Case01/Inputs//'

vec_flag = True
if vec_flag:
    result = run_handler_logistic_vec.main(train_images_fname=input_file_path + 'train-images-idx3-ubyte',
                                          train_labels_fname=input_file_path + 'train-labels-idx1-ubyte',
                                          test_images_fname=input_file_path + 't10k-images-idx3-ubyte',
                                          test_labels_fname=input_file_path + 't10k-labels-idx1-ubyte')
else:
    # Non vectorised for comparison
    result = run_handler_logistic.main(train_images_fname=input_file_path + 'train-images-idx3-ubyte',
                                      train_labels_fname=input_file_path + 'train-labels-idx1-ubyte',
                                      test_images_fname=input_file_path + 't10k-images-idx3-ubyte',
                                      test_labels_fname=input_file_path + 't10k-labels-idx1-ubyte')

print("Overall time: {}".format(round(mytime.time() - start_time, 2)))
