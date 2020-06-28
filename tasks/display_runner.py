"""
runfile('display_runner.py', wdir='/home/admin_thinkpad/Git/dl_repository/tasks')
"""
from modules import run_handler_display
import time as mytime

start_time = mytime.time()
input_file_path = r'/home/admin_thinkpad/Git/dl_repository/tests/Case01/Inputs//'

result = run_handler_display.main(train_images_fname=input_file_path + 'train-images-idx3-ubyte',
                                  train_labels_fname=input_file_path + 'train-labels-idx1-ubyte',
                                  test_images_fname=input_file_path + 't10k-images-idx3-ubyte',
                                  test_labels_fname=input_file_path + 't10k-labels-idx1-ubyte')

print("Overall time: {}".format(round(mytime.time() - start_time, 2)))
