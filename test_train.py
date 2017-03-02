import tensorflow as tf
from train_pal import *
kernel = variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
print(tf.get_collection('losses'))