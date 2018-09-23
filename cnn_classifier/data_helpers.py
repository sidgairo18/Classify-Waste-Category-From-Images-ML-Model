import cv2
import numpy as np
import tensorflow as tf
import pdb

#This file is the data helper file which contains the helper functions like reading and manipulating data.

#example Shape function

a = tf.truncated_normal([16,128,128,3])
sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(tf.shape(a))

b = tf.reshape(a, [16, 49152])
sess.run(tf.shape(b))

'''
1. For the final layer we use softmax probabilities.
2. Training data - 80% of the images.
3. Validation data - 20% of the images.
4. Test set - Sample a separate testing set.
'''
