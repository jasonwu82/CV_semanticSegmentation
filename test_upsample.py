#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

### upsample functions
import upsample_pal as MyUpSample


##===============================================================##
## input image
##===============================================================##
filename_in = "machi.jpg"
img1 = scp.misc.imread(filename_in)
img1_shape = img1.shape

num_classes=img1_shape[2]

##===============================================================##
## tf session
##===============================================================##
with tf.Session() as sess:
	images = tf.placeholder("float", )
	feed_dict = {images: img1}
	batch_images = tf.expand_dims(images, 0)

	print('==> Building Network...')
	with tf.name_scope("upsample"):
		upsample_op = MyUpSample._upsample(batch_images, num_classes=num_classes, 
                                           name="upsample", debug=True,
                                           ksize=32, stride=32)	
		
	print('==> Init tensorflow session...')
	init = tf.global_variables_initializer()
	sess.run(init)

	print('==> Running tf session (the Network)...')
	upsampled = sess.run(upsample_op, feed_dict=feed_dict)
	

img_up = upsampled
img_up_shape = img_up.shape


##===============================================================##
## save & plot results
##===============================================================##
print('==> Saving upsampled image...')
filename_out = filename_in.rsplit('.', 1)[0] + "_upsampled.jpg"
scp.misc.imsave(filename_out, img_up)
	

print('==> Ploting images...')
f, axarr = plt.subplots(1,2, sharex=True)
axarr[0].imshow(img1)
axarr[0].set_title('Original image %dx%d\n ("%s")' %(img1_shape[0], img1_shape[1], filename_in))
axarr[1].imshow(img_up)
axarr[1].set_title('Upsampled image %dx%d\n ("%s")' %(img_up_shape[0],img_up_shape[1], filename_out))
plt.show()	
