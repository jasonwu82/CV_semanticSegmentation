import tensorflow as tf
#put all globals in settings
import settings

import numpy as np
from math import ceil


##===============================================================##
## 
##===============================================================##
def _upsample(value, num_classes, name, debug,
                       ksize=64, stride=32):
					   			   
	## Must have strides[0] = strides[3] = 1. 
	## For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].  
	strides = [1, stride, stride, 1]

	with tf.variable_scope(name):
		#--- create filter output_shape from input_shape
		in_shape = tf.shape(value)
		
		h = ((in_shape[1] - 1) * stride) + 1
		w = ((in_shape[2] - 1) * stride) + 1
		new_shape = [in_shape[0], h, w, num_classes]
		
		#--- output_shape size
		output_shape = tf.stack(new_shape)

		#--- create filter weights from filter_shape
		in_features = num_classes
		#in_features = value.get_shape()[3].value
		#logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
		f_shape = [ksize, ksize, num_classes, in_features]
		weights = get_deconv_filter(f_shape)

		#--- bilinear filtering (deconv)
		deconv = tf.nn.conv2d_transpose(value, weights, output_shape, strides=strides, padding='SAME')

		if debug:
			deconv = tf.Print(deconv, [tf.shape(deconv)], 
                              message='Shape of %s' % name,
                              summarize=4, first_n=1)
			
		#--- Sqeeze & Convert from float to uint8 (for testing upsample of raw image) 
		deconv_out = tf.bitcast(tf.cast(tf.squeeze(deconv), tf.int8), tf.uint8)
		
		#if debug:
		#	deconv_out = tf.Print(deconv_out, [tf.shape(deconv_out)], 
        #                      message='Shape of %s (deconv_out)' % name,
        #                      summarize=4, first_n=1)		
	return deconv_out 


##===============================================================##
## Generate bilinear filter weights
##===============================================================##
def get_deconv_filter(f_shape):
	width = f_shape[0]
	heigh = f_shape[0]   ##should be f_shape[1] ????
	f = ceil(width/2.0)
	c = (2 * f - 1 - f % 2) / (2.0 * f)
	bilinear = np.zeros([f_shape[0], f_shape[1]])
	for x in range(width):
		for y in range(heigh):
			value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
			bilinear[x, y] = value
	weights = np.zeros(f_shape)
	for i in range(f_shape[2]):
		weights[:, :, i, i] = bilinear

	init = tf.constant_initializer(value=weights, dtype=tf.float32)
	return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)
		