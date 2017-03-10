import tensorflow as tf
#put all globals in settings
import settings
import numpy as np

def loss(logits, labels):
	"""
	From Wei
	Args:
		logits: Logits from inference and upsampled(), with settings.BATCH_SIZE, settings.NUM_CLASSES 
			, and same size to the labelled image. ([batch, height, width, classes])
		labels: Batched labels from image labels, same size with the images. 
	Returns:
		Loss tensor of type float.
	"""
	labels = tf.to_int64(labels)
	comparison = tf.equal( labels, tf.constant(255,dtype=tf.int64) )
	labels_new = labels
	#labels_new = labels.assign( tf.where(comparison, tf.zeros_like(labels), labels) )
	#labels_new = tf.assign( labels,tf.where(comparison, tf.zeros_like(labels), labels) )
	labels_new = tf.where(comparison, tf.zeros_like(labels), labels)
	#Calculate the average cross entropy loss across the batch.
	#labels_new = tf.cast(labels_new, tf.int64)
	#labels_new = tf.to_int64(labels_new)
	#labels_new = tf.cast(, tf.int64)
	print(logits)
	print(labels)
	print(labels_new)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_new, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)
	
	#The total loss is defined as the cross entropy loss plus all of the weight
	#decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')	
	
