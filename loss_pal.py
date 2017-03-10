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
	
	comparison = tf.equal( labels, tf.constant(255) )
	labels_new = labels.assign( tf.where(comparison, tf.zeros_like(labels), labels) )
	
	#Calculate the average cross entropy loss across the batch.
	labels_new = tf.cast(labels_new, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)
	
	#The total loss is defined as the cross entropy loss plus all of the weight
	#decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')	
	
""" 
init_a=tf.constant([[1,2,3,4],[4,3,2,1]])
a=tf.Variable(init_a)
start_op =  tf.initialize_all_variables()
comparison = tf.equal( a, tf.constant(4) )
conditional_assignment_op = a.assign( tf.where(comparison, tf.zeros_like(a), a) )

with tf.Session() as session:
    # Equivalent to: a = np.array( [1, 2, 3, 1] )
    session.run( start_op )
    print( a.eval() )

    # Equivalent to: a[a==1] = 0
    session.run( conditional_assignment_op )
    print( a.eval() )
"""