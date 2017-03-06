from input_assg2 import *
from PIL import Image
import settings
import train_pal

def actual_train():
	with tf.Graph().as_default():
		#Returns and create (if necessary) the global step variable
	    global_step = tf.contrib.framework.get_or_create_global_step()
	    images_batch,labels_batch = generate_image_and_label_batch(result.data, result.label, min_queue_examples=20,
                                    batch_size=BATCH_SIZE, shuffle=True)
	    # Get images and labels for CIFAR-10.
	    #images, labels = cifar10.distorted_inputs()

	    # Build a Graph that computes the logits predictions from the
	    # inference model.
	    logits = train_pal.inference(images_batch)

	    # Calculate loss.
	    #loss = cifar10.loss(logits, labels)

	    # Build a Graph that trains the model with one batch of examples and
	    # updates the model parameters.
	    train_op = cifar10.train(loss, global_step)

	    class _LoggerHook(tf.train.SessionRunHook):
	      """Logs loss and runtime."""

	      def begin(self):
	        self._step = -1

	      def before_run(self, run_context):
	        self._step += 1
	        self._start_time = time.time()
	        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

	      def after_run(self, run_context, run_values):
	        duration = time.time() - self._start_time
	        loss_value = run_values.results
	        if self._step % 10 == 0:
	          num_examples_per_step = FLAGS.batch_size
	          examples_per_sec = num_examples_per_step / duration
	          sec_per_batch = float(duration)

	          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
	                        'sec/batch)')
	          print (format_str % (datetime.now(), self._step, loss_value,
	                               examples_per_sec, sec_per_batch))

	    with tf.train.MonitoredTrainingSession(
	        checkpoint_dir=FLAGS.train_dir,
	        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
	               tf.train.NanTensorHook(loss),
	               _LoggerHook()],
	        config=tf.ConfigProto(
	            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
	    	while not mon_sess.should_stop():
	    		mon_sess.run(train_op)
BATCH_SIZE = settings.BATCH_SIZE

DATA_DIR = './data/TrainVal/VOCdevkit/VOC2011/JPEGImages'
LABEL_DIR = './data/TrainVal/VOCdevkit/VOC2011/SegmentationClass'
filenames = read_filenames_from_txt('./data/TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt')

#filenames = filenames[0:10]
print('read in %d (data, labels) files' %len(filenames))
data_queue,label_queue = create_queue(filenames,DATA_DIR,LABEL_DIR)
result = read_PAS(data_queue, label_queue)
#images_batch,labels_batch = generate_image_and_label_batch(result.data, result.label, min_queue_examples=20,
#                                    batch_size=BATCH_SIZE, shuffle=True)
actual_train()

'''
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	#image_tensor,file_name_tensor,label_file_name_tensor = sess.run([result.data,result.data_file_name,result.label_file_name])
	images_batch_tensor, labels_batch_tensor = sess.run([images_batch,labels_batch])
	#print(images_batch_tensor[0])
	#print(file_name_tensor)
	#print(label_file_name_tensor)
	#print(image_tensor)

	# show image of the first element in batch
	# to check correctness
	for i in range(BATCH_SIZE):
		tmp = images_batch_tensor[i]
		tmp = tmp.astype('uint8')
		img = Image.fromarray(tmp, 'RGB')
		img.show()
		tmp = labels_batch_tensor[i]
		tmp = tmp.astype('uint8')
		img = Image.fromarray(tmp, 'RGB')
		img.show()
	#
	coord.request_stop()
	coord.join(threads, stop_grace_period_secs=10)
'''