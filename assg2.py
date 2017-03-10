from input_assg2 import *
from PIL import Image
import settings
import train_pal
import time
import datetime
import loss_pal
import numpy
steps = 0
def actual_train():
	#with tf.Graph().as_default():
	#Returns and create (if necessary) the global step variable
    data_queue,label_queue = create_queue(filenames,DATA_DIR,LABEL_DIR)
    result = read_PAS(data_queue, label_queue)
    global_step = tf.contrib.framework.get_or_create_global_step()
    images_batch,labels_batch = generate_image_and_label_batch(result.data, result.label, min_queue_examples=3,
                                batch_size=BATCH_SIZE, shuffle=True)
    # Get images and labels for CIFAR-10.
    #images, labels = cifar10.distorted_inputs()
    print("after read")
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = train_pal.inference(images_batch)
    print("after inference")
    # Calculate loss.
    #loss = cifar10.loss(logits, labels)
    #TODO: remove this dummy
    #resized = tf.image.resize_images(input_tensor, [new_height, new_width])
    
    #tf.shape(labels_batch)
    loss = loss_pal.loss(logits,labels_batch)
    #loss = tf.ones([1,2])
    #loss = tf.reduce_sum(logits)
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    print("after reduce_sum")
    train_op = train_pal.train(loss, global_step)
    print("after train")
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
    print("haha")
    with tf.Session() as sess:
      print("@@")
      sess.run(tf.global_variables_initializer())
      print("coor")
      coord = tf.train.Coordinator()
      print("before start queue")
      threads = tf.train.start_queue_runners(coord=coord, sess=sess)
      print("before run")
      my_label = tf.get_collection('label')
      print(sess.run(my_label))
      print(sess.run([train_op]))
      print("This is %d step" %steps)
      step += 1
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
    '''    
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir='checkpoints/',
        hooks=[tf.train.StopAtStepHook(last_step=settings.MAX_STEPS),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=settings.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=mon_sess)
        print("This is %d step" %steps)
        mon_sess.run(train_op)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
      print("Train finish")
    '''
BATCH_SIZE = settings.BATCH_SIZE

DATA_DIR = './data/TrainVal/VOCdevkit/VOC2011/JPEGImages'
LABEL_DIR = './data/TrainVal/VOCdevkit/VOC2011/SegmentationClass'
filenames = read_filenames_from_txt('./data/TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt')
numpy.set_printoptions(threshold=numpy.nan)
#filenames = filenames[0:10]
print('read in %d (data, labels) files' %len(filenames))

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