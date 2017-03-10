from input_assg2 import *
from PIL import Image
import settings
BATCH_SIZE = settings.BATCH_SIZE

DATA_DIR = './data/TrainVal/VOCdevkit/VOC2011/JPEGImages'
LABEL_DIR = './data/TrainVal/VOCdevkit/VOC2011/SegmentationClass'
filenames = read_filenames_from_txt('./data/TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt')

#filenames = filenames[0:10]
print('read in %d (data, labels) files' %len(filenames))
data_queue,label_queue = create_queue(filenames,DATA_DIR,LABEL_DIR)
result = read_PAS(data_queue, label_queue)
images_batch = []
labels_batch = []
#if BATCH_SIZE > 1:
images_batch,labels_batch = generate_image_and_label_batch(result.data, result.label, min_queue_examples=20,
                                    batch_size=BATCH_SIZE, shuffle=True)
#else:
#	images_batch,labels_batch = tf.expand_dims(result.data, 0), tf.expand_dims(result.label, 0)


#init_op = tf.initialize_all_variables()
init_op = tf.global_variables_initializer()
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
	for i in range(settings.BATCH_SIZE):
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