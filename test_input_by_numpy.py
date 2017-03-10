from input_by_numpy import *
import numpy as np
#from PIL import Image
import scipy.misc as misc
readimg = readIMage('./data/TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt',
	'./data/TrainVal/VOCdevkit/VOC2011/JPEGImages',
	'./data/TrainVal/VOCdevkit/VOC2011/SegmentationClass')
#(res_image,res_label) = readimg.read_next_natch()

#print(out[0])
#print(out[1])
#print(out[0][0])
#print(out[1][0])
np.set_printoptions(threshold=np.nan)

for j in range(1):
	(res_image,res_label) = readimg.read_next_natch()
	for i in range(len(res_image)):
		#print(np.array(out[0][i]))
		img = Image.fromarray(np.array(res_image[i]), 'RGB')
		#print(res_image[i][0])
		#misc.imshow(res_image[i])
		#misc.imshow(res_label[i])
		img.show()
		#print(out[1][0])
		#print(np.array(out[1][i]))
		#res_label[i] = np.array(res_label[i])
		#res_label[i] = res_label[i].astype('uint8')
		#print(res_label[i])
		#print(out[1][i].shape())
		img = Image.fromarray(res_label[i])
		#.convert('RGB')
		img.show()
