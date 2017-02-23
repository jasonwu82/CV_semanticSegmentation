import input_assg2

filenames = input_assg2.read_filenames_from_txt('./data/TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt')
print(filenames[0:10])
