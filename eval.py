import tensorflow as tf
from input_assg2 import *
from PIL import Image
import settings
import train_pal
import time
from datetime import datetime
import loss_pal
import numpy
from input_by_numpy import *
import scipy as scp
from label_color import color_map, label_to_rgb

def eval_test(images_batch,labels_batch):
    #variable_averages = tf.train.ExponentialMovingAverage(
    #    cifar10.MOVING_AVERAGE_DECAY)
    #variables_to_restore = variable_averages.variables_to_restore()
    #saver = tf.train.Saver(variables_to_restore)	
    
    """
    images_batch=images_batch[0]
    
    labels_batch=labels_batch[0]
    images_batch=tf.expand_dims(images_batch,axis=0)
    labels_batch=tf.expand_dims(labels_batch,axis=0)
    """
    
    #labels = tf.to_int32(labels_batch)
    labels=tf.cast(labels_batch,tf.uint8)
    
    comparison = tf.equal( labels, tf.constant(255,dtype=tf.uint8) )
    labels = tf.where(comparison, tf.zeros_like(labels), labels)    
    logits = train_pal.inference(images_batch)
    
    
    #get the classes of the highest score in logits
    (top_values,top_indices)=tf.nn.top_k(logits)
    
    top_indices=tf.cast(top_indices,tf.uint8)
    #top_indices = tf.to_int32(top_indices)
    
    labels=tf.expand_dims(labels,-1)
    
    pred_correct=tf.equal(top_indices,labels)
    corr_num=tf.reduce_sum(tf.cast(pred_correct, tf.float32))
    total_num=tf.cast(tf.size(labels),tf.float32)
    accur=tf.divide(corr_num,total_num)
    
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('checkpoints/')
        saver=tf.train.Saver()
        saver.restore(sess,ckpt.model_checkpoint_path)
        (res_image,res_label) = readimg.read_next_natch(batch=BATCH_SIZE)       
        total_acc=0
        for i in range(BATCH_SIZE):
            feed_dict={images_batch: numpy.expand_dims(res_image[i],axis=0),label_batch:numpy.expand_dims(res_label[i],axis=0)}
            (preds,groundTruth,cur_acc)=sess.run([top_indices,labels,accur],feed_dict=feed_dict)
            print("current accuracy: {}".format(cur_acc))
            total_acc+=cur_acc
            
            
            
            groundTruth=numpy.squeeze(groundTruth)
            groundTruth=numpy.uint8(groundTruth)
            

            tmp2=res_label[i]
            tmp2=numpy.uint8(tmp2)
            
            
            #print("==================> ground truth <==================")
            #print(groundTruth)

            preds=numpy.squeeze(preds)
            preds=numpy.uint8(preds)   
            #print("==================> prediction <==================")         
            #print(preds)
            
            
            #logits=numpy.squeeze(logits)
            #logits=numpy.uint8(logits)   
            #print("==================> logits <==================")         
            #print(logits)
            
            img = Image.fromarray(label_to_rgb(preds,cmap),'RGB')

            img.show()
            img.save("prediction.png","PNG")
            img = Image.fromarray(label_to_rgb(groundTruth,cmap),'RGB')
            img.show()
            img.save("groundTruth.png","PNG")
            #tmp2=res_label[i]
            #tmp2=numpy.uint8(tmp2)
            #img = Image.fromarray(tmp2*10,'P')
            #img.show()
            """
            
            tmp=color_image(groundTruth, num_classes=20)
            scp.misc.imsave('test_label.png', tmp)
            """
        print("average accuracy over {} images: {}".format(BATCH_SIZE, total_acc/BATCH_SIZE))
"""
def color_image(image, num_classes=20):
    import matplotlib as mpl
    import matplotlib.cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))        
"""    
BATCH_SIZE = 5

DATA_DIR = './data/TrainVal/VOCdevkit/VOC2011/JPEGImages'
LABEL_DIR = './data/TrainVal/VOCdevkit/VOC2011/SegmentationClass'

#need to make a test.txt later to read test data later
filenames = read_filenames_from_txt('./data/TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt')
#numpy.set_printoptions(threshold=numpy.nan)
numpy.set_printoptions(threshold=1000,edgeitems=10)
#readimg = readIMage('./data/TrainVal/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt',
readimg = readIMage('./input_person_test.txt',
  './data/TrainVal/VOCdevkit/VOC2011/JPEGImages',
  './data/TrainVal/VOCdevkit/VOC2011/SegmentationClass')

print('read in %d (data, labels) files' %len(filenames))
images_batch = tf.placeholder(tf.float32, shape=(1,None,None,3),name='imageHolder')
label_batch = tf.placeholder(tf.uint8,shape=(1,None,None),name='labelHolder')
cmap=color_map(settings.NUM_CLASSES)
eval_test(images_batch,label_batch)        
        
"""       
test_logit=tf.constant([[[1,2,3],[3,2,1]],[[2,3,1],[3,2,4]],[[4,2,1],[3,4,2]]])
(top_values,top_indices)=tf.nn.top_k(test_logit)
labels=tf.constant([[2,0],[2,2],[3,1]])
labels=tf.expand_dims(labels,-1)
pred_correct=tf.equal(top_indices,labels)
corr_num=tf.reduce_sum(tf.cast(pred_correct, tf.float32))
total_num=tf.cast(tf.size(labels),tf.float32)
acc=tf.divide(corr_num,total_num)
with tf.Session() as sess:
    accuracy=sess.run(acc)
    print(test_logit.shape)
    print(test_logit.eval())
    print(top_indices.shape)
    print(top_indices.eval())
    print(pred_correct.eval())
    print(corr_num.eval())
    print(total_num.eval())
    print(accuracy)
"""