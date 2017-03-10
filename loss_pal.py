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
  logits=np.asarray(logits)
  labels=np.asarray(labels)
  
  #take exponential
  logits=np.exp(logits)
  
  num_classes=logits.shape[3]
  #process labels, so that it's size is [batch, height, width, classes].
  #if label is 255, the vector for that pixel is all ones
  #otherwise the vector is 1 on the correct class and 0 elsewhere
  labels_exp=np.zeros(logits.shape)
  for index,val in np.ndenumerate(labels):
	if val<num_classes:
	  labels_exp[index][val]=1
	else:#label==255
	  labels_exp[index]=np.ones(num_classes)  
  
  #numerator of e^y/sum(e^i), shape [batch, height, width]
  numerator = np.einsum("ijkl,ijkl->ijk",logits,labels_exp)
  
  #denominator of e^y/sum(e^i), shape [batch, height, width]
  denominator=np.sum(logits,axis=3)
  
  #loss for each pixel in each batch, 
  loss_tensor=np.divide(numerator,denominator)
  loss_tensor=-np.log(loss_tensor)
  
  #loss for each image in the batch(summed over pixels), shape [batch]
  loss_batch=np.sum(loss_tesor,axis=(2,3))
  
  return loss_batch
  