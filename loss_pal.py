import tensorflow as tf
#put all globals in settings
import settings
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
  pass