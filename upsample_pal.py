import tensorflow as tf
#put all globals in settings
import settings
def upsampling(down_sampled_batch_logits, original_shape):
	"""
  From Wei
  Args:
     down_sampled_batch_logits: batch down sampled logits ([batch, down_height, down_width, classes])
     original_shape: tensor of the original shape ([batch, height, width])
  Returns:
    upsampled logits with same size as the original images.
  """
  pass