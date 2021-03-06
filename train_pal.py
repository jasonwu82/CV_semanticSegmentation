import tensorflow as tf
import settings
import re
LEARNING_RATE_DECAY_FACTOR = settings.LEARNING_RATE_DECAY_FACTOR  # Learning rate decay factor.
INITIAL_LEARNING_RATE = settings.INITIAL_LEARNING_RATE       # Initial learning rate.
def debug_tensor(tensor):
  if settings.DEBUG:
    return tf.Print(tensor, [tensor.name,tf.shape(tensor), tensor], message="Tensor is: ",summarize=10)
  else:
    return tensor
def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  TOWER_NAME = 'tower'
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var
def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  
  var = tf.get_variable(
      name,
      shape,
      initializer=tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var
def conv_layer(in_data,depth,layer_name,isTrain=False):
  with tf.variable_scope(layer_name) as scope:
    in_depth = depth[0]
    out_depth = depth[1]
    print(layer_name + " depth: ", depth)
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, in_depth, out_depth],
                                         stddev=5e-2,
                                         wd=0.0)
    if layer_name=='conv1':
      tensor = kernel
      kernel = tf.Print(tensor, [tensor.name,tf.shape(tensor), tensor], message="Tensor is: ",summarize=100)
      #kernel = debug_tensor(kernel)
    #conv = tf.nn.conv2d(in_data, kernel, [1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(in_data, kernel, [1, 1, 1, 1], padding='SAME')
    
    
    biases = _variable_on_cpu('biases', [out_depth], tf.constant_initializer(0.0))
    if layer_name == 'conv1':
      biases = debug_tensor(biases)
    pre_activation = tf.nn.bias_add(conv, biases)
    layer_res = tf.nn.relu(pre_activation, name=scope.name)
    #layer_res = tf.sigmoid(pre_activation, name=scope.name)
    #layer_res = tf.nn.relu6(pre_activation, name=scope.name)
    #if layer_name != 'conv_6_nopool' and layer_name != 'conv_7_nopool':
    if layer_name not in {'conv_6_nopool','conv_7_nopool'}:
      print(layer_name, " max pooled")
      layer_res = tf.nn.max_pool(layer_res, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    else:
      keep_prob = 1
      if isTrain == True:
        keep_prob = 0.7
      print(layer_name, " use dropout at probability = ", keep_prob)
      layer_res = tf.nn.dropout(layer_res, keep_prob=keep_prob)
    #conv_layer_dict[layer_name] = layer_res
    _activation_summary(layer_res)
    return layer_res
def inference(images,isTrain=False):
  """Build the PASCAL model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  
  images = debug_tensor(images)
  #conv_layer_dict = {}
  
  conv1 = conv_layer(images,settings.layer_depth['conv1'],'conv1')
  conv2 = conv_layer(conv1,settings.layer_depth['conv2'],'conv2')
  conv3 = conv_layer(conv2,settings.layer_depth['conv3'],'conv3')
  conv4 = conv_layer(conv3,settings.layer_depth['conv4'],'conv4')
  conv5 = conv_layer(conv4,settings.layer_depth['conv5'],'conv5')
  conv_6_nopool = conv_layer(conv5,settings.layer_depth['conv_6_nopool'],'conv_6_nopool',isTrain)
  conv_7_nopool = conv_layer(conv_6_nopool,settings.layer_depth['conv_7_nopool'],'conv_7_nopool',isTrain)
  conv_7_nopool = debug_tensor(conv_7_nopool)

  deconv32 = []
  with tf.variable_scope('deconv_32') as scope:
    
    b = tf.get_variable('bias',shape=[settings.NUM_CLASSES]
      ,initializer=tf.constant_initializer(0.0))
    
    b = debug_tensor(b)
    
    w = tf.get_variable("weight",shape=[64,64, settings.NUM_CLASSES,settings.layer_depth['conv_7_nopool'][1]],
     #initializer=tf.truncated_normal_initializer(stddev=5e-2))
     initializer=tf.constant_initializer(0.25))
    
    w = debug_tensor(w)
    
    out_shape = tf.stack([settings.BATCH_SIZE,tf.shape(images)[1],tf.shape(images)[2],settings.NUM_CLASSES])
    out_shape = debug_tensor(out_shape)
    deconv = tf.nn.conv2d_transpose(conv_7_nopool, 
      w, output_shape=out_shape, strides=[1, 32, 32, 1], padding="SAME")
    deconv32 = tf.nn.bias_add(deconv, b)
    deconv32 = debug_tensor(deconv32)

  
  return deconv32

def train(total_loss, global_step):
  """
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  '''
  num_batches_per_epoch = settings.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / settings.BATCH_SIZE
  decay_steps = int(num_batches_per_epoch * settings.NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  lr = debug_tensor(lr)
  tf.summary.scalar('learning_rate', lr)
  '''
  # Generate moving averages of all losses and associated summaries.
  # here the loss_average_op is only for 
  # loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  #with tf.control_dependencies([loss_averages_op]):
  #opt = tf.train.GradientDescentOptimizer(lr)
  opt = tf.train.AdamOptimizer(settings.INITIAL_LEARNING_RATE)
  grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)
  

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      settings.MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


