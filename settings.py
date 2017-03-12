BATCH_SIZE  = 1
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.5       # Initial learning rate.
NUM_EPOCHS_PER_DECAY = 350
MOVING_AVERAGE_DECAY = 0.9999
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2223 # number of images under segmentationClass file
NUM_CLASSES = 21
DEBUG = True
#depth = [3,96,256,384,384,256,4096,4096,21]
#depth = [3,96,256,256,384,21]
layer_depth = {'conv1': (3,96), 'conv2':(96,256),'conv3': (256,256), 'conv4':(256,384),'conv5':(384,21) }
layer_depth = {'conv1': (3,21)}
MAX_STEPS = 10000
log_device_placement = False
#shuffle = False
shuffle = True
summaries_dir = 'summary/'