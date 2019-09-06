# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np

from nn_dual import nn_input
from tagio.tag import tagdata

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 40,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../xray_data',
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = nn_input.IMAGE_SIZE
NUM_CLASSES = nn_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer, trainable=True):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd, trainable=True):
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
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev),
                         trainable=trainable)
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs(shuffle=True):
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
   raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'synthetic_data_binary')
  return nn_input.distorted_inputs(data_dir=data_dir,
                                   batch_size=FLAGS.batch_size,
                                   shuffle=shuffle)


def inputs(eval_data, num_threads=16):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  if eval_data:
    data_dir = os.path.join(FLAGS.data_dir, 'real_data_binary')
  else:
    data_dir = os.path.join(FLAGS.data_dir, 'synthetic_data_binary')
  return nn_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.batch_size, num_threads=num_threads)


def inference(images, oneds, architecture='5_layer_dual_dropout', dropout=False, keep_prob=0.5):
  """ from myalexnet_forward.py
  Args:
    images: images from distorted_inputs() or inputs().
  Returns:
    logits from final fully connected layer.
  """
  flag_train_conv_wghts = False
  conv_weights = []

  with tf.variable_scope('conv1') as scope:
    weights = _variable_with_weight_decay('weights', shape = [11, 11, 1, 96],
                                          stddev=0.01, wd=0.0,
                                          trainable=flag_train_conv_wghts)
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.1),
                              trainable=flag_train_conv_wghts)
    conv = tf.nn.conv2d(images, weights, [1,4,4,1], padding='SAME')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

    # visualize the learned filter
    filter_layer1_width = 8
    filter_layer1_height = 12
    filter_w = 11
    filter_d = 1
    filter_layer1 = tf.transpose(tf.reshape(weights, ([filter_w,filter_w,filter_d, filter_layer1_width, filter_layer1_height])), [4, 3, 0, 1, 2])
    filter_layer1 = tf.split(0, filter_layer1_height, filter_layer1)
    for height_idx in range(0, filter_layer1_height):
      filter_layer1[height_idx] = tf.reshape(filter_layer1[height_idx], [filter_layer1_width, filter_w, filter_w, filter_d])
      tmp = tf.split(0, filter_layer1_width, filter_layer1[height_idx])
      for width_idx in range(0, filter_layer1_width):
        tmp[width_idx] = tf.reshape(tmp[width_idx], [filter_w,filter_w])
      filter_layer1[height_idx] = tf.transpose(tf.concat(0, tmp))
    filter_layer1 = tf.concat(0, filter_layer1)
    filter_layer1 = tf.reshape(filter_layer1, [1, filter_layer1_height*filter_w, filter_layer1_width*filter_w, 1])
    tf.image_summary('filter1', filter_layer1)

  pool1 = tf.nn.max_pool(conv1, ksize=[1,4,4,1], strides=[1,2,2,1], padding='VALID', name='pool1')
  conv_weights.extend([weights, biases])

  with tf.variable_scope('conv2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[5,5,96,192],
                                          stddev=0.01, wd=0.0,
                                          trainable=flag_train_conv_wghts)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1),
                              trainable=flag_train_conv_wghts)
    conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  pool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
  conv_weights.extend([weights, biases])

  with tf.variable_scope('conv3') as scope:
    weights = _variable_with_weight_decay('weights', shape=[3,3,192,384],
                                          stddev=0.01, wd=0.0,
                                          trainable=flag_train_conv_wghts)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1),
                              trainable=flag_train_conv_wghts)
    conv = tf.nn.conv2d(pool2, weights, [1, 1, 1, 1], padding='SAME')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)
  conv_weights.extend([weights, biases])

  with tf.variable_scope('conv4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[3, 3, 384, 256],
                                          stddev=0.01, wd=0.0,
                                          trainable=flag_train_conv_wghts)
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1),
                              trainable=flag_train_conv_wghts)
    conv = tf.nn.conv2d(conv3, weights, [1, 1, 1, 1], padding='SAME')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)
  conv_weights.extend([weights, biases])

  with tf.variable_scope('conv5') as scope:
    weights = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256],
                                          stddev=0.01, wd=0.0,
                                          trainable=flag_train_conv_wghts)
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1),
                              trainable=flag_train_conv_wghts)
    conv = tf.nn.conv2d(conv4, weights, [1, 1, 1, 1], padding='SAME')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv5)
  conv_weights.extend([weights, biases])

  pool3 = tf.nn.max_pool(conv5, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')

  ''' START 1D CNN '''
  if architecture == '5_layer_dual_dropout':    # legacy naming; 5-layer image + 3-layer curve CNNs
    with tf.variable_scope('conv1d1') as scope:
      expand_tensor_1 = tf.expand_dims(oneds, 1)
      expand_tensor = tf.expand_dims(expand_tensor_1, -1) # [batch_size, 1, 256, 1]
      weights = _variable_with_weight_decay('weights', shape=[1, 7, 1, 64],
                                            stddev=0.01, wd=0.0)
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      conv1d = tf.nn.conv2d(expand_tensor, weights, [1, 1, 4, 1], padding='SAME')
      bias = tf.nn.bias_add(conv1d, biases)
      conv1d1 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv1d1)
    pool1d1 = tf.nn.max_pool(conv1d1, ksize=[1, 1, 4, 1], strides=[1, 1, 2, 1], padding='VALID', name='pool1d1')

    with tf.variable_scope('conv1d2') as scope:
      weights = _variable_with_weight_decay('weights', shape=[1, 3, 64, 128],
                                            stddev=0.01, wd=0.0)
      biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
      conv1d = tf.nn.conv2d(pool1d1, weights, [1, 1, 1, 1], padding='SAME')
      bias = tf.nn.bias_add(conv1d, biases)
      conv1d2 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv1d2)
    pool1d2 = tf.nn.max_pool(conv1d2, ksize=[1, 1, 4, 1], strides=[1, 1, 2, 1], padding='VALID', name='pool1d2')

    with tf.variable_scope('conv1d3') as scope:
      weights = _variable_with_weight_decay('weights', shape=[1, 3, 128, 128],
                                            stddev=0.01, wd=0.0)
      biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
      conv1d = tf.nn.conv2d(pool1d2, weights, [1, 1, 1, 1], padding='SAME')
      bias = tf.nn.bias_add(conv1d, biases)
      conv1d3 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv1d3)
    pool1d3 = tf.nn.max_pool(conv1d3, ksize=[1, 1, 4, 1], strides=[1, 1, 2, 1], padding='VALID', name='pool1d3')

    output1d = pool1d3
  elif architecture == '5_layer_2_layer':   # 5-layer image + 2-layer curve (from CIFAR example) CNNs
    with tf.variable_scope('conv1d1') as scope:
      expand_tensor_1 = tf.expand_dims(oneds, 1)
      expand_tensor = tf.expand_dims(expand_tensor_1, -1) # [batch_size, 1, 256, 1]
      weights = _variable_with_weight_decay('weights', shape=[1, 5, 1, 64],
                                            stddev=0.01, wd=0.0)
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      conv1d = tf.nn.conv2d(expand_tensor, weights, [1, 1, 1, 1], padding='SAME')
      bias = tf.nn.bias_add(conv1d, biases)
      conv1d1 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv1d1)
    pool1d1 = tf.nn.max_pool(conv1d1, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='VALID', name='pool1d1')

    with tf.variable_scope('conv1d2') as scope:
      weights = _variable_with_weight_decay('weights', shape=[1, 5, 64, 64],
                                            stddev=0.01, wd=0.0)
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      conv1d = tf.nn.conv2d(pool1d1, weights, [1, 1, 1, 1], padding='SAME')
      bias = tf.nn.bias_add(conv1d, biases)
      conv1d2 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv1d2)
    pool1d2 = tf.nn.max_pool(conv1d2, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='VALID', name='pool1d2')

    output1d = pool1d2
  else:
    raise Exception()
  ''' END 1D CNN '''

  with tf.variable_scope('local4') as scope:
    reshape = tf.reshape(pool3, [pool3.get_shape()[0].value, -1])   # 1st dim-#images, 2nd dim-flatten feature
    reshape1d = tf.reshape(output1d, [output1d.get_shape()[0].value, -1])
    concat_tensor = tf.concat(1, [reshape, reshape1d])
    dim = concat_tensor.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 4000], stddev=0.01, wd=0.0)
    biases = _variable_on_cpu('biases', [4000], tf.constant_initializer(0.1))
    local4 = tf.nn.relu( tf.matmul(concat_tensor, weights) + biases, name=scope.name)
    _activation_summary(local4)

    weights_oneds = tf.slice(weights, [0, pool3.get_shape()[0].value],
                             [weights.get_shape()[0].value, -1], name='weights_oneds')
    tf.histogram_summary(weights.name, weights)
    tf.histogram_summary(weights_oneds.name, weights_oneds)
    if dropout:
      local4_out = tf.nn.dropout(local4, keep_prob)
    else:
      local4_out = local4

  with tf.variable_scope('local5') as scope:
    weights = _variable_with_weight_decay('weights', shape=[4000, 4000], stddev=0.01, wd=0.0)
    biases = _variable_on_cpu('biases', [4000], tf.constant_initializer(0.1))
    local5 = tf.nn.relu(tf.matmul(local4_out, weights) + biases, name=scope.name)
    _activation_summary(local5)
    if dropout:
      local5_out = tf.nn.dropout(local5, keep_prob)
    else:
      local5_out = local5

  with tf.variable_scope('sigmoid_linear') as scope:
    weights = _variable_with_weight_decay('weights', shape=[4000, NUM_CLASSES], stddev=0.01, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
    sigmoid_linear = tf.matmul(local5_out, weights) + biases
    _activation_summary(sigmoid_linear)

  # weights_all = ['conv1/weights', 'conv1/biases', 'conv2/weights', 'conv2/biases',
  #                'conv3/weights', 'conv3/biases', 'conv4/weights', 'conv4/biases',
  #                'conv5/weights', 'conv5/biases']
  # weights_all = [tf.get_variable(x) for x in weights_all]
  # output the FC layers for SVM;
  # output all conv weight tensors for referencing in fresh start (load from ckpt)
  return local4, local5, sigmoid_linear, conv_weights


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.float32)
  #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  #    logits, labels, name='cross_entropy_per_example')

  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
    logits, labels, name='componentwise_cross_entropy')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    #opt = tf.train.MomentumOptimizer(lr, 0.9)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


# def maybe_download_and_extract():
#   """Download and extract the tarball from Alex's website."""
#   dest_directory = FLAGS.data_dir
#   if not os.path.exists(dest_directory):
#     os.makedirs(dest_directory)
#   filename = DATA_URL.split('/')[-1]
#   filepath = os.path.join(dest_directory, filename)
#   if not os.path.exists(filepath):
#     def _progress(count, block_size, total_size):
#       sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
#           float(count * block_size) / float(total_size) * 100.0))
#       sys.stdout.flush()
#     filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
#     print()
#     statinfo = os.stat(filepath)
#     print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
#     tarfile.open(filepath, 'r:gz').extractall(dest_directory)
