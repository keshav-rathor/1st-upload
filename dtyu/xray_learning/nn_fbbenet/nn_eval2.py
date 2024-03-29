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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

from nn_fbb import model
from nn_fbb import nn_input
import os
import sys
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('architecture', '3fc',
                           """Network acrhitecture.""")
tf.app.flags.DEFINE_string('eval_dir', '../xray_data/fbb_output/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'train_eval',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../xray_data/fbb_output/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples',
                            nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
eval_dir = FLAGS.eval_dir + FLAGS.architecture + '_eval'
checkpoint_dir = FLAGS.checkpoint_dir + FLAGS.architecture + '_train'
if FLAGS.eval_data == 'test':
  num_examples = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
else:
  num_examples = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_VAL
batch_size = FLAGS.batch_size


def eval_once(saver, coefs_op, prob_op, fc1_op, fc2_op, mask=None, noplot=False):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    prob_op: Probability op.
    gt_op: Ground truth op.
    summary_op: Summary op.
    mask: indicate the entries used for eval.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    output_dir = '/home/zquan/xray_data/fbb_output'
    feature_path = os.path.join(output_dir, 'xtest.npy')
    label_path = os.path.join(output_dir, 'ytest.npy')
    x = np.load(feature_path)
    y = np.load(label_path)
    fc1_all = np.zeros([x.shape[0], fc1_op.get_shape()[1].value])
    fc2_all = np.zeros([x.shape[0], fc2_op.get_shape()[1].value])

    n_steps = math.ceil(x.shape[0] / FLAGS.batch_size)
    for i in range(n_steps):
      print('%d / %d' % (i, n_steps))
      batch_end = min(x.shape[0], (i + 1) * FLAGS.batch_size)
      xxi = x[i * FLAGS.batch_size:batch_end, :630]
      for j in range(xxi.shape[0]):
        xxi[j, :] /= np.amax(abs(xxi[j, :]))
      mask = np.array([0, 1, 2, 3, 4, 9, 10])
      yi = y[i * FLAGS.batch_size:batch_end, mask]
      pred, fc1, fc2 = sess.run([prob_op, fc1_op, fc2_op],
                                feed_dict={coefs_op: xxi})
      fc1_all[i * FLAGS.batch_size:batch_end, :] = fc1
      fc2_all[i * FLAGS.batch_size:batch_end, :] = fc2

      if not noplot:
        # plot result!
        for j in range(xxi.shape[0]):
          fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 9), squeeze=False)
          axes[0][0].plot(xxi[j, :])
          axes[0][0].set_title('Input')
          axes[0][1].plot(yi[j, :])
          axes[0][1].plot(pred[j, :])
          axes[0][1].set_title('Predict/GT')
          axes[1][0].plot(fc1[j, :])
          axes[1][0].set_title('FC1')
          axes[1][1].plot(fc2[j, :])
          axes[1][1].set_title('FC2')
          fig.savefig(os.path.join(output_dir, 'nn_analysis/%d.png' % (i * FLAGS.batch_size + j)))
          plt.close(fig)
    np.save(os.path.join(output_dir, 'fc1_all.npy'), fc1_all)
    np.save(os.path.join(output_dir, 'fc2_all.npy'), fc2_all)

    # # Start the queue runners.
    # coord = tf.train.Coordinator()
    # try:
    #   threads = []
    #   for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
    #     threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
    #                                      start=True))
    #
    #   num_iter = int(math.ceil(num_examples / batch_size))
    #   true_count = 0  # Counts the number of correct predictions.
    #   total_sample_count = num_iter * batch_size
    #   result_size = [total_sample_count, model.NUM_CLASSES]
    #   # TODO keys
    #   pred_all = np.zeros(result_size)
    #   gt_all = np.zeros(result_size)
    #   step = 0
    #   while step < num_iter and not coord.should_stop():
    #     pred, gt = sess.run([prob_op, gt_op])
    #     pred_all[step*batch_size:(step+1)*batch_size,:] = pred
    #     gt_all[step*batch_size:(step+1)*batch_size,:] = gt
    #
    #     print('%d / %d' % (step + 1, num_iter))
    #     step += 1
    #
    #   # Compute precision @ 1.
    #   if mask is not None:
    #     pred_all = pred_all[:, mask]
    #     gt_all = gt_all[:, mask]
    #   gt_all = gt_all.astype(np.int)
    #   pred_all_th = (pred_all > 0.5).astype(np.int)
    #   true_count = (pred_all_th == gt_all).astype(np.int).sum()
    #   precision = true_count / gt_all.size
    #   ap = average_precision_score(gt_all, pred_all, average=None)
    #   map = np.mean(ap)
    #   print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    #   print('AP = ')
    #   print(ap)
    #   print('mAP = %f' % map)
    #   np.save('pred.npy', pred_all)
    #   np.save('gt.npy', gt_all)
    #
    #   #summary = tf.Summary()
    #   #summary.ParseFromString(sess.run(summary_op))
    #   #summary.value.add(tag='Precision @ 1', simple_value=precision)
    #   #summary_writer.add_summary(summary, global_step)
    # except Exception as e:  # pylint: disable=broad-except
    #   coord.request_stop(e)
    #
    # coord.request_stop()
    # coord.join(threads, stop_grace_period_secs=10)
    # return pred_all, gt_all


def evaluate(data_dir='', ids=None, fcout=False):
  """Eval CIFAR-10 for a number of steps."""
  #global num_examples, batch_size  # will reset if data is from file list
  mask = None  # used for inconsistent tags (real data)
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    coefs = tf.placeholder(tf.float32, [None, 630])
    # if eval_data:
    #   mask = np.array([0, 1, 2, 3, 4, 10, 11, 13, 14])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, fc1, fc2 = model.inference(coefs)
    probs = tf.sigmoid(logits)

    ## Calculate predictions.
    #top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(eval_dir, g)

    while True:
      eval_once(saver, coefs, probs, fc1, fc2, mask, noplot='--noplot' in sys.argv)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  #alexnet.maybe_download_and_extract()
  if tf.gfile.Exists(eval_dir):
    tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
