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

from nn_concat import model
from nn_concat import nn_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('architecture', '5layer',
                           """Network acrhitecture.""")
tf.app.flags.DEFINE_string('eval_dir', '../xray_data/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../xray_data/',
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
batch_size = 40#FLAGS.batch_size


def eval_once(saver, images, prob_op, data_dir, ids):
  """Run Eval once.

  Args:
    saver: Saver.
    images: input pl.
    prob_op: Probability op.
    data_dir: dataset directory.
    ids: filenames.
  """
  global num_examples, batch_size
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

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(len(ids) / batch_size))
      result_size = [len(ids), model.NUM_CLASSES]
      pred_all = np.zeros(result_size)
      gt_all = None#np.zeros(result_size)
      step = 0
      while step < num_iter and not coord.should_stop():
        imgdata = np.zeros([batch_size, model.IMAGE_SIZE, model.IMAGE_SIZE, 1])
        batch_end = min( (step+1)*batch_size, len(ids))
        imgdata[:batch_end - step*batch_size, :, :, :] = nn_input.synthetic_input_by_ids_np(data_dir, ids[step*batch_size:batch_end])
        pred = sess.run(prob_op, feed_dict={images: imgdata})
        pred_all[step*batch_size:batch_end, :] = pred[:batch_end - step*batch_size, :]

        print('%d / %d' % (step + 1, num_iter))
        step += 1

      # Compute precision @ 1.
      #if mask is not None:
      #  pred_all = pred_all[:, mask]
      #  gt_all = gt_all[:, mask]
      #gt_all = gt_all.astype(np.int)
      #pred_all_th = (pred_all > 0.5).astype(np.int)
      #true_count = (pred_all_th == gt_all).astype(np.int).sum()
      #precision = true_count / gt_all.size
      #ap = average_precision_score(gt_all, pred_all_th, average=None)
      #map = np.mean(ap)
      #print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      #print('AP = ')
      #print(ap)
      #print('mAP = %f' % map)
      #np.save('pred.npy', pred_all)
      #np.save('gt.npy', gt_all)

      #summary = tf.Summary()
      #summary.ParseFromString(sess.run(summary_op))
      #summary.value.add(tag='Precision @ 1', simple_value=precision)
      #summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    return pred_all, gt_all


def evaluate(data_dir='', ids=None):
  """Eval CIFAR-10 for a number of steps."""
  global num_examples, batch_size  # will reset if data is from file list
  mask = None  # used for inconsistent tags (real data)
  with tf.Graph().as_default() as g:
    images = tf.placeholder(np.float32, [batch_size, model.IMAGE_SIZE, model.IMAGE_SIZE, 1])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    probs = tf.sigmoid(model.inference(images))

    ## Calculate predictions.
    #top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.no_op(name='no_summary')	#tf.train.SummaryWriter(eval_dir, g)

    while True:
#(saver, images, prob_op, ids)
      pred_all, gt_all = eval_once(saver, images, probs, data_dir, ids)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

    return pred_all, gt_all


def main(argv=None):  # pylint: disable=unused-argument
  if '--test' in argv[1:]:
    # ids = [ ['45f18c5c_varied_sm', '{:08x}'.format(i)]
    #         for i in range(200)]
    ids = ['{:08x}'.format(i) for i in range(200)]
    evaluate('../ImageTagger/test', ids)
  else:
    evaluate()


if __name__ == '__main__':
  #tf.app.run()
  pass
