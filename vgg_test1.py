from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import gzip
import sys
import time
from os import path
import tempfile
import numpy

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import dtypes

from six.moves import urllib
from six.moves import xrange

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


TOWER_NAME = 'tower'

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1


class DataSet(object):

  def __init__(self, images, labels, dtype = dtypes.float32):
    dtype = dtypes.as_dtype(dtype).base_dtype
    
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)

    assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' %(images.shape, labels.shape))
    self._num_examples = images.shape[0]
    ###########
    ###########
    #print('shape[2]: ', images.shape[2])
    assert images.shape[3] == 1
    images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

    if dtype == dtypes.float32:
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    start = self._index_in_epoch

    self._index_in_epoch += batch_size
    # print('start: ', start)
    # print('index_in_epoch: ', self._index_in_epoch)
    # print('self._num_examples: ', self._num_examples)
    
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      # print('perm', perm)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      # print('batch_size: ', batch_size)
      # print('self._num_examples: ', self._num_examples)
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]
 

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
  print('Extracting', filename)
  with gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    # print('magic number: ', magic)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in image file: %s' %(magic, filename))
    num_images = _read32(bytestream)
    print('num_images: ', num_images)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype = numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, num_classes = 10):
  print('Extracting', filename)
  with gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj = f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %(magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype = numpy.uint8)

    ######ATTENTION!!! ERRORS MAY HAPPEN IF NOT USE dense_to_one_hot()######
    return dense_to_one_hot(labels, num_classes)



def read_data_sets(file_path, dtype= dtypes.float32):
  TRAIN_IMAGES = 'TrainingDataGray224.gz'
  #TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
 
  TRAIN_LABELS = 'TrainingDataLabel.txt.gz'
  #TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'

  TEST_IMAGES = 'TestingDataGray224.gz'
  #TEST_IMAGES = 't10k-images-idx3-ubyte.gz' 
 
  TEST_LABELS = 'TestingDataLabel.txt.gz'
  #TEST_LABELS = 't10k-labels-idx1-ubyte.gz' 
 
  VALIDATION_SIZE = 40

  local_file = file_path + TRAIN_IMAGES
  train_images = extract_images(local_file)

  local_file = file_path + TRAIN_LABELS
  train_labels = extract_labels(local_file)

  local_file = file_path + TEST_IMAGES
  test_images = extract_images(local_file)

  local_file = file_path + TEST_LABELS
  test_labels = extract_labels(local_file)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  train = DataSet(train_images, train_labels, dtype=dtype)
  validation = DataSet(validation_images, validation_labels, dtype=dtype)
  test = DataSet(test_images, test_labels, dtype=dtype)

  return Datasets(train=train, validation=validation, test=test)


def read_labels(filename):
  print('Extracting', filename)
  with gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj = f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %(magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype = numpy.uint8)
  print('Extracting labels done!')
  return labels

#################################################################################

def _images_distortion(batch_images, batch_size, batch_label, sess, channels = 1):
  temp_batch_images = tf.reshape(batch_images, [batch_size, 28, 28, channels])
  temp_batch_labels = tf.reshape(batch_label, [batch_size, 10, 1])
  #print('label: ', temp_batch_labels.eval(feed_dict = None, session = sess))
  #print(batch_label)
  for n in range(batch_size):
    contrast_image = tf.image.random_contrast(temp_batch_images.eval(feed_dict = None, session = sess)[n], lower = 0., upper = 1.8)
    #contrast_image = tf.image.random_contrast(temp_batch_images, lower = 0., upper = 1.8)

    #print(contrast_image.get_shape())
    tensor_contrast_image = tf.reshape(contrast_image.eval(feed_dict = None, session = sess), [1, 28, 28, channels])
    temp_batch_images = tf.concat(0, [temp_batch_images, tensor_contrast_image])


    ttt_batch_labels = tf.reshape(temp_batch_labels.eval(feed_dict = None, session = sess)[n], [1,10,1])
    temp_batch_labels = tf.concat(0, [temp_batch_labels, ttt_batch_labels])


    flip_image = tf.image.flip_left_right(temp_batch_images.eval(feed_dict = None, session = sess)[n])
    
    tensor_flip_image = tf.reshape(flip_image, [1, 28, 28, channels])
    temp_batch_images = tf.concat(0, [temp_batch_images, tensor_flip_image])

    #print(result_batch_images.eval(feed_dict = None, session = sess)) 
    ttt_batch_labels = tf.reshape(temp_batch_labels.eval(feed_dict = None, session = sess)[n], [1,10,1])

    #print(ttt_batch_labels)
    temp_batch_labels = tf.concat(0, [temp_batch_labels, ttt_batch_labels])

  temp_batch_images = tf.reshape(temp_batch_images, [-1, 784*channels])
  temp_batch_labels = tf.reshape(temp_batch_labels, [-1, 10])


  return temp_batch_images.eval(None, sess), temp_batch_labels.eval(None, sess), batch_size*3

#################################################################################

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

def inference(data, x, y, keep_prob, sess):
  # Network Parameters
  
  # define weights and biases
  def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)
  
  def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

  def conv2d(x, W, strides=1):
    return tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
  
  def max_pool(x, name, ksize=2, strides=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding='SAME', name = name)

  ### first convolutional layer
  ###### conv1_1
  W_conv1_1 = weight_variable([3, 3, 1, 64], 'w_conv1_1')
  b_conv1_1 = bias_variable([64], 'b_conv1_1')
  x_image = tf.reshape(x, [-1, 224, 224, 1])
  image_summary_x = tf.image_summary('input_image', x_image, 10)
  rconv1_1 = conv2d(x_image, W_conv1_1)
  h_conv1_1 = tf.nn.relu(rconv1_1 + b_conv1_1, name = 'h_conv1_1')
  _activation_summary(h_conv1_1)

  ###### conv1_2
  W_conv1_2 = weight_variable([3, 3, 64, 64], 'w_conv1_2')
  b_conv1_2 = bias_variable([64], 'b_conv1_2')
  rconv1_2 = conv2d(h_conv1_1, W_conv1_2)
  h_conv1_2 = tf.nn.relu(rconv1_2 + b_conv1_2, name = 'h_conv1_2')
  _activation_summary(h_conv1_2)

  ###### pool 1
  h_pool1 = max_pool(h_conv1_2, 'h_pool1')


  ### second convolutional layer
  ###### conv2_1
  W_conv2_1 = weight_variable([3, 3, 64, 128], 'w_conv2_1')
  b_conv2_1 = bias_variable([128], 'b_conv2_1')
  h_conv2_1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1) + b_conv2_1, name = 'h_conv2_1')
  
  ######add summary
  _activation_summary(h_conv2_1)

  ###### conv2_2 
  W_conv2_2 = weight_variable([3, 3, 128, 128], 'w_conv2_2')
  b_conv2_2 = bias_variable([128], 'b_conv2_2')
  h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2, name = 'h_conv2_2')

  ###### pool 2
  h_pool2 = max_pool(h_conv2_2, 'h_pool2')


  ### third convolutional layer
  ###### conv3_1
  W_conv3_1 = weight_variable([3, 3, 128, 256], 'w_conv3_1')
  b_conv3_1 = bias_variable([256], 'b_conv3_1')
  h_conv3_1 = tf.nn.relu(conv2d(h_pool2, W_conv3_1) + b_conv3_1, name = 'h_conv3_1')
  
  ###### conv3_2
  W_conv3_2 = weight_variable([3, 3, 256, 256], 'w_conv3_2')
  b_conv3_2 = bias_variable([256], 'b_conv3_2')
  h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, W_conv3_2) + b_conv3_2, name = 'h_conv3_2')

  ###### conv3_3
  W_conv3_3 = weight_variable([3, 3, 256, 256], 'w_conv3_3')
  b_conv3_3 = bias_variable([256], 'b_conv3_3')
  h_conv3_3 = tf.nn.relu(conv2d(h_conv3_2, W_conv3_3) + b_conv3_3, name = 'h_conv3_3')

  ###### pool 3
  h_pool3 = max_pool(h_conv3_3, 'h_pool3')

  ###### add summary
  _activation_summary(h_pool3)


  ### fourth convolutional layer
  ###### conv4_1
  W_conv4_1 = weight_variable([3, 3, 256, 512], 'w_conv4_1')
  b_conv4_1 = bias_variable([512], 'b_conv4_1')
  h_conv4_1 = tf.nn.relu(conv2d(h_pool3, W_conv4_1) + b_conv4_1, name = 'h_conv4_1')

  ###### conv4_2
  W_conv4_2 = weight_variable([3, 3, 512, 512], 'w_conv4_2')
  b_conv4_2 = bias_variable([512], 'b_conv4_2')
  h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1, W_conv4_2) + b_conv4_2, name = 'h_conv4_2')

  ###### conv4_3
  W_conv4_3 = weight_variable([3, 3, 512, 512], 'w_conv4_3')
  b_conv4_3 = bias_variable([512], 'b_conv4_3')
  h_conv4_3 = tf.nn.relu(conv2d(h_conv4_2, W_conv4_3) + b_conv4_3, name = 'h_conv4_3')

  ###### pool 4
  h_pool4 = max_pool(h_conv4_3, 'h_pool4')


  ### fifth convolutional layer
  ###### conv5_1
  W_conv5_1 = weight_variable([3, 3, 512, 512], 'w_conv5_1')
  b_conv5_1 = bias_variable([512], 'b_conv5_1')
  h_conv5_1 = tf.nn.relu(conv2d(h_pool4, W_conv5_1) + b_conv5_1, name = 'h_conv5_1')

  ###### conv5_2
  W_conv5_2 = weight_variable([3, 3, 512, 512], 'w_conv5_2')
  b_conv5_2 = bias_variable([512], 'b_conv5_2')
  h_conv5_2 = tf.nn.relu(conv2d(h_conv5_1, W_conv5_2) + b_conv5_2, name = 'h_conv5_2')

  ###### conv5_3
  W_conv5_3 = weight_variable([3, 3, 512, 512], 'w_conv5_3')
  b_conv5_3 = bias_variable([512], 'b_conv5_3')
  h_conv5_3 = tf.nn.relu(conv2d(h_conv5_2, W_conv5_3) + b_conv5_3, name = 'h_conv5_3')

  ###### pool 5
  h_pool5 = max_pool(h_conv5_3, 'h_pool5')


  ### fully-connected layer
  ###### fc1
  shape = int(numpy.prod(h_pool5.get_shape()[1:]))
  print('h_pool5 ', h_pool5.get_shape())
  print('shape ', shape)
  print('orig_shape ', h_pool5.get_shape()[1:])
  W_fc1 = weight_variable([shape, 4096], 'w_fc1')
  b_fc1 = bias_variable([4096], 'b_fc1')
  h_pool2_flat = tf.reshape(h_pool5, [-1, shape])
  print('flat ', h_pool2_flat)
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name = 'h_fc1')
  ###### add summary
  _activation_summary(h_fc1)

  ###### fc2
  W_fc2 = weight_variable([4096, 4096], 'w_fc2')
  b_fc2 = bias_variable([4096], 'b_fc2')
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2, name = 'h_fc2')

  ###### fc3
  W_fc3 = weight_variable([4096, 10], 'w_fc3')
  b_fc3 = bias_variable([10], 'b_fc3')
  h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3, name = 'h_fc3')
  
  print('h_fc1 ', h_fc1)

  #W_fc1_plus = weight_variable([625, 1024], 'w_fc1_plus')
  #b_fc1_plus = bias_variable([1024], 'b_fc1_plus')
  #h_fc1_plus = tf.nn.relu(tf.matmul(h_fc1, W_fc1_plus) + b_fc1_plus, name = 'h_fc1_plus')
  #_activation_summary(h_fc1_plus)

  ### dropout
  #keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
  #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  ### Readout layer: add a softmax layer
  #W_fc2 = weight_variable([625, 10], 'w_fc2')
  #b_fc2 = bias_variable([10], 'b_fc2')

  #logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  #_activation_summary(logits)

  
  return h_fc3,rconv1_1 


def CNNtrain(data):
  learning_rate = 0.001
  training_iters = 20000
  batch_size = 40
  display_step = 10 
  
  n_input = 50176 # MNIST data input (img shape: 28*28)
  n_classes = 10 # MNIST total classes (0-9 digits)
  dropout = 0.75 # Dropout, probability to keep units
  
  sess = tf.Session()
  ######################################################################
  # tf Graph input
  x = tf.placeholder(tf.float32, [None, n_input], name='pixel-input')
  y = tf.placeholder(tf.float32, [None, n_classes], name='label-input')
  keep_prob = tf.placeholder(tf.float32, name = 'keep_prob') #dropout (keep probability)
  
  logits,h_pool2 = inference(data, x, y, keep_prob, sess)


  tf.add_to_collection("logits", logits)


  global_step = tf.Variable(0, trainable=False)
  
  print('logits ', logits)

  y_conv=tf.nn.softmax(logits)
  print('test y_conv: ', y_conv)
  ######add summary
  _activation_summary(y_conv)

  tf.add_to_collection("y_conv", y_conv)


  ### train and evaluate the model
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]))
  print('cross_entropy ', y*tf.log(y_conv)) 
  

  lr = tf.train.exponential_decay(learning_rate, global_step, 250,LEARNING_RATE_DECAY_FACTOR, staircase = True)

 
  train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
  tf.add_to_collection("train_step", train_step)  

  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
  tf.add_to_collection("correct_prediction", correct_prediction)

  ###test
  model_predict = tf.argmax(y_conv,1)
  tf.add_to_collection("predict", model_predict)
  ###test end
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.add_to_collection("accuracy", accuracy)  

  accuracy_summary = tf.scalar_summary("accuracy", accuracy)


  ### create a saver
  saver = tf.train.Saver(tf.all_variables())

  ### build the summary operation based on the TF collection of summaries
  summary_op = tf.merge_all_summaries()

  ### initialize vars
  sess.run(tf.initialize_all_variables())

  summary_writer = tf.train.SummaryWriter("/home/chunbo/research/procedure/9_6_2016/data/test1", sess.graph)

  step = 0
  tf.train.write_graph(sess.graph_def, "/home/chunbo/research/procedure/9_6_2016/data/", "test.pb", False)
  # print(sess.graph_def)


  ######
  t_batch_size = 40
  ######

  # Keep training until reach max iterations
  ###step * batch_size < training_iters
  ###step*batch_size < 10000
  while step * batch_size < training_iters:
    batch_x, batch_y = data.train.next_batch(batch_size)
    #print('test1 ', batch_y.shape)
    # batch_x, batch_y, ts = _images_distortion(batch_x, batch_size, batch_y, sess, 1)
    # Run optimization op (backprop)
    #print('batch_x: ', batch_x)
    _,test_yconv = sess.run([train_step,y_conv], feed_dict={x:batch_x, y: batch_y, keep_prob: 1.0})
    #print('size:', batch_x.shape)
    #print('y_conv: ', test_yconv)
    print('step: ', step)
    #print('batch_x: ', batch_x.shape())
    #if step % display_step == 0:

      # Calculate batch loss and accuracy
      #loss, acc = sess.run([cross_entropy, accuracy], feed_dict = {x:batch_x, y:batch_y, keep_prob:1.0})
      #print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    if (step+1) % display_step == 0:
      _,summary_str,lr_show = sess.run([train_step,summary_op,lr], feed_dict = {x:batch_x, y:batch_y, keep_prob:1.0})
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()
      # summary_writer.add_summary(image_summary, step)
      saver.save(sess, 'model.ckpt', step)
      print("learning rate: " + str(lr_show))
    step += 1
  print ("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
  print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: data.test.images, y: data.test.labels,keep_prob: 1.0}))
  # tf.train.write_graph(sess.graph_def, "/home/chunbosong/Documents/test/myCNNtest2_mnist/savedGraph", "test.pb", False)
  # e_meta_graph_def = tf.train.export_meta_graph(filename = '/home/chunbo/research/procedure/9_6_2016/data/my-model.meta')
  summary_writer.close()
  sess.close()


 

def main(_):
  my_data = read_data_sets('/home/chunbo/research/procedure/9_6_2016/data/')
  
  # TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'

  # TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'

  # data = extract_images('/home/chunbosong/data/data/' + TRAIN_IMAGES)

  # label = extract_labels('/home/chunbosong/data/data/' + TRAIN_LABELS, 10)

  # print(my_data.train._labels[0])
  #test_labels = read_labels('/home/chunbosong/data/data/t10k-labels-idx1-ubyte.gz')

  CNNtrain(my_data)
   

if __name__ == '__main__':
  tf.app.run()
