
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import glob
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


import struct

FLAGS = tf.app.flags.FLAGS
#/home/student/cburke/cam4sample'
#/home/student/cburke/tfstuff/images/ssu/Camera3Faraway/output'
#
# Input and output file flags.
tf.app.flags.DEFINE_string('image_dir', '/home/student/cburke/cam4sample',
                           """Path to folders of labeled images.""")
tf.app.flags.DEFINE_string('output_graph', './tmp/output_graph.pb',
                           """Where to save the trained graph.""")
tf.app.flags.DEFINE_string('output_labels', './tmp/output_labels.txt',
                           """Where to save the trained graph's labels.""")
tf.app.flags.DEFINE_string('final_tensor_name', 'final_result',
                           """The name of the output classification layer in"""
                           """ the retrained graph.""")
tf.app.flags.DEFINE_string('summaries_dir', './tmp/testingLogs',
                          """Where to save summary logs for TensorBoard.""")

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
FINAL_LAYER_TENSOR_NAME = 'final_result:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_graph(class_count):
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Returns:
    Graph holding the trained network, and various tensors we'll be
    manipulating.
  """
  with tf.Session() as sess:
    model_filename = './tmp/output_graph.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor, final_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME, FINAL_LAYER_TENSOR_NAME]))
  ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')
                                        
  bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder')

  return sess.graph, bottleneck_input, jpeg_data_tensor, resized_input_tensor, final_tensor, ground_truth_input



def create_image_list(image_dir):
  """Extracts images from the given directory
  """
  if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None
  result = []

  extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
  file_list = []
  dir_name = image_dir
  print("Looking for images in '" + dir_name + "'")
  for extension in extensions:
    file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
    file_list.extend(glob.glob(file_glob))
  if not file_list:
    print('No files found')
    return None
	
  for file_name in file_list:
    base_name = os.path.basename(file_name)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put an image in, the data set creator has a way of
    # grouping photos that are close variations of each other. For example
    # this is used in the plant disease data set to group multiple pictures of
    # the same leaf.
    hash_name = re.sub(r'_nohash_.*$', '', file_name)
    result.append(base_name)
    
  return result
  
  
def getClassNames():
	class_names = [];
	with open(FLAGS.output_labels, 'rb') as f:
		for line in f:
			class_names.append(line.rstrip())
	return class_names
			
  
def showClassifications():
  	image_list = create_image_list(FLAGS.image_dir)
	graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor, final_tensor, ground_truth = (create_graph())	
	with tf.Session() as sess:
		for image in image_list:
			filename = os.path.join(FLAGS.image_dir, image)
			image_data = tf.gfile.FastGFile(filename, 'rb').read()
			predictions = sess.run(final_tensor,{jpeg_data_tensor: image_data})
			predictions.sort()
			print(predictions)
	return predictions

def get_image_and_ground_truth(class_names):
	full_image = []
	full_class = []
	for index in range(len(class_names)):
		imageClass = class_names[index];
		where = os.path.join(FLAGS.image_dir, imageClass);
		image_list = create_image_list(where);
		for i in image_list:
			full_image.append(os.path.join(where, i));
			temp = np.zeros(len(class_names), dtype=np.float32);
			temp[index] = 1.0;
			temp = np.transpose(temp)
			full_class.append(temp);
	return (full_image, full_class)

def getPredictions(result_tensor):
	a = tf.argmax(result_tensor, 1);
	#b = tf.argmax(ground_truth_tensor, 1);
	#correct_prediction = tf.equal(a, b);
	#tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
	return a;
	
def compare_them(predictions, ground_truth_tensor):
	num_correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(ground_truth_tensor, 1));
	return num_correct;
	
def get_image_data(image_list):
	image_data = [];
	for index in range(len(image_list)):
		image_name = image_list[index];
		#print("getting ", image_name);
		image_data.append(tf.gfile.FastGFile(image_name, 'rb').read())
	return image_data;
	
def process_image_step(sess, image_list, bottleneck_tensor, jpeg_data_tensor):
	print("running images through network...");
	processed = [];
	for image in image_list:
		image_data = gfile.FastGFile(image, 'rb').read()
		bottleneck_values = sess.run(
			bottleneck_tensor,
			{jpeg_data_tensor: image_data})
		bottleneck_values = np.squeeze(bottleneck_values)
		processed.append(bottleneck_values);
	print("completed image preprocessing...");
	return processed;

"""
Got this from here:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
Makes a pretty plot of a confusion matrix
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 3),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def writeOutLogs():
	merged = tf.summary.merge_all()
	test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

def main(_):
	class_list = getClassNames();
	(image_list, ground_truth) = get_image_and_ground_truth(class_list);
	#for i in range(len(image_list)):
		#print(i, image_list[i], ground_truth[i])
	graph, bottleneck_input, jpeg_data_tensor,resized_image_tensor, final_tensor,ground_truth_tensor = (create_graph(len(class_list)))
	image_data = get_image_data(image_list)
	with tf.Session() as sess:
		#just manually crunch through all the images and predict what they are
		prediction_list = []
		for i in range(len(image_data)):
			prediction = sess.run(
							getPredictions(final_tensor),
							feed_dict={jpeg_data_tensor: image_data[i]})
			prediction_list.append(prediction);
		prediction_list = np.squeeze(prediction_list);
		#now get the truth from the ground_truth_tensor
		truth_op = tf.argmax(ground_truth_tensor, 1);
		truth = sess.run(truth_op, feed_dict={ground_truth_tensor: ground_truth});
		print ("predictions", prediction_list);
		print ("truth      ", truth);
		eq = np.equal(prediction_list, truth)
		results = np.mean(eq);
		conf_mat = confusion_matrix(prediction_list, truth);
		print ("results", results);
		print ("confusion matrix:");
		print(conf_mat);
		plt.figure()
		plot_confusion_matrix(conf_mat, classes=class_list, normalize=True,
                      title='Normalized confusion matrix')
        plt.show()

if __name__ == '__main__':
  tf.app.run()
