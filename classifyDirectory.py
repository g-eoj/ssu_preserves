
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


import struct

FLAGS = tf.app.flags.FLAGS

# Input and output file flags.
tf.app.flags.DEFINE_string('image_dir', '/where/are/your/images/',
                           """Path to folders of labeled images.""")
tf.app.flags.DEFINE_string('output_graph', 'tmp/output_graph.pb',
                           """Where to save the trained graph.""")
tf.app.flags.DEFINE_string('output_labels', 'tmp/output_labels.txt',
                           """Where to save the trained graph's labels.""")
tf.app.flags.DEFINE_string('final_tensor_name', 'final_result',
                           """The name of the output classification layer in"""
                           """ the retrained graph.""")

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


def create_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Returns:
    Graph holding the trained network, and various tensors we'll be
    manipulating.
  """
  with tf.Session() as sess:
    model_filename = './~/tmp/output_graph.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor, final_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME, FINAL_LAYER_TENSOR_NAME]))
  return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor, final_tensor



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

def main(_):
  	image_list = create_image_list(FLAGS.image_dir)
	graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor, final_tensor = (create_graph())	
	with tf.Session() as sess:
		for image in image_list:
			filename = os.path.join(FLAGS.image_dir, image)
			image_data = tf.gfile.FastGFile(filename, 'rb').read()
			predictions = sess.run(final_tensor,{jpeg_data_tensor: image_data})
			predictions.sort()
			print(predictions)
	
if __name__ == '__main__':
  tf.app.run()
