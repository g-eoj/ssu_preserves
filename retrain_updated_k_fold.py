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
"""Simple transfer learning with an Inception v3 architecture model.

With support for TensorBoard.

This example shows how to take a Inception v3 architecture model trained on
ImageNet images, and train a new top layer that can recognize other classes of
images.

The top layer receives as input a 2048-dimensional vector for each image. We
train a softmax layer on top of this representation. Assuming the softmax layer
contains N labels, this corresponds to learning N + 2048*N model parameters
corresponding to the learned biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. Once your images are
prepared, you can run the training with a command like this:


```bash
bazel build tensorflow/examples/image_retraining:retrain && \
bazel-bin/tensorflow/examples/image_retraining/retrain \
    --image_dir ~/flower_photos
```

Or, if you have a pip installation of tensorflow, `retrain.py` can be run
without bazel:

```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos
```

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the label_image sample code.


To use with TensorBoard:

By default, this script will log summaries to ./tmp/retrain_updated_kfold_logs directory

Visualize the summaries with this command:

tensorboard --logdir ./tmp/retrain_updated_kfold_logs

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import argparse
from datetime import datetime
import math
import os.path
import random
import re
import struct
import sys
import tarfile
import exifread as EXIF
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import itertools
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None

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
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M




def create_image_lists(image_dir):
    """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  folds, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.


  Returns:
    A list of dictionaries containing an entry for each label subfolder, with images split
    into training and testing sets within each label.
  """
    # Get number of folds
    folds = FLAGS.how_many_folds
    # Create result_list which will contain as many dictionaries as there are folds
    # Each dictionary differs in which fold is assigned as the test fold
    result_list = []
    # Create a dictionary which will contain a list of images for each fold
    # Sole purpose is to generate text files that state what is in each fold
    fold_dict_all = {}

    l = 1
    while l <= folds:
        # Append k empty dictionaries to result_list
        result_list.append({})
        # Append k empty lists to fold_dict
        fold_dict_all[l] = []
        l += 1

    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print('WARNING: Folder {} has more than {} images. Some images will '
                  'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        # Separate into image groups based on capture event timestamp
        list_image_groups = []  # list of image groups   example:[[img1,img2,img3],[img4,img5],[img6,img7],...]
        not_first_run = False  # boolean to indicate if it isn't the first iteration of the loop
        temp_list = []  # temp list that will contain current image group. It resets once the image group is added to the list of image groups

        for file_name in file_list:  # get one image out of list of all images in that folder

            is_mirror_applied = "mirror" in file_name
            is_blur_applied = "blur" in file_name
            is_brightness_applied = "brightness" in file_name
            is_rotation_applied = "rotate" in file_name
            is_crop_applied = "crop" in file_name
            #Distortions added below will NOT be added to folds
            if is_crop_applied == False and is_brightness_applied == False and is_blur_applied == False and is_mirror_applied == False and is_rotation_applied == False:

                with open(file_name, 'rb') as fh:  # get the capture date
                    tags = EXIF.process_file(fh, stop_tag="EXIF DateTimeOriginal")
                    dateTaken = tags["EXIF DateTimeOriginal"]

                new = datetime.strptime(str(dateTaken), "%Y:%m:%d %H:%M:%S")  # assign new the image's timestamp

                if not_first_run:
                    difference = abs(
                        new - old)  # Calculate difference between current image and previous image timestamp

                    if difference.seconds <= FLAGS.time_stamp_difference:  # if difference is less than 2 seconds than we add it to the same image group
                        temp_list.append(os.path.basename(file_name))

                    else:  # if it is more than 2 seconds than we add temp_list to list_image_groups and create a new temp list with the current image
                        list_image_groups.append(temp_list)
                        temp_list = []
                        temp_list.append(os.path.basename(file_name))

                else:  # if first run just append image to temp list. Cannot find difference between first image and previous image since there was no previous image
                    temp_list.append(os.path.basename(file_name))

                old = datetime.strptime(str(dateTaken), "%Y:%m:%d %H:%M:%S")  # Set current image timestamp to old
                not_first_run = True  # Change boolean to indicate that it is no longer the first iteration

        list_image_groups.append(
            temp_list)  # add the final temp list to list_image_groups since it will not be added otherwise

        '''Uncomment below loop and comment above loop to stop separating based on capture event
        for file_name in file_list:
            temp_list.append(os.path.basename(file_name))
            list_image_groups.append(temp_list)
            temp_list = []'''

        # Shuffle the list of image groups to make each run random
        random.shuffle(list_image_groups)

        # Creates a dictionary with k number of folds which will store what image groups are within each fold
        fold_dict = {}
        y = 1
        while y <= folds:
            fold_dict[y] = []  # create blank list for each fold
            y += 1

        # Gets an image group from list_image_groups and puts it into the smallest fold in fold_dict
        # This way all the folds will have almost an equal number of folds
        for value in list_image_groups:
            smallest = min(fold_dict, key=lambda k: len(fold_dict[k]))
            fold_dict[smallest].extend(value)
            fold_dict_all[smallest].extend(value)

        # Prints out number of images per fold
        p = 1
        while p <= folds:
            print("Fold " + str(p) + ":", len(fold_dict[p]))
            p += 1

        # Creates result_list: a list containing k dictionaries each differing in which fold is the test set
        t = 1
        for x in result_list:  # Get one dictionary at a time
            training_folds = []
            for i in fold_dict.keys():
                if t != i:
                    training_folds.extend(fold_dict[i])  # Add all but one fold to training
            x[label_name] = {  # Add training and testing to dictionary
                'dir': dir_name,
                'training': training_folds,
                'testing': fold_dict[t]
            }
            t += 1

    # Creates text files detailing images in each fold
    m = 1
    while m <= len(fold_dict):
        save_path = '/Users/snadendla/software/ssu_preserves/tmp/fold_contents'
        name_of_file = 'fold' + str(m) + '_set.txt'
        completeName = os.path.join(save_path, name_of_file)
        file_v = open(completeName, 'w')
        file_v.write("Fold " + str(m) + " Set:")
        file_v.write(str(fold_dict_all[m]))
        file_v.close()
        m += 1

    return result_list


def get_image_path(image_lists, label_name, index, image_dir, category):
    """"Returns a path to an image for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category):
    """"Returns a path to a bottleneck file for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.
  """
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category) + '.txt'


def create_inception_graph():
    """"Creates a graph from saved GraphDef file and returns a Graph object.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
    with tf.Graph().as_default() as graph:
        model_filename = os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def maybe_download_and_extract():
    """Download and extract model tar file.

  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.
  """
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                                 filepath,
                                                 _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def write_list_of_floats_to_file(list_of_floats, file_path):
    """Writes a given list of floats to a binary file.

  Args:
    list_of_floats: List of floats we want to write to a file.
    file_path: Path to a file where list of floats will be stored.

  """

    s = struct.pack('d' * BOTTLENECK_TENSOR_SIZE, *list_of_floats)
    with open(file_path, 'wb') as f:
        f.write(s)


def read_list_of_floats_from_file(file_path):
    """Reads list of floats from a given file.

  Args:
    file_path: Path to a file where list of floats was stored.
  Returns:
    Array of bottleneck values (list of floats).

  """

    with open(file_path, 'rb') as f:
        s = struct.unpack('d' * BOTTLENECK_TENSOR_SIZE, f.read())
        return list(s)


bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           bottleneck_tensor):
    """Create a single bottleneck file."""
    print('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    except:
        raise RuntimeError('Error during processing file %s' % image_path)

    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             bottleneck_tensor):
    """Retrieves or calculates bottleneck values for an image.

  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string  of the subfolders containing the training
    images.
    category: Name string of which  set to pull images from - training, testing,
    or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    bottleneck_tensor: The output tensor for the bottleneck values.

  Returns:
    Numpy array of values produced by the bottleneck layer for the image.
  """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          bottleneck_dir, category)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        print('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):
    """Ensures all the training, testing, and validation bottlenecks are cached.

  Because we're likely to read the same image multiple times (if there are no
  distortions applied during training) it can speed things up a lot if we
  calculate the bottleneck layer values once for each image during
  preprocessing, and then just read those cached values repeatedly during
  training. Here we go through all the images we've found, calculate those
  values, and save them off.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    bottleneck_tensor: The penultimate output layer of the graph.

  Returns:
    Nothing.
  """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index,
                                         image_dir, category, bottleneck_dir,
                                         jpeg_data_tensor, bottleneck_tensor)

                how_many_bottlenecks += 1
                # if how_many_bottlenecks % 100 == 0:
                # print(str(how_many_bottlenecks) + ' bottleneck files created.')
                # Uncomment above two lines to print how_many_bottlenecks


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  bottleneck_tensor):
    """Retrieves bottleneck values for cached images.

  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.

  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: If positive, a random sample of this size will be chosen.
    If negative, all bottlenecks will be retrieved.
    category: Name string of which set to pull from - training, testing, or
    validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    image_dir: Root folder string of the subfolders containing the training
    images.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.

  Returns:
    List of bottleneck arrays, their corresponding ground truths, and the
    relevant filenames.
  """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                        image_dir, category)
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                                  image_index, image_dir, category,
                                                  bottleneck_dir, jpeg_data_tensor,
                                                  bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                                      image_index, image_dir, category,
                                                      bottleneck_dir, jpeg_data_tensor,
                                                      bottleneck_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(
        sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
        distorted_image, resized_input_tensor, bottleneck_tensor):
    """Retrieves bottleneck values for training images, after distortions.

  If we're training with distortions like crops, scales, or flips, we have to
  recalculate the full model for every image, and so we can't use cached
  bottleneck values. Instead we find random images for the requested category,
  run them through the distortion graph, and then the full graph to get the
  bottleneck results for each.

  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: The integer number of bottleneck values to return.
    category: Name string of which set of images to fetch - training, testing,
    or validation.
    image_dir: Root folder string of the subfolders containing the training
    images.
    input_jpeg_tensor: The input layer we feed the image data to.
    distorted_image: The output node of the distortion graph.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.

  Returns:
    List of bottleneck arrays and their corresponding ground truths.
  """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                    category)
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()
        # Note that we materialize the distorted_image_data as a numpy array before
        # sending running inference on the image. This involves 2 memory copies and
        # might be optimized in other implementations.
        distorted_image_data = sess.run(distorted_image,
                                        {input_jpeg_tensor: jpeg_data})
        bottleneck = run_bottleneck_on_image(sess, distorted_image_data,
                                             resized_input_tensor,
                                             bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """Whether any distortions are enabled, from the input flags.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.

  Returns:
    Boolean value indicating whether any distortions should be applied.
  """
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
            (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """Creates the operations to apply the specified distortions.

  During training it can help to improve the results if we run the images
  through simple distortions like crops, scales, and flips. These reflect the
  kind of variations we expect in the real world, and so can help train the
  model to cope with natural data more effectively. Here we take the supplied
  parameters and construct a network of operations to apply them to an image.

  Cropping
  ~~~~~~~~

  Cropping is done by placing a bounding box at a random position in the full
  image. The cropping parameter controls the size of that box relative to the
  input image. If it's zero, then the box is the same size as the input and no
  cropping is performed. If the value is 50%, then the crop box will be half the
  width and height of the input. In a diagram it looks like this:

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  Scaling
  ~~~~~~~

  Scaling is a lot like cropping, except that the bounding box is always
  centered and its size varies randomly within the given range. For example if
  the scale percentage is zero, then the bounding box is the same size as the
  input and no scaling is applied. If it's 50%, then the bounding box will be in
  a random range between half the width and height and full size.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.
    graph.

  Returns:
    The jpeg input layer and the distorted result tensor.
  """

    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
    precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d,
                                   [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
                                    MODEL_INPUT_DEPTH])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=brightness_min,
                                         maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
    """Adds a new softmax and fully-connected layer for traini0ng.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
    recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(tf.float32)
        tf.summary.scalar('learning rate', lr)

    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32,
                                            [None, class_count],
                                            name='GroundTruthInput')

    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count],
                                                stddev=0.001)

            layer_weights = tf.Variable(initial_value, name='final_weights')

            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        # optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        # train_step = optimizer.minimize(cross_entropy_mean)
        train_step = tf.train.AdamOptimizer(lr).minimize(
            cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor, lr)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Tuple of (evaluation step, prediction).
  """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def save_graph_to_file(sess, graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return


def prepare_file_system():
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
    return


def add_confusion_matrix_step(result_tensor, ground_truth_input):
    with tf.name_scope('accuracy'):
        with tf.name_scope('confusion_matrix'):
            confusion = tf.contrib.metrics.confusion_matrix(tf.argmax(result_tensor, 1),
                                                            tf.argmax(ground_truth_input, 1))
            # print(tf.argmax(ground_truth_input, 1))
            # tf.image_summary('confusion matrix', confusion)
        return confusion


def plot_confusion_matrix(
    cm,
    classes = ['bobcat', 'deer', 'human', 'nothing', 'possum', 'skunk', 'squirrel', 'turkey'],
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues,
    ):
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
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for (i, j) in itertools.product(range(cm.shape[0]),
                                    range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 3), horizontalalignment='center'
                 , color=('white' if cm[i, j] > thresh else 'black'))

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', format='png')





def main(_):
    average_accuracy = []
    current_fold_num = 0


    # Look at the folder structure, and create list containing dictionaries that differ in which fold was assigned as test
    list_of_image_lists = create_image_lists(FLAGS.image_dir)

    # initialize blank 8*8 matrix that will contain a total confusion matrix for all the folds
    num_classes = len(list_of_image_lists[1].keys())
    confusion_all = [[0 for j in range(num_classes)] for i in range(num_classes)]
    # Get one dicitonary at a time and run that session
    for image_lists in list_of_image_lists:
        current_fold_num += 1
        # Prepare necessary directories  that can be used during training
        prepare_file_system()

        # Set up the pre-trained graph.
        maybe_download_and_extract()
        graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
            create_inception_graph())

        # See if the command-line flags mean we're applying any distortions.
        do_distort_images = should_distort_images(
            FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
            FLAGS.random_brightness)

        with tf.Session(graph=graph) as sess:

            if do_distort_images:
                # We will be applying distortions, so setup the operations we'll need.
                (distorted_jpeg_data_tensor,
                 distorted_image_tensor) = add_input_distortions(
                    FLAGS.flip_left_right, FLAGS.random_crop,
                    FLAGS.random_scale, FLAGS.random_brightness)
            else:
                # We'll make sure we've calculated the 'bottleneck' image summaries and
                # cached them on disk.
                cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                                  FLAGS.bottleneck_dir, jpeg_data_tensor,
                                  bottleneck_tensor)

            # Add the new layer that we'll be training.
            (train_step, cross_entropy, bottleneck_input, ground_truth_input,
             final_tensor, lr) = add_final_training_ops(len(image_lists.keys()),
                                                        FLAGS.final_tensor_name,
                                                        bottleneck_tensor)

            # Create the operations we need to evaluate the accuracy of our new layer.
            evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)
            confusion_step = add_confusion_matrix_step(final_tensor, ground_truth_input)

            # Merge all the summaries and write them out to the summaries_dir
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                                 sess.graph)

            validation_writer = tf.summary.FileWriter(
                FLAGS.summaries_dir + '/validation')

            # Set up all our weights to their initial default values.
            init = tf.global_variables_initializer()
            sess.run(init)

            # Run the training for as many cycles as requested on the command line.
            for i in range(FLAGS.how_many_training_steps):
                # Get a batch of input bottleneck values, either calculated fresh every
                # time with distortions applied, or from the cache stored on disk.
                if do_distort_images:
                    (train_bottlenecks,
                     train_ground_truth) = get_random_distorted_bottlenecks(
                        sess, image_lists, FLAGS.train_batch_size, 'training',
                        FLAGS.image_dir, distorted_jpeg_data_tensor,
                        distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
                else:
                    (train_bottlenecks,
                     train_ground_truth, _) = get_random_cached_bottlenecks(
                        sess, image_lists, FLAGS.train_batch_size, 'training',
                        FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                        bottleneck_tensor)
                # Learning rate calculation
                max_learning_rate = 0.001
                min_learning_rate = 0.0001
                decay_speed = 2000.0  # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
                learning_rate_decayed = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(
                    -i / decay_speed)
                # Feed the bottlenecks and ground truth into the graph, and run a training
                # step. Capture training summaries for TensorBoard with the `merged` op.
                train_summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth,
                               lr: learning_rate_decayed})
                train_writer.add_summary(train_summary, i)

                # Store intermediate results
                intermediate_frequency = FLAGS.intermediate_store_frequency

                if intermediate_frequency > 0 and (i % intermediate_frequency == 0) and i > 0:
                    intermediate_file_name = FLAGS.intermediate_output_graphs_dir + 'intermediate_' + str(i) + '.pb'
                    print('Save intermediate result to : ' + intermediate_file_name)
                    save_graph_to_file(sess, graph, intermediate_file_name)

            # We've completed all our training, so run a final test evaluation on
            # some new images we haven't used before.
            test_bottlenecks, test_ground_truth, test_filenames = (
                get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size,
                                              'testing', FLAGS.bottleneck_dir,
                                              FLAGS.image_dir, jpeg_data_tensor,
                                              bottleneck_tensor))
            test_accuracy, predictions = sess.run(
                [evaluation_step, prediction],
                feed_dict={bottleneck_input: test_bottlenecks,
                           ground_truth_input: test_ground_truth})

            average_accuracy.append(
                test_accuracy * 100)  # Add test accuracy to array which will at the end compute average fold test accuracy
            print('\n', 'Fold:', current_fold_num)
            print('Test accuracy = %.1f%% (N=%d)' % (
            test_accuracy * 100, len(test_bottlenecks)))  # Print test accuracy for current fold

            confusion = sess.run(confusion_step,
                                 feed_dict={bottleneck_input: test_bottlenecks,
                                            ground_truth_input: test_ground_truth})  # generate confusion matrix for current fold
            confusion_all += confusion  # add current confusion matrix to cumulative confusion matrix
            if FLAGS.print_misclassified_test_images:
                print('=== MISCLASSIFIED TEST IMAGES ===')  # Print the misclassified images for current fold
                for i, test_filename in enumerate(test_filenames):
                    if predictions[i] != test_ground_truth[i].argmax():
                        print('%70s  %s' % (test_filename,
                                            list(image_lists.keys())[predictions[i]]))

            save_graph_to_file(sess, graph, FLAGS.output_graph)
            with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
                f.write('\n'.join(image_lists.keys()) + '\n')
    print("\n")
    print("Average Accuracy:", str(round(np.mean(average_accuracy),
                                         2)))  # Print final test accuracy- an average of all the individual fold accuracies
    print("Standard Deviation:", round(np.std(average_accuracy), 3))
    print("\n")
    plot_confusion_matrix(confusion_all)  # Plot and print confusion matrix using matplotlib



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='./training_images/',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='./tmp/output_updated_kfold_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--intermediate_output_graphs_dir',
        type=str,
        default='./tmp/intermediate_updated_kfold_graph/',
        help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=0,
        help='How many steps to store intermediate graph. If "0" then will not store.'
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='./tmp/output_updated_kfold_labels.txt',
        help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='./tmp/retrain_updated_kfold_logs',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=100,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--how_many_folds',
        type=int,
        default=5,
        help='How many folds to use in k-fold validation'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )

    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=5,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=50,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
    )

    parser.add_argument(
        '--time_stamp_difference',
        type=int,
        default=2,
        help='Group images within this many seconds together'
    )

    parser.add_argument(
        '--print_misclassified_test_images',
        default=True,
        help="""\
      Whether to print out a list of all misclassified test images.\
      """,
        action='store_true'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='./tmp/imagenet',
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='./tmp/bottleneck',
        help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\
      The name of the output classification layer in the retrained graph.\
      """
    )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
        action='store_true'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
