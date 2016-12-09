# CS 385 Project 

### Setup

Install [miniconda](http://conda.pydata.org/miniconda.html)

Create a conda environment:

    conda create --name tf python=3

Activate the environment:

    source activate tf

Install packages:

    conda install scikit-learn
    conda install matplotlib

Install TensorFlow on Mac:
 
    pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0-py3-none-any.whl

Install TensorFlow on Linux:

    pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp35-cp35m-linux_x86_64.whl

If you need a different version of TensorFlow check the [list of builds.](https://www.tensorflow.org/versions/master/get_started/os_setup.html)

### Usage

Activate the conda environment:

    source activate tf


To do transfer learning with the Inception-v3 model:

To retrain the inceptionV3 network, set the params correctly in retrain.py.
Put the training images in folders named with the image class label. 
Then put these folders in the folder "training_images".
Finally run the line below, on the first run it will download the Inception-v3 model:

    python retrain.py

This will print out a lot, including a confusion matrix at the end
You can then run `./scripts/tensorboard` (make sure the script is pointed to your log dir correctly though) to see some of the results in tensorboard.
The retrained model files will be saved as "tmp/output_graph.pb" and "tmp/output_labels.txt"


To classify an image: 

    python classify.py /path/to/image

This will assign the image a score for each class the model knows about and display the image with matplotib:

    $ python classify.py images/deer/EK000052.JPG
    W tensorflow/core/framework/op_def_util.cc:332] Op BatchNormWithGlobalNormalization is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
    deer (score = 0.99342)
    squirrel (score = 0.00253)
    nothing (score = 0.00154)
    possum (score = 0.00133)
    turkey (score = 0.00092)
    skunk (score = 0.00026)

The model can be changed in the classify.py script by changing the "labels" and "model" variables. 
