# CS 385 Project 

### Setup

Install [miniconda](http://conda.pydata.org/miniconda.html)

Create a conda environment:

    conda create --name tf python=3

Activate the environment:

    source activate tf

Install packages:

    conda install scikit-learn
    # conda install jupyter matplotlib (not needed currently)

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

This will print out a lot, including a confusion matrix at the end.

You can then run `./scripts/tensorboard` (make sure the script is pointed to your log dir correctly though) to see some of the results in tensorboard.

`retrain.py` will create a log directory based on the paramaters supplied at the top of the file. Currently those params are set to `./tmp/<outputs>` for various `<outputs>`.

`./scripts/tensorboard` will simply run `tensorboard` in the current directory with `./tmp/` as the log directory.

The retrained model files will be saved as "tmp/output_graph.pb" and "tmp/output_labels.txt"


To classify an image: 

Put the retrained model files in the "retrained_models" folder.
The path to the model files can be changed in the classify.py script by changing the "labels" and "model" variables. 

    python classify.py /path/to/image

This will print out the top-5 predictions for the specified model.
