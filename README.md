# CS 385 Project 

### Link to manually classified image set

https://drive.google.com/open?id=0B-AmqiOmDIQER0lObEo1QWJPRjQ

### Setup

Install [miniconda](http://conda.pydata.org/miniconda.html).

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


#### Transfer learning with the Inception-v3 model:

Put the training images in folders named with the image class label. 
Each class needs at least 20 images.
Then put these folders in the folder "training_images" so the directory structure looks something like:

    ssu_preserves    
    └── training_images
        ├── bobcat
        ├── deer
        ├── human
        ├── nothing
        ├── possum
        ├── skunk
        ├── squirrel
        └── turkey

Finally run the line below, on the first run it will download the Inception-v3 model:

    python retrain.py

Before the model starts retraining it needs to create 'bottlenecks' for each image. 
'bottlenecks' are cached so subsequent runs will not do this. 
Every run the images are randomly split into three groups: 90% training, 10% validation, 10% testing.

While the model is retraining it will print periodic updates: 

    2016-12-07 00:21:09.566693: Step 19000: Train accuracy = 100.0%
    2016-12-07 00:21:09.566764: Step 19000: Cross entropy = 0.059450
    2016-12-07 00:21:09.591180: Step 19000: Validation accuracy = 90.0%

Generally you want 'Validation accuracy' to stay similar to 'Train accuray' and 'Cross entropy' to have a decreasing trend (allowing for some noise). 
If this isn't happening try changing one or more of the following parameters in retrain.py: 'train_batch_size' or 'learning_rate'. 

If 'Cross entropy' is still decreasing when retraining ends you may want to increase the value of 'how_many_training_steps'.

When it's done retraining, images from the testing group are used to get final accuracy and a confusion matrix, which are printed along with a parameter summary:

    Final test accuracy = 96.2%
    [[16  0  0  0  0  1]
     [ 0  9  1  1  0  0]
     [ 0  0 14  0  0  0]
     [ 0  0  0 12  0  0]
     [ 0  0  0  0 15  0]
     [ 0  0  0  0  0 11]]
    dict_keys(['squirrel', 'possum', 'deer', 'nothing', 'skunk', 'turkey'])
    Training Steps: 20000
    Learning Rate: 0.001
    Train Batch Size: 20
    Validation Batch Size: 20
    Test Batch Size: 100

You can then run `./scripts/tensorboard` (make sure the script is pointed to your log dir correctly though) to see some of the results in tensorboard.

`retrain.py` will create a log directory based on the paramaters supplied at the top of the file. Currently those params are set to `./tmp/<outputs>` for various `<outputs>`.

`./scripts/tensorboard` will simply run `tensorboard` in the current directory with `./tmp/` as the log directory.

The retrained model files will be saved as "tmp/output_graph.pb" and "tmp/output_labels.txt"


#### Classify an image with a retrained model: 

Assuming a retrained model is available, to classify an image run:

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

The model used for classification can be changed in the classify.py script by changing the "labels" and "model" variables. 

### Running a test on a directory of images
Set up the flags at the top of ClassifyDirectory.py.
With a directory of images set up in a similar manner as that required for `retrain.py`, eg
```
    image_dir
    ├── aaanone
    ├── animal
    ├── deer
    └── squirrel
```
Run the script `ClassifyDirectory.py` by typing `python ClassifyDirectory.py`.
This will classify all images in the directory and check them against their labels. The folder names MUST be the same as the folder names used when retraining the network (with `retrain.py`). They also should be all lower-case.
Misclassified images will be output to the directory specified in the flags at the top of the file. Bottlenecks will be created and stored in a file called `bottlenecks.p` in the same directory as the test images. The script will also produce `ROC.png` and `confusion_matrix.png`, rendered images of ROC curves and a confidence matrix.
