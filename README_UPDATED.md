# SHIP 2017 Project 

## Updates to Classify

The following improvements were made to classify.py in order to further validate the model's performance

### Updates to classify.py

classify.py now given an image will apply distortions to it and will make a prediction based on the average
confidence of the distorted and original images.

### Updates to classify_directory.py

classify_directory.py now given an image and its distortions generated using 
retrain_updated_image_augmentation.py will treat distorted images as versions of the original.
The confidence score it has for the original and the distortions will be averaged; this average confidence 
score is used to calculate the model's prediction

## Updates to Retrain

The following improvements were made to retrain.py in order to further validate the model's performance

### retrain_updated.py

retrain.py was updated to tensorflow 1.2.
An exponentially decaying learning rate has been implemented.

### retrain_updated_time_stamp.py

Groups images if they were taken within 2 seconds of each other and places them in the same set(training, 
test, or validation). This way very similar looking images are not placed in different sets which would bias 
the data.
Three text files are now generated which each show the images within their respective image set(training, 
test, or validation). The images are always placed into the same set. The text files are stored within the tmp 
folder under set_contents.

### retrain_updated_k_fold.py

The model now uses k-fold validation. The number of folds can be set to an arbitrary k by editing the hyper-
parameter: '--how_many_folds'.
The confusion matrix is now plotted using matplotlib
A text file for each fold is generated showing the images within that fold. The images are not always placed 
into the same fold. This can be changed by removing the line:
    random.shuffle(list_image_groups). 
The text files are stored within the tmp folder under fold_contents.


### retrain_updated_image_augmentation.py

This model with k-fold validation will non-randomly augment images if the hyper-parameter: '--augment_images' 
is set to True. Run the augmentation once as it will create all the distorted images and bottlenecks. On 
subsequent runs set '--augment_images' to False and choose the distorted images you want to include by editing the if statement in 
create_image_lists which currently prevents all distorted images from being included in the folds.

### retrain_updated_model_v3.py

The current model was replaced with a different frozen version of Inception v3. The implementation is not yet 
complete and the model does not create bottlenecks


