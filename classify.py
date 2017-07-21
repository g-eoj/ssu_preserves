import pylab
import sys
import os
import tensorflow as tf
from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance


# location of the labels used in the retrained model
labels = "./tmp/output_updated_timestamp_labels.txt"
# location of the graph for the retrained model
model = "./tmp/output_updated_timestamp_graph.pb"

# change this as you see fit
# image_path = sys.argv[1] # uncomment to provide path from terminal
image_path = './classify_training/bobcat/Cam3_EK000257.JPG'  # to run from pycharm
image_path_equalize = './classify_training/equalize_distorted.JPG'
image_path_mirror = './classify_training/mirror_distorted.JPG'
image_path_blur = './classify_training/blur_distorted.JPG'
image_path_rotate = './classify_training/rotate_distorted.JPG'


original = Image.open(image_path)

equalize_image = ImageOps.equalize(original)
equalize_image.save(image_path_equalize)

image = Image.open(image_path_equalize)

mirror_image = ImageOps.mirror(image)
mirror_image.save(image_path_mirror)

rotate_image = image.rotate(-10)
width, height = rotate_image.size
rotate_image_crop = ImageOps.crop(rotate_image, height / 8)
rotate_image_crop.save(image_path_rotate)

blur_image = ImageOps.box_blur(image, 3)
blur_image.save(image_path_blur)



# Read in the image_data
image_data_original = tf.gfile.FastGFile(image_path, 'rb').read()
image_data_equalize = tf.gfile.FastGFile(image_path_equalize, 'rb').read()
image_data_mirror = tf.gfile.FastGFile(image_path_mirror, 'rb').read()
image_data_blur = tf.gfile.FastGFile(image_path_blur, 'rb').read()
image_data_rotate = tf.gfile.FastGFile(image_path_rotate, 'rb').read()
image_data_list = [image_data_equalize]



score_dict= {}
# Loads label file for the retrained model, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile(labels)]
for name in label_lines:
    score_dict[name] = 0

i = 0
for image_data in image_data_list:
    i+=1
    # Unpersists graph from file
    with tf.gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor,
                 {'DecodeJpeg/contents:0': image_data})
        print(predictions)
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            score_dict[human_string] += score

sorted_score_dict = sorted(score_dict.items(), key=lambda x: x[::-1])
print (sorted_score_dict)
print (score_dict)
for l in reversed(sorted_score_dict):

    print('%s (score = %.5f)' % (l[0], round(l[1]/i, 5)))



# uncomment below to display input image
#img = pylab.imread(image_path)
#pylab.imshow(img)
#pylab.show()

# delete temporary saved images with distortions
os.remove(image_path_mirror)
os.remove(image_path_blur)
os.remove(image_path_rotate)
os.remove(image_path_equalize)


